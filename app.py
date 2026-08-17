# ==========================================================================
#  LNP Data Studio  —  웹으로 논문 데이터를 입력하고 모델을 돌리는 앱
#  ------------------------------------------------------------------------
#  실행:  streamlit run app.py
#  필요:  streamlit pandas numpy scikit-learn pdfplumber (rdkit 선택)
#         같은 폴더에 lnp_entry.py, lnp_pdf.py 가 있어야 합니다.
#
#  탭 구성
#    1. PDF 업로드   — 논문 PDF에서 후보를 뽑아 표로 보여주고 확인 후 추가
#    2. 직접 입력    — 폼으로 한 줄씩 입력 (SMILES 자동 조회)
#    3. 데이터 관리  — 전체 표 편집, 검증, CSV 업로드/다운로드
#    4. 모델 실행    — 논문 단위 CV + 진단 + 순위 평가
#    5. 최적화       — 앵커 영점 기반 최적 레시피 탐색
#    6. What-If      — 특정 성분 비율 변경에 따른 효과 시뮬레이션
#    7. PEG 비율 변경 — PEG 변경에 따른 신뢰할 수 있는 구간 예측
# ==========================================================================

import io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

import lnp_harvest
import lnp_pdf
import lnp_entry as LE
import lnp_predictor_v3_patched as v3
import lnp_anchor
import lnp_app_patch as P
import lnp_optimize as O
from app_tabs_optimize import tab_optimize, tab_whatif
import lnp_peg as PG
from app_tab_peg import tab_peg

# 💡 새로운 픽스 모듈 & 저장소 모듈 불러오기
import lnp_app_fix2 as F2
import lnp_store as ST

try:
    import lnp_pdf as LP
    PDF_OK = True
except Exception as _e:
    PDF_OK = False
    PDF_ERR = str(_e)

st.set_page_config(page_title="LNP Data Studio", page_icon="🧪", layout="wide")

DATA_FILE = "lnp_web_data.csv"

# --------------------------------------------------------------------------
# 상태 및 데이터 저장 (Google Sheets 연동 지원)
# --------------------------------------------------------------------------
def _empty():
    return pd.DataFrame(columns=LE.COLS)

store = ST.get_store(st)

# 💡 [패치 A] 통합된 초기화 로직: Store 로드를 최우선으로, 없으면 로컬 백업 호출
if "df" not in st.session_state:
    loaded = store.load()
    if loaded is None and os.path.exists(DATA_FILE):
        try:
            loaded = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        except:
            loaded = None
    st.session_state.df = loaded if loaded is not None else _empty()

def save_disk():
    store.save(st.session_state.df)

def add_rows(new: pd.DataFrame):
    cur = st.session_state.df
    out = pd.concat([cur, new], ignore_index=True)
    head = [c for c in LE.COLS if c in out.columns]
    st.session_state.df = out[head + [c for c in out.columns if c not in head]]
    save_disk()

# 💡 모든 탭이 같은 데이터를 쓰도록 work_df 생성
work_df, work_info = F2.get_working_df(st.session_state.df, patch_mod=P)

# 💡 초고속 렌더링을 위한 모델 캐싱
cached_model = F2.make_cached_base_model(st, v3)

# --------------------------------------------------------------------------
# 사이드바 — 현황
# --------------------------------------------------------------------------
st.sidebar.title("🧪 LNP Data Studio")

ST.show_store_status(st, store)
F2.show_persistence_warning(st.sidebar)

n_rows = len(work_df)
n_pap = (work_df["reference_doi"].astype(str).str.strip().str.lower().nunique()
         if n_rows and "reference_doi" in work_df else 0)
st.sidebar.metric("수집한 처방", f"{n_rows} 행")
st.sidebar.metric("논문 수 (실질 표본)", f"{n_pap} 편")

need = max(0, 20 - n_pap)
st.sidebar.progress(min(n_pap / 20, 1.0))
if need:
    st.sidebar.caption(f"논문 단위 CV 권장선(20편)까지 **{need}편** 남았습니다.")
else:
    st.sidebar.caption("논문 수가 충분합니다. 모델 실행 탭으로 가세요.")

st.sidebar.divider()
st.sidebar.caption(
    "**왜 논문 수인가**\n\n같은 논문의 처방들은 같은 랩·같은 프로토콜이라 "
    "서로 닮아 있습니다. 무작위로 나누면 성능이 부풀려지므로 논문 단위로 "
    "나눠야 하고, 그때 실질 표본 수는 행 수가 아니라 논문 수입니다.")

st.sidebar.markdown(F2.ACCURACY_NOTE)

if len(st.session_state.df):
    st.sidebar.divider()
    # 💡 [패치 F] 다운로드 버튼 2종류로 분리 (정제본 vs 원본)
    buf_work = io.StringIO()
    work_df.to_csv(buf_work, index=False)
    st.sidebar.download_button(f"정제 데이터 내려받기 ({len(work_df)}행)", 
                               buf_work.getvalue().encode("utf-8-sig"),
                               "lnp_data_clean.csv", "text/csv", use_container_width=True)
    
    buf_raw = io.StringIO()
    st.session_state.df.to_csv(buf_raw, index=False)
    st.sidebar.download_button(f"원본 전체 내려받기 ({len(st.session_state.df)}행)", 
                               buf_raw.getvalue().encode("utf-8-sig"),
                               "lnp_data_raw.csv", "text/csv", use_container_width=True)

tab_pdf, tab_form, tab_data, tab_model, tab_opt, tab_what, tab_peg_view = st.tabs(
    ["📄 PDF 업로드", "✍️ 직접 입력", "📊 데이터 관리", "🤖 모델 실행", "🎯 최적화", "⚖️ What-If", "📉 PEG 비율 변경"])

# ==========================================================================
# 탭 1 — PDF 업로드
# ==========================================================================
with tab_pdf:
    st.header("논문 PDF에서 데이터 뽑기")
    st.warning("**자동 입력기가 아니라 초안 작성기입니다.** LNP 논문의 EE 값은 상당수가 그림(막대그래프)에만 있고 본문 텍스트에는 없습니다. 조성 성분 순서도 논문마다 달라서, 뽑아낸 값은 반드시 근거 문장을 보고 확인해야 합니다. 목표는 타이핑을 줄이는 것이지 검토를 없애는 것이 아닙니다.")

    if not PDF_OK:
        st.error(f"PDF 모듈을 못 불러왔습니다: {PDF_ERR}\n\n`pip install pdfplumber pypdf` 후 다시 실행하세요.")
    else:
        up = st.file_uploader("논문 PDF", type=["pdf"], key="pdfup")
        if up:
            with st.spinner("PDF 읽는 중..."):
                try:
                    ex = LP.extract(io.BytesIO(up.read()))
                except Exception as e:
                    st.error(f"추출 실패: {e}")
                    ex = None

            if ex:
                c1, c2, c3 = st.columns(3)
                c1.metric("페이지", ex["n_pages"])
                c2.metric("조성 후보", len(ex["ratios"]))
                c3.metric("EE 후보", len(ex.get("ee_specific", [])))

                doi = st.text_input("DOI (확인/수정)", value=ex["doi"] or "", help="같은 논문에서 나온 행은 모두 이 DOI를 씁니다. 논문 단위 CV의 기준이므로 정확해야 합니다.")
                
                st.subheader("찾은 조성 후보")
                if not ex["ratios"]:
                    st.info("조성을 못 찾았습니다. Methods 나 SI 에서 직접 찾아 '직접 입력' 탭에 넣으세요.")
                for i, r in enumerate(ex["ratios"]):
                    with st.expander(f"{r['ratio']}   (p.{r['page']}, 합계 {r['sum']})" + ("  ⚠️ 확인 필요" if r["chem_warnings"] else ""), expanded=(i == 0)):
                        st.code(r["evidence"], language=None)
                        cc1, cc2 = st.columns(2)
                        cc1.write(f"**논문 표기 그대로:** `{r['ratio_as_written']}`")
                        cc2.write(f"**표준 순서로 변환:** `{r['ratio']}`")
                        for w in r["chem_warnings"]:
                            st.warning(w)

                st.divider()
                st.subheader("표 편집 후 추가")
                draft = LP.to_draft_rows(ex)
                if doi:
                    draft["reference_doi"] = doi
                edited = st.data_editor(draft, num_rows="dynamic", use_container_width=True, key="pdf_edit")

                if st.button("이 행들을 데이터에 추가", type="primary"):
                    keep = edited[edited["lipid_molar_ratio"].astype(str).str.strip() != ""]
                    keep = keep[pd.to_numeric(keep["encapsulation_efficiency_percent_std_num"], errors="coerce").notna()]
                    if keep.empty:
                        st.error("몰비와 EE가 모두 채워진 행이 없습니다.")
                    else:
                        with st.spinner("SMILES 조회 중..."):
                            keep = LE.resolve_smiles(keep, verbose=False)
                        add_rows(keep)
                        st.success(f"{len(keep)}행 추가했습니다.")
                        st.rerun()

# ==========================================================================
# 탭 2 — 직접 입력
# ==========================================================================
with tab_form:
    st.header("직접 입력")
    st.caption("모르는 값은 비워 두세요. 0 이나 'N/A' 로 채우면 모델이 왜곡됩니다.")

    prev_doi = ""
    if len(st.session_state.df) and "reference_doi" in st.session_state.df:
        s = st.session_state.df["reference_doi"].dropna().astype(str)
        prev_doi = s.iloc[-1] if len(s) else ""

    with st.form("entry", clear_on_submit=False):
        st.subheader("필수")
        f1, f2 = st.columns(2)
        doi = f1.text_input("논문 DOI", value=prev_doi, placeholder="10.1038/s41586-021-03534-y")
        ratio = f2.text_input("지질 몰비", placeholder="50:10:38.5:1.5", help="이온화 : 헬퍼 : 콜레스테롤 : PEG 순서")
        g1, g2 = st.columns(2)
        ee = g1.number_input("EE (%)", 0.0, 100.0, 90.0, 0.1)
        ion = g2.text_input("이온화지질 이름", placeholder="SM-102")

        st.subheader("권장")
        h1, h2, h3 = st.columns(3)
        cargo = h1.selectbox("cargo", ["", "mRNA", "siRNA", "saRNA", "pDNA", "ASO", "circRNA"])
        helper = h2.text_input("헬퍼 지질", placeholder="DSPC")
        peg = h3.text_input("PEG 지질", placeholder="DMG-PEG2000")
        i1, i2 = st.columns(2)
        npr = i1.number_input("N/P 비 (0=미기재)", 0.0, 60.0, 0.0, 0.5)
        ph = i2.number_input("완충액 pH (0=미기재)", 0.0, 9.0, 0.0, 0.1)

        ok = st.form_submit_button("추가", type="primary")

    if ok:
        if not doi.strip():
            st.error("DOI가 필요합니다. 논문 단위 CV의 기준입니다.")
        elif not ratio.strip():
            st.error("몰비가 필요합니다.")
        else:
            e = LE.Entry()
            e.paper(doi.strip())
            e.add(ratio.strip(), ee, ion=ion.strip() or None, helper=helper.strip() or None, peg=peg.strip() or None, cargo=cargo or None, np_ratio=npr or None, ph=ph or None)
            new = e.to_frame()
            with st.spinner("SMILES 조회 중..."):
                new = LE.resolve_smiles(new, verbose=False)
            add_rows(new)
            st.success(f"추가했습니다.")

# ==========================================================================
# 탭 3 — 데이터 관리
# ==========================================================================
with tab_data:
    st.header("데이터 관리")
    
    F2.show_data_consistency(st, work_df, st.session_state.df)

    up2 = st.file_uploader("기존 CSV 불러오기 (표준 형식으로 자동 정렬 및 정제)", type=["csv"], key="csvup")
    if up2 is not None:
        raw = up2.read()
        for enc in ("utf-8-sig", "cp949", "latin-1"):
            try:
                d_raw = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except: continue
        
        d_clean = pd.DataFrame(columns=st.session_state.df.columns)
        for col in st.session_state.df.columns:
            if col in d_raw.columns:
                d_clean[col] = d_raw[col]
            else:
                d_clean[col] = np.nan
        
        ee_col = "encapsulation_efficiency_percent_std_num"
        d_clean[ee_col] = d_clean[ee_col].map(P.robust_ee)
        
        num_cols = ["np_ratio_std_num", "buffer_ph_std_num", "particle_size_nm_std_num", "pdi_std_num", "zeta_potential_mv_std_num"]
        for col in num_cols:
            if col in d_clean.columns:
                d_clean[col] = pd.to_numeric(d_clean[col], errors='coerce')

        st.write("---")
        st.subheader("가져온 데이터 미리보기 (자동 정제됨)")
        st.dataframe(d_clean.head(5))
        
        n_invalid = d_clean[ee_col].isna().sum()
        if n_invalid > 0:
            st.warning(f"⚠️ 경고: {n_invalid}개 행은 EE 수치가 없어서 추가에서 제외됩니다.")
            
        c1, c2 = st.columns(2)
        if c1.button(f"정상 데이터 {len(d_clean) - n_invalid}행 추가하기"):
            valid_d = d_clean.dropna(subset=[ee_col])
            valid_d = P.dedupe(valid_d)
            add_rows(valid_d)
            st.success("✅ 중복 제거 후 안전하게 추가되었습니다.")
            st.rerun()
            
        # 💡 [패치 E] 교체 시 전체 데이터 날림 방지
        if c2.button("기존 데이터를 이 파일로 교체"):
            valid_d = P.dedupe(d_clean.dropna(subset=[ee_col]))
            if len(valid_d) == 0:
                st.error("추가할 수 있는 행이 없습니다(EE 수치 없음). 교체하지 않았습니다.")
            elif len(valid_d) < len(st.session_state.df) * 0.5:
                st.warning(f"현재 {len(st.session_state.df)}행 → {len(valid_d)}행으로 절반 이하가 됩니다.")
                if st.checkbox("그래도 위험을 감수하고 교체합니다"):
                    st.session_state.df = valid_d; save_disk(); st.rerun()
            else:
                st.session_state.df = valid_d; save_disk(); st.rerun()

    st.divider()
    st.subheader("📝 엑셀에서 바로 복사/붙여넣기")
    if len(st.session_state.df) == 0:
        st.info("아직 데이터가 없습니다.")
    else:
        ed = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True, key="main_edit", height=380)
        
        b1, b2, b3 = st.columns(3)
        if b1.button("편집 내용 저장", type="primary"):
            if "lipid_molar_ratio" in ed.columns:
                valid_ed = ed.dropna(subset=["lipid_molar_ratio"])
                valid_ed = valid_ed[valid_ed["lipid_molar_ratio"].astype(str).str.strip() != ""]
            else:
                valid_ed = ed
            st.session_state.df = valid_ed
            save_disk()
            st.success("✅ 편집 내용이 성공적으로 저장되었습니다.")
            st.rerun()

# ==========================================================================
# 탭 4 — 모델 실행 및 앵커링
# ==========================================================================
with tab_model:
    st.header("모델 실행")
    st.caption("논문 단위 교차검증으로 평가합니다. 무작위 분할은 성능을 부풀리므로 쓰지 않습니다.")

    if n_pap < 5:
        st.warning(f"논문이 {n_pap}편입니다. 논문 단위 5-fold CV에는 최소 5편 이상이 필요합니다.")

    if st.button("평가 실행", type="primary", disabled=(n_pap < 3)):
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from scipy.stats import spearmanr

        X, y, groups, num_cols, cat_cols = F2.build_eval_matrix(work_df, v3)

        pre = ColumnTransformer([
            ("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_cols),
            ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                            ("o", OneHotEncoder(handle_unknown="ignore", min_frequency=2))]), cat_cols)]
            if cat_cols else
            [("n", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num_cols)])
        
        model = Pipeline([("pre", pre),
                          ("m", RandomForestRegressor(n_estimators=400, min_samples_leaf=3, max_features=0.5, random_state=42, n_jobs=-1))])

        k = min(5, groups.nunique())
        cv = GroupKFold(n_splits=k)
        with st.spinner("교차검증 중..."):
            pm = cross_val_predict(model, X, y, cv=cv, groups=groups)
            pb = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=cv, groups=groups)

        mae_m, mae_b = mean_absolute_error(y, pm), mean_absolute_error(y, pb)
        gain = (mae_b - mae_m) / mae_b * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("모델 MAE", f"{mae_m:.2f} %p")
        c2.metric("baseline MAE", f"{mae_b:.2f} %p")
        c3.metric("개선율", f"{gain:+.1f} %", delta=f"{'유의미' if gain > 5 else '미미'}")

        icc = F2.icc1(work_df)
        st.write(f"**논문 간 분산 비중(ICC) = {icc:.2f}**")

        rhos = [spearmanr(y.loc[idx], pd.Series(pm, index=y.index).loc[idx])[0] 
                for gid, idx in y.groupby(groups).groups.items() if len(idx) >= 4]
        rhos = [r for r in rhos if not np.isnan(r)]
        
        if rhos:
            st.success(f"**논문 내 순위 상관 중앙값 rho = {np.median(rhos):.2f}** ({sum(r > 0 for r in rhos)}/{len(rhos)}편에서 양수)")

        res = pd.DataFrame({"실측 EE": y, "예측 EE": pm, "논문": groups})
        st.scatter_chart(res, x="실측 EE", y="예측 EE")

    # ==========================================
    # ⚓ 앵커링 (실측 영점 조절) 기능 탭
    # ==========================================
    st.divider()
    st.subheader("⚓ 앵커링 (영점 조절) 기반 정밀 예측")
    st.markdown("AI가 추천하는 조성으로 실험하거나, 원하는 조성을 직접 지정하여 영점을 조절합니다.")
    
    # 💡 [패치 C] 앵커 선택 전 논문을 먼저 필터링하도록 수정
    papers = sorted(work_df["reference_doi"].dropna().astype(str).unique())
    sel_paper = st.selectbox("📌 앵커를 고를 기준 논문 선택", ["(선택하세요)"] + papers)
    
    if sel_paper != "(선택하세요)":
        sub_df = work_df[work_df["reference_doi"].astype(str) == sel_paper]
        # sub_df 내부의 행 위치를 이용해 앵커 선택
        anchor_idx_sub, anchor_y = F2.anchor_selector(st, sub_df, n=3, key_prefix="anc_sub")
        # 선택된 sub_df 내 인덱스를 실제 work_df의 전체 인덱스로 변환
        anchor_idx = sub_df.iloc[anchor_idx_sub].index.tolist() if anchor_idx_sub else []
        
        if st.button("영점 조절 후 전체 예측 실행 (다운로드)", type="primary"):
            if anchor_idx and anchor_y and len(anchor_idx) == len(anchor_y):
                with st.spinner("정직한 Out-of-fold 예측 및 앵커링 검증 중... (약 15~20초 소요)"):
                    
                    # 💡 [패치 B, C, D] In-sample 과적합을 배제한 전체 554행 정밀 예측표 생성
                    tab, summ = F2.anchored_full_table(work_df, v3, lnp_anchor, anchor_idx, anchor_y)
                    
                    if summ.get("warning"):
                        st.warning(summ["warning"])
                        
                    st.divider()
                    st.write("### 📊 앵커링 실전 검증 리포트 및 최종 결과표")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("전체 논문 MAE", f"{summ['mae_all']:.1f} %p")
                    if summ.get("mae_anchor_paper") is not None:
                        c2.metric("앵커 논문 (보정 후)", f"{summ['mae_anchor_paper']:.1f} %p")
                        c3.metric("앵커 논문 (보정 전)", f"{summ['mae_anchor_paper_noanchor']:.1f} %p")
                        
                    st.caption(f"영점 보정량 **{summ['offset']:+.1f} %p** — 앵커와 같은 논문의 행에만 적용되었습니다. 다른 논문에 다른 랩의 영점을 가져다 쓸 근거가 없기 때문입니다.")

                    st.dataframe(tab, use_container_width=True, height=520)
                    
                    st.download_button("결과 CSV 전체 내려받기",
                                       tab.to_csv(index=False).encode("utf-8-sig"),
                                       "lnp_anchored_predictions.csv", "text/csv",
                                       use_container_width=True)
            else:
                st.warning("선택된 앵커 수와 실측값 수가 일치하지 않거나 누락되었습니다.")
    else:
        st.info("먼저 앵커를 고를 논문을 선택해주세요.")

# ==========================================================================
# 탭 5 — 🎯 최적화 (캐싱 적용)
# ==========================================================================
with tab_opt:
    if len(work_df) > 10:
        base_model = cached_model(work_df)
        tab_optimize(st, work_df, base_model, v3_module=v3)
    else:
        st.warning("🚨 데이터가 너무 적습니다. '데이터 관리' 탭에서 데이터를 더 추가해주세요.")

# ==========================================================================
# 탭 6 — ⚖️ What-If (캐싱 적용)
# ==========================================================================
with tab_what:
    if len(work_df) > 10:
        base_model = cached_model(work_df)
        tab_whatif(st, work_df, base_model, v3_module=v3)
    else:
        st.warning("🚨 데이터가 너무 적습니다. '데이터 관리' 탭에서 데이터를 더 추가해주세요.")

# ==========================================================================
# 탭 7 — 📉 PEG 비율 변경 
# ==========================================================================
with tab_peg_view:
    if len(work_df) > 10:
        tab_peg(st, work_df)
    else:
        st.warning("🚨 데이터가 너무 적습니다. '데이터 관리' 탭에서 데이터를 더 추가해주세요.")
