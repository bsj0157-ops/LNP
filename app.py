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
#    4. 모델 실행    — 논문 단위 CV + 진단 + 순위 평가 및 앵커링
#    5. 최적화       — 앵커 영점 기반 최적 레시피 탐색 (영점 동기화)
#    6. What-If      — 특정 성분 비율 변경에 따른 효과 시뮬레이션 (영점 동기화)
#    7. PEG 비율 변경 — PEG 변경에 따른 신뢰할 수 있는 구간 예측 (영점 동기화)
#    8. 🤖 자동 수집  — PMC 오픈액세스에서 논문 자동 검색 및 정제 후 추가
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
import app_tab_peg as TP

# 💡 새로운 모듈 통합
import lnp_app_fix2 as F2
import lnp_store as ST
import lnp_autoharvest as AH
import app_tab_harvest as TH
import lnp_app_cache as C
import lnp_app_guard as GD
import lnp_anchor2 as A2
import app_tab_anchor2 as T2
import app_tabs_offset as TO

# 💡 실시간 정확도 노트 및 가벼운 모델, 불확실성 모듈 임포트
import lnp_app_livenote as LN
import lnp_features_lean as FL
import lnp_uncertainty as U
import lnp_model_v4 as M4

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

if "df" not in st.session_state:
    try:
        loaded = store.load()
    except Exception as e:
        st.caption(f"Store load 에러 발생: {e}")
        loaded = None

    if loaded is None and os.path.exists(DATA_FILE):
        try:
            loaded = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        except Exception as e:
            st.caption(f"로컬 파일 읽기 에러 발생: {e}")
            loaded = None

    st.session_state.df = loaded if loaded is not None else _empty()

def save_disk():
    store.save(st.session_state.df)

def add_rows(new_rows: pd.DataFrame):
    res = GD.screen_new_rows(new_rows, st.session_state.df, reduce_fn=AH.reduce_to_four)
    
    for m in res["messages"]:
        st.info(m)
        
    if len(res["rejected"]):
        st.caption("⚠️ 아래 행은 추가되지 않았습니다.")
        st.dataframe(res["rejected"][["why"] + [c for c in new_rows.columns if c in res["rejected"].columns]])
        
    if not res["accepted"].empty:
        st.session_state.df = pd.concat([st.session_state.df, res["accepted"]], ignore_index=True)
        save_disk()
        st.success(f"{len(res['accepted'])}행 추가 및 저장 완료.")
        st.rerun()

# --------------------------------------------------------------------------
# 캐시 시스템 설치 및 실행
# --------------------------------------------------------------------------
cached = C.install(st, F2, v3, P)
work_df, work_info = cached["working_df"](st.session_state.df)
oof = cached["oof"](work_df)

# 💡 [패치] Logit 타깃 + RF/ExtraTrees 앙상블이 적용된 V4 모델로 교체
cached_model = M4.make_cached_v4_model(st, v3)

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

# 💡 실시간 구간별 정확도로 사이드바 동적 반영
st.sidebar.markdown(LN.accuracy_note(work_df, oof, base_note=F2.ACCURACY_NOTE))

if len(st.session_state.df):
    st.sidebar.divider()
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

# 탭 구성
tab_pdf, tab_form, tab_data, tab_model, tab_opt, tab_what, tab_peg_view, tab_harvest = st.tabs(
    ["📄 PDF 업로드", "✍️ 직접 입력", "📊 데이터 관리", "🤖 모델 실행", "🎯 최적화", "⚖️ What-If", "📉 PEG 비율", "🤖 자동 수집"])

# ==========================================================================
# 탭 1 — PDF 업로드
# ==========================================================================
with tab_pdf:
    st.header("논문 PDF에서 데이터 뽑기")
    st.warning("**자동 입력기가 아니라 초안 작성기입니다.** LNP 논문의 EE 값은 상당수가 그림(막대그래프)에만 있고 본문 텍스트에는 없습니다. 조성 성분 순서도 논문마다 달라서, 뽑아낸 값은 반드시 근거 문장을 보고 확인해야 합니다.")

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
                    draft["reference_doi"] = doi.strip().lower()
                    
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
        clean_doi = doi.strip().lower()
        if not clean_doi:
            st.error("DOI가 필요합니다. 논문 단위 CV의 기준입니다.")
        elif not ratio.strip():
            st.error("몰비가 필요합니다.")
        else:
            e = LE.Entry()
            e.paper(clean_doi)
            e.add(ratio.strip(), ee, ion=ion.strip() or None, helper=helper.strip() or None, peg=peg.strip() or None, cargo=cargo or None, np_ratio=npr or None, ph=ph or None)
            new = e.to_frame()
            with st.spinner("SMILES 조회 중..."):
                new = LE.resolve_smiles(new, verbose=False)
            add_rows(new)

# ==========================================================================
# 탭 3 — 데이터 관리
# ==========================================================================
with tab_data:
    st.header("데이터 관리")
    
    F2.show_data_consistency(st, work_df, st.session_state.df)

    up2 = st.file_uploader("기존 CSV 불러오기 (표준 형식으로 자동 정렬 및 정제)", type=["csv"], key="csvup")
    if up2 is not None:
        raw = up2.read()
        d_raw = None
        for enc in ("utf-8-sig", "cp949", "latin-1"):
            try:
                d_raw = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except Exception as e:
                parse_err = e
                continue
                
        if d_raw is None:
            st.error(f"CSV 파싱 에러 발생: {parse_err}")
        else:
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
                add_rows(valid_d)
                
            if c2.button("기존 데이터를 이 파일로 교체"):
                valid_d = d_clean.dropna(subset=[ee_col])
                if len(valid_d) == 0:
                    st.error("추가할 수 있는 행이 없습니다(EE 수치 없음). 교체하지 않았습니다.")
                elif len(valid_d) < len(st.session_state.df) * 0.5:
                    st.warning(f"현재 {len(st.session_state.df)}행 → {len(valid_d)}행으로 절반 이하가 됩니다.")
                    if st.checkbox("그래도 위험을 감수하고 교체합니다"):
                        st.session_state.df = _empty()
                        add_rows(valid_d)
                else:
                    st.session_state.df = _empty()
                    add_rows(valid_d)

    st.divider()
    st.subheader("📝 엑셀에서 바로 복사/붙여넣기")
    if len(st.session_state.df) == 0:
        st.info("아직 데이터가 없습니다.")
    else:
        ed = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True, key="main_edit", height=380)
        
        # 💡 [패치] 편집 내용 저장 시에도 행 축소 경고를 위한 안전 확인 프로세스 추가
        b1, b2, b3 = st.columns(3)
        if b1.button("편집 내용 적용 심사", type="primary"):
            if "lipid_molar_ratio" in ed.columns:
                valid_ed = ed.dropna(subset=["lipid_molar_ratio"])
                valid_ed = valid_ed[valid_ed["lipid_molar_ratio"].astype(str).str.strip() != ""]
            else:
                valid_ed = ed
            
            res = GD.screen_new_rows(valid_ed, _empty(), reduce_fn=AH.reduce_to_four)
            
            st.session_state["temp_edited_res"] = res
            st.rerun()
            
        if "temp_edited_res" in st.session_state:
            res = st.session_state["temp_edited_res"]
            
            if len(res["rejected"]):
                st.warning(f"{len(res['rejected'])}개의 불량 행이 감지되어 제거되었습니다.")
            
            if len(res["accepted"]) < len(st.session_state.df):
                st.warning(f"⚠️ 원본 {len(st.session_state.df)}행에서 {len(res['accepted'])}행으로 크게 줄어듭니다! (중복 제거 등 원인)")
                
                if st.checkbox("이대로 덮어쓰기에 동의합니다."):
                    if st.button("확정 및 저장"):
                        st.session_state.df = res["accepted"]
                        save_disk()
                        del st.session_state["temp_edited_res"]
                        st.success("✅ 편집 내용이 안전하게 덮어씌워졌습니다.")
                        st.rerun()
                else:
                    if st.button("취소 및 초기화"):
                        del st.session_state["temp_edited_res"]
                        st.rerun()
            else:
                st.session_state.df = res["accepted"]
                save_disk()
                del st.session_state["temp_edited_res"]
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
        from sklearn.dummy import DummyRegressor
        from sklearn.metrics import mean_absolute_error
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from scipy.stats import spearmanr

        # 💡 [V4 모델 패치] 이전의 무거운 파이프라인을 걷어내고 V4(Logit + 앙상블)로 교체
        X, y, groups, num_cols, cat_cols = FL.build_lean_matrix(work_df, v3)
        model = M4.LogitEnsemble(num_cols=list(num_cols), cat_cols=list(cat_cols))

        k = min(5, groups.nunique())
        cv = GroupKFold(n_splits=k)
        
        with st.spinner("V4 모델 기반 교차검증 중..."):
            # M4 모델의 predict 함수는 내부적으로 Logit 변환과 역변환을 수행합니다.
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

    st.divider()
    T2.render(st, work_df, v3, F2, oof_series=oof)

# ==========================================================================
# 탭 5, 6, 7 — 영점 전파를 위한 감싸개(Wrapper) 적용
# ==========================================================================
with tab_opt:
    if len(work_df) > 10:
        base_model = cached_model(work_df)
        TO.tab_optimize_anchored(st, work_df, base_model, v3, O)
    else:
        st.warning("🚨 데이터가 너무 적습니다. '데이터 관리' 탭에서 데이터를 더 추가해주세요.")

with tab_what:
    if len(work_df) > 10:
        base_model = cached_model(work_df)
        TO.tab_whatif_anchored(st, work_df, base_model, v3, O)
    else:
        st.warning("🚨 데이터가 너무 적습니다. '데이터 관리' 탭에서 데이터를 더 추가해주세요.")

with tab_peg_view:
    if len(work_df) > 10:
        TO.tab_peg_anchored(st, work_df, TP)
    else:
        st.warning("🚨 데이터가 너무 적습니다. '데이터 관리' 탭에서 데이터를 더 추가해주세요.")

# ==========================================================================
# 탭 8 — 🤖 자동 수집 탭 연결
# ==========================================================================
with tab_harvest:
    TH.render(work_df, on_add=add_rows)
