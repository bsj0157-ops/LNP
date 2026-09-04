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

import numpy as np
import pandas as pd
import streamlit as st

import lnp_entry as LE
import lnp_predictor_v3_patched as v3
import lnp_app_patch as P
import lnp_optimize as O
import app_tab_peg as TP

# 💡 새로운 모듈 통합
import lnp_app_fix2 as F2
import lnp_store as ST
import lnp_autoharvest as AH
import app_tab_harvest as TH
import lnp_app_cache as C
import lnp_app_guard as GD
import app_tab_anchor2 as T2
import app_tabs_offset as TO

# 💡 실시간 정확도 노트 및 가벼운 모델, 불확실성 모듈 임포트
import lnp_app_livenote as LN
import lnp_model_v7 as M7          # v6 -> v7 (원 스케일 HistGB, clipping 없음)

# lnp_pdf 는 여기서만 임포트합니다. 위에서 무조건 임포트하면 pdfplumber 가 없을 때
# 이 try 에 닿기 전에 앱 전체가 죽어서 PDF_OK 안내가 뜨지 않습니다.
try:
    import lnp_pdf as LP
    PDF_OK, PDF_ERR = True, ""
except Exception as _e:
    PDF_OK, PDF_ERR = False, str(_e)

st.set_page_config(page_title="LNP Data Studio", page_icon="🧪", layout="wide")

DATA_FILE = "lnp_web_data.csv"

# 데이터가 바뀌어도 변하지 않는 해석 주의사항만 남깁니다.
# 수치(MAE·ICC·적중률)는 F2.ACCURACY_NOTE 에 박아 두지 말고 아래 버튼으로 실측합니다.
ACCURACY_CAVEATS = """**읽는 법**

- 앵커링 탭이 띄우는 MAE 는 **학습에 쓴 행을 다시 예측한 값**입니다. 새 논문 정확도가 아닙니다.
- EE 변동의 상당 부분이 조성이 아니라 '어느 논문인지'에서 옵니다(ICC 는 아래에서 실측).
- 앵커링 효과는 논문에 따라 갈립니다. 원래 크게 틀리는 논문에서만 이득이 관찰됐습니다.
- 비율만 바꾼 what-if 방향 판단은 무작위 수준이었습니다. 순위·방향 결정에 쓰지 마십시오.
"""

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

BACKUP_FILE = "lnp_web_data_backup.csv"

def save_disk(backup: bool = False):
    """저장 실패를 삼키지 않습니다. 실패하면 메모리에만 남았다고 알립니다."""
    if backup:
        try:
            prev = store.load()
            if prev is not None and len(prev):
                prev.to_csv(BACKUP_FILE, index=False, encoding="utf-8-sig")
        except Exception as e:
            st.warning(f"백업 실패 — 교체를 중단하는 편이 안전합니다: {e}")
            return False
    try:
        store.save(st.session_state.df)
        return True
    except Exception as e:
        st.error(f"저장 실패 ({store.describe()}): {e}\n"
                 "화면의 데이터는 이 세션에만 있습니다. 사이드바에서 내려받아 두십시오.")
        return False

def add_rows(new_rows: pd.DataFrame):
    res = GD.screen_new_rows(new_rows, st.session_state.df, reduce_fn=AH.reduce_to_four)
    
    for m in res["messages"]:
        st.info(m)
        
    if len(res["rejected"]):
        st.caption("⚠️ 아래 행은 추가되지 않았습니다.")
        st.dataframe(res["rejected"][["why"] + [c for c in new_rows.columns if c in res["rejected"].columns]])
        
    if not res["accepted"].empty:
        st.session_state.df = pd.concat([st.session_state.df, res["accepted"]], ignore_index=True)
        if save_disk():
            st.success(f"{len(res['accepted'])}행 추가 및 저장 완료.")
        st.rerun()

# --------------------------------------------------------------------------
# 캐시 시스템 설치 및 실행
# --------------------------------------------------------------------------
cached = C.install(st, F2, v3, P)
work_df, work_info = cached["working_df"](st.session_state.df)

def get_oof():
    """논문 단위 CV 예측. 첫 계산에 수십 초 걸리므로 **쓰는 곳에서만** 부릅니다.

    이전 판은 모듈 스코프에서 계산해, PDF 탭만 쓰려고 접속한 사람도
    데이터가 바뀔 때마다 전체 CV 를 기다려야 했습니다.
    """
    return cached["oof"](work_df)

cached_model = M7.make_cached_v7_model(st, v3)

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

with st.sidebar.expander("예측 정확도 (실측)"):
    st.markdown(ACCURACY_CAVEATS)
    if st.button("현재 데이터로 계산", key="acc_note"):
        st.markdown(LN.accuracy_note(work_df, get_oof(), base_note=""))
    else:
        st.caption("숫자는 논문 단위 CV 를 돌려야 나옵니다(수십 초). "
                   "F2.ACCURACY_NOTE 의 고정 수치는 옛 데이터·옛 모델 값이라 뺐습니다.")

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
# 탭 3 — 데이터 관리 (💡 엄격한 양식 검사 로직 적용 완료)
# ==========================================================================
with tab_data:
    st.header("데이터 관리")
    F2.show_data_consistency(st, work_df, st.session_state.df)
    
    st.info("💡 CSV 업로드 시 데이터베이스 양식(컬럼명)과 완벽히 일치해야 덮어쓰기가 가능합니다. (좌측 사이드바의 '원본 전체 내려받기' 양식 참고)")
    up2 = st.file_uploader("기존 CSV 불러오기 (표준 형식 엄격 검사)", type=["csv"], key="csvup")
    
    if up2 is not None:
        raw = up2.read()
        d_raw, parse_err = None, None
        # latin-1 은 어떤 바이트든 통과시키므로 후보에서 뺐습니다. 그대로 두면
        # cp949/utf-16 파일이 '성공'으로 읽히고 DOI 가 깨진 채 저장됩니다.
        # DOI 는 논문 단위 CV 의 그룹 키라서, 깨지면 같은 논문이 여러 논문으로 갈립니다.
        for enc in ("utf-8-sig", "utf-8", "cp949", "utf-16"):
            try:
                cand = pd.read_csv(io.BytesIO(raw), encoding=enc)
            except Exception as e:
                parse_err = e
                continue
            if "reference_doi" in cand.columns:
                s = cand["reference_doi"].astype(str)
                # DOI 는 ASCII 입니다. 비ASCII 가 섞였으면 인코딩을 잘못 잡은 것입니다.
                if s.str.contains(r"[^\x00-\x7f]").mean() > 0.05:
                    parse_err = f"{enc} 로 읽었으나 DOI 에 깨진 문자가 섞였습니다"
                    continue
            d_raw = cand
            break

        if d_raw is None:
            st.error(f"CSV 를 읽지 못했습니다: {parse_err}\n"
                     "엑셀이면 '다른 이름으로 저장 > CSV UTF-8' 로 저장해 주세요.")
        else:
            # 💡 [핵심 패치] 양식 엄격 검사 로직
            expected_cols = list(st.session_state.df.columns)
            missing_cols = [col for col in expected_cols if col not in d_raw.columns]
            
            if missing_cols:
                st.error("🚨 **양식 불일치 에러!** 업로드한 CSV 파일이 표준 양식과 맞지 않아 데이터를 덮어쓸 수 없습니다.")
                st.warning(f"**누락된 필수 컬럼 ({len(missing_cols)}개):**\n" + ", ".join(missing_cols))
                st.caption("해결 방법: 좌측 사이드바에서 '원본 전체 내려받기'를 클릭하여 최신 양식을 확인하신 후, 동일한 컬럼 구조로 맞추어 다시 업로드해 주세요.")
            else:
                # 양식이 완벽히 일치할 때만 정상 처리 진행
                d_clean = pd.DataFrame(columns=expected_cols)
                for col in expected_cols:
                    d_clean[col] = d_raw[col]
                
                ee_col = "encapsulation_efficiency_percent_std_num"
                d_clean[ee_col] = d_clean[ee_col].map(P.robust_ee)
                
                num_cols = ["np_ratio_std_num", "buffer_ph_std_num", "particle_size_nm_std_num", "pdi_std_num", "zeta_potential_mv_std_num"]
                for col in num_cols:
                    if col in d_clean.columns: d_clean[col] = pd.to_numeric(d_clean[col], errors='coerce')

                st.write("---")
                st.success("✅ CSV 양식 검사 통과! 데이터가 표준 양식과 완벽히 일치합니다.")
                st.subheader("가져온 데이터 미리보기 (자동 정제됨)")
                st.dataframe(d_clean.head(5))
                
                n_invalid = d_clean[ee_col].isna().sum()
                if n_invalid > 0: st.warning(f"⚠️ 경고: {n_invalid}개 행은 EE 수치가 없어서 제외됩니다.")
                    
                c1, c2 = st.columns(2)
                if c1.button(f"정상 데이터 {len(d_clean) - n_invalid}행 추가하기"):
                    add_rows(d_clean.dropna(subset=[ee_col]))
                    
                # 교체는 되돌릴 수 없습니다 — 백업과 확인 입력을 요구합니다.
                valid_d = d_clean.dropna(subset=[ee_col])
                cur_n = len(st.session_state.df)
                c2.caption(f"교체하면 현재 {cur_n}행이 {len(valid_d)}행으로 **바뀝니다**. "
                           f"{max(0, cur_n - len(valid_d))}행이 사라질 수 있습니다.")
                confirm = c2.text_input("교체를 원하면 REPLACE 를 입력하세요", key="confirm_replace")
                if c2.button(f"🔄 {len(valid_d)}행으로 교체", type="primary",
                             disabled=(confirm.strip().upper() != "REPLACE")):
                    if len(valid_d) == 0:
                        st.error("유효한 데이터(EE 수치 포함)가 없어 교체할 수 없습니다.")
                    else:
                        prev = st.session_state.df.copy()
                        st.session_state.df = valid_d.copy().reset_index(drop=True)
                        if save_disk(backup=True):
                            st.success(f"✅ {len(valid_d)}행으로 교체했습니다. "
                                       f"이전 데이터는 {BACKUP_FILE} 에 남겼습니다.")
                            st.rerun()
                        else:
                            st.session_state.df = prev      # 저장 실패 시 되돌립니다

    st.divider()
    st.subheader("📝 엑셀에서 바로 복사/붙여넣기")
    if len(st.session_state.df) == 0:
        st.info("아직 데이터가 없습니다.")
    else:
        ed = st.data_editor(st.session_state.df, num_rows="dynamic", use_container_width=True, key="main_edit", height=380)
        # 이 버튼은 DB 전체를 심사 통과 행으로 **교체**합니다. 이전 판은 기각 행과
        # 몰비가 빈 행을 말없이 버리고 저장까지 해서, 표에서 셀 하나만 고쳐도
        # 심사에 걸린 행이 영구히 사라졌습니다. 무엇이 빠지는지 먼저 보여줍니다.
        if st.button("편집 내용 심사 (아직 저장 안 함)", key="screen_edit"):
            has_ratio = (ed["lipid_molar_ratio"].notna() if "lipid_molar_ratio" in ed.columns
                         else pd.Series(True, index=ed.index))
            st.session_state["edit_screen"] = {
                "res": GD.screen_new_rows(ed[has_ratio], _empty(), reduce_fn=AH.reduce_to_four),
                "n_no_ratio": int((~has_ratio).sum()),
                "n_in": len(ed),
            }

        scr = st.session_state.get("edit_screen")
        if scr:
            res, acc = scr["res"], scr["res"]["accepted"]
            n_drop = scr["n_in"] - len(acc)
            for m in res["messages"]:
                st.info(m)
            if scr["n_no_ratio"]:
                st.warning(f"몰비가 빈 {scr['n_no_ratio']}행은 심사 대상에서 빠집니다.")
            if len(res["rejected"]):
                st.error(f"기각 {len(res['rejected'])}행 — 저장하면 사라집니다.")
                st.dataframe(res["rejected"][["why"] + [c for c in ed.columns
                                                        if c in res["rejected"].columns]])
            st.caption(f"편집 {scr['n_in']}행 -> 통과 {len(acc)}행 (**{n_drop}행 감소**)")
            k1, k2 = st.columns(2)
            confirm_e = k1.text_input("저장하려면 SAVE 를 입력하세요", key="confirm_edit")
            if k1.button(f"통과 {len(acc)}행으로 저장", type="primary",
                         disabled=(confirm_e.strip().upper() != "SAVE" or acc.empty)):
                prev = st.session_state.df.copy()
                st.session_state.df = acc.reset_index(drop=True)
                if save_disk(backup=True):
                    st.session_state.pop("edit_screen", None)
                    st.success(f"✅ {len(acc)}행 저장. 이전 데이터는 {BACKUP_FILE} 에 있습니다.")
                    st.rerun()
                else:
                    st.session_state.df = prev
            if k2.button("취소", key="cancel_edit"):
                st.session_state.pop("edit_screen", None)
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
        from scipy.stats import spearmanr
        
        with st.spinner("논문 단위 CV 진행 중…"):
            rep = M7.cv_report(work_df, v3)

        c1, c2, c3 = st.columns(3)
        c1.metric("모델 MAE", f"{rep['mae_model']:.2f} %p")
        c2.metric("baseline MAE", f"{rep['mae_baseline']:.2f} %p")
        c3.metric("개선율", f"{rep['gain_pct']:+.1f} %", delta=f"{'유의미' if rep['gain_pct'] > 5 else '미미'}")

        st.caption("💡 " + rep["picks"][0] + " — v7 은 타깃 변환·clipping 이 없어 "
                   "EE 90 % 이상도 예측할 수 있습니다(v6 은 구조적으로 불가).")
        st.caption(f"트리 산포 -> 오차 환산 계수 실측 {rep['scale_hat']:.2f} "
                   f"(lnp_optimize.UNCERTAINTY_SCALE 를 이 값으로 맞추십시오)")

        icc = F2.icc1(work_df)
        st.write(f"**논문 간 분산 비중(ICC) = {icc:.2f}**")

        groups = rep["groups"]
        pm = rep["pred"]
        y = rep["y"]
        
        rhos = [spearmanr(y.loc[idx], pd.Series(pm, index=y.index).loc[idx])[0] 
                for gid, idx in y.groupby(groups).groups.items() if len(idx) >= 4]
        rhos = [r for r in rhos if not np.isnan(r)]
        
        if rhos:
            st.success(f"**논문 내 순위 상관 중앙값 rho = {np.median(rhos):.2f}** ({sum(r > 0 for r in rhos)}/{len(rhos)}편에서 양수)")

        res = pd.DataFrame({"실측 EE": y, "예측 EE": pm, "논문": groups})
        st.scatter_chart(res, x="실측 EE", y="예측 EE")

    st.divider()
    # 앵커링 패널은 논문 단위 CV 예측(oof)이 필요합니다. 페이지를 열 때마다
    # 자동으로 돌지 않게 사용자가 열 때만 계산합니다.
    if st.checkbox("앵커링 패널 열기 (논문 단위 CV 계산, 수십 초)", key="open_anchor"):
        T2.render(st, work_df, v3, F2, oof_series=get_oof())

# ==========================================================================
# 탭 5, 6, 7 — 영점 전파를 위한 감싸개(Wrapper) 적용
# ==========================================================================
with tab_opt:
    if len(work_df) > 10:
        base_model = cached_model(work_df)
        st.caption(base_model.transform_note)
        TO.tab_optimize_anchored(st, work_df, base_model, v3, O)
    else:
        st.warning("🚨 데이터가 너무 적습니다. '데이터 관리' 탭에서 데이터를 더 추가해주세요.")

with tab_what:
    if len(work_df) > 10:
        base_model = cached_model(work_df)
        st.caption(base_model.transform_note)
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
