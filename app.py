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
import lnp_anchor                  # 예전에 저장한 앵커 (파일 이름이 lnp_anchor_2.py라면 lnp_anchor_2 as lnp_anchor 로 적어주세요)
import lnp_app_patch as P  # 👈 [추가!] 패치 모듈 불러오기

try:
    import lnp_pdf as LP
    PDF_OK = True
except Exception as _e:
    PDF_OK = False
    PDF_ERR = str(_e)

st.set_page_config(page_title="LNP Data Studio", page_icon="🧪", layout="wide")

DATA_FILE = "lnp_web_data.csv"


# --------------------------------------------------------------------------
# 상태
# --------------------------------------------------------------------------
def _empty():
    return pd.DataFrame(columns=LE.COLS)


if "df" not in st.session_state:
    if os.path.exists(DATA_FILE):
        st.session_state.df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    else:
        st.session_state.df = _empty()


def save_disk():
    st.session_state.df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


def add_rows(new: pd.DataFrame):
    cur = st.session_state.df
    st.session_state.df = pd.concat([cur, new], ignore_index=True)[
        [c for c in LE.COLS if c in set(LE.COLS)]
        + [c for c in new.columns if c not in LE.COLS]]
    save_disk()


# --------------------------------------------------------------------------
# 사이드바 — 현황
# --------------------------------------------------------------------------
df = st.session_state.df
st.sidebar.title("🧪 LNP Data Studio")
n_rows = len(df)
n_pap = (df["reference_doi"].astype(str).str.strip().str.lower().nunique()
         if n_rows and "reference_doi" in df else 0)
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

if n_rows:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.sidebar.download_button("전체 CSV 내려받기", buf.getvalue().encode("utf-8-sig"),
                               "lnp_data.csv", "text/csv", use_container_width=True)

tab_pdf, tab_form, tab_data, tab_model = st.tabs(
    ["📄 PDF 업로드", "✍️ 직접 입력", "📊 데이터 관리", "🤖 모델 실행"])


# ==========================================================================
# 탭 1 — PDF 업로드
# ==========================================================================
with tab_pdf:
    st.header("논문 PDF에서 데이터 뽑기")
    st.warning(
        "**자동 입력기가 아니라 초안 작성기입니다.** LNP 논문의 EE 값은 상당수가 "
        "그림(막대그래프)에만 있고 본문 텍스트에는 없습니다. 조성 성분 순서도 "
        "논문마다 달라서, 뽑아낸 값은 반드시 근거 문장을 보고 확인해야 합니다. "
        "목표는 타이핑을 줄이는 것이지 검토를 없애는 것이 아닙니다.")

    if not PDF_OK:
        st.error(f"PDF 모듈을 못 불러왔습니다: {PDF_ERR}\n\n"
                 "`pip install pdfplumber pypdf` 후 다시 실행하세요.")
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

                doi = st.text_input(
                    "DOI (확인/수정)", value=ex["doi"] or "",
                    help="같은 논문에서 나온 행은 모두 이 DOI를 씁니다. "
                         "논문 단위 CV의 기준이므로 정확해야 합니다.")
                if ex.get("doi_alternatives") and len(ex["doi_alternatives"]) > 1:
                    st.caption(f"PDF에서 찾은 다른 DOI: "
                               f"{', '.join(ex['doi_alternatives'][1:5])} "
                               f"(참고문헌 DOI일 수 있으니 확인하세요)")

                st.subheader("찾은 조성 후보")
                if not ex["ratios"]:
                    st.info("조성을 못 찾았습니다. Methods 나 SI 에서 직접 찾아 "
                            "'직접 입력' 탭에 넣으세요.")
                for i, r in enumerate(ex["ratios"]):
                    with st.expander(
                            f"{r['ratio']}   (p.{r['page']}, 합계 {r['sum']})"
                            + ("  ⚠️ 확인 필요" if r["chem_warnings"] else ""),
                            expanded=(i == 0)):
                        st.code(r["evidence"], language=None)
                        cc1, cc2 = st.columns(2)
                        cc1.write(f"**논문 표기 그대로:** `{r['ratio_as_written']}`")
                        cc2.write(f"**표준 순서로 변환:** `{r['ratio']}`")
                        if r["order_detected"]:
                            st.caption(f"문장에서 읽은 성분 순서: "
                                       f"{' → '.join(r['order'])}")
                        else:
                            st.caption("성분 순서를 못 읽어 표기 그대로 두었습니다. "
                                       "이온화:헬퍼:콜레스테롤:PEG 순서가 맞는지 확인하세요.")
                        for w in r["chem_warnings"]:
                            st.warning(w)

                st.subheader("찾은 EE 후보")
                spec = ex.get("ee_specific", [])
                if spec:
                    st.dataframe(pd.DataFrame(
                        [{"EE (%)": d["ee"], "페이지": d["page"],
                          "근거": d["evidence"][:170]} for d in spec]),
                        use_container_width=True, hide_index=True)
                else:
                    st.info(
                        "본문에서 구체적인 EE 수치를 못 찾았습니다. "
                        "이 논문은 EE를 그림에만 실었을 가능성이 큽니다 — "
                        "Figure 캡션과 SI 표를 확인해 직접 넣으세요.")
                gen = [d for d in ex["ee"] if d.get("generic")]
                if gen:
                    with st.expander(f"일반론 문장으로 판단해 제외한 값 {len(gen)}개"):
                        st.caption("'typically close to 100%' 같은 배경 서술입니다. "
                                   "이 논문의 측정값이 아닙니다.")
                        st.dataframe(pd.DataFrame(
                            [{"값": d["ee"], "근거": d["evidence"][:170]} for d in gen]),
                            use_container_width=True, hide_index=True)

                st.subheader("기타 추출값")
                oc1, oc2 = st.columns(2)
                with oc1:
                    st.write("**지질**")
                    st.write(f"- 이온화: {', '.join(ex['ionizable']) or '못 찾음'}")
                    st.write(f"- 헬퍼: {', '.join(ex['helper']) or '못 찾음'}")
                    st.write(f"- PEG: {', '.join(ex['peg']) or '못 찾음'}")
                    st.write(f"- cargo: {', '.join(ex['cargo']) or '못 찾음'}")
                with oc2:
                    st.write("**공정/물성 (첫 값만)**")
                    for k, lab in [("np_ratio", "N/P"), ("ph", "pH"),
                                   ("size", "크기(nm)"), ("pdi", "PDI"),
                                   ("zeta", "제타(mV)")]:
                        v = ex.get(k) or []
                        st.write(f"- {lab}: "
                                 + (", ".join(str(d['value']) for d in v[:5]) or "못 찾음"))

                st.divider()
                st.subheader("표 편집 후 추가")
                st.caption("아래 표를 직접 고칠 수 있습니다. EE는 대부분 손으로 "
                           "넣어야 합니다. 빈 행은 저장 시 무시됩니다.")
                draft = LP.to_draft_rows(ex)
                if doi:
                    draft["reference_doi"] = doi
                edited = st.data_editor(draft, num_rows="dynamic",
                                        use_container_width=True, key="pdf_edit")

                if st.button("이 행들을 데이터에 추가", type="primary"):
                    keep = edited[
                        edited["lipid_molar_ratio"].astype(str).str.strip() != ""]
                    keep = keep[
                        pd.to_numeric(
                            keep["encapsulation_efficiency_percent_std_num"],
                            errors="coerce").notna()]
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
    if len(df) and "reference_doi" in df:
        s = df["reference_doi"].dropna().astype(str)
        prev_doi = s.iloc[-1] if len(s) else ""

    with st.form("entry", clear_on_submit=False):
        st.subheader("필수")
        f1, f2 = st.columns(2)
        doi = f1.text_input("논문 DOI", value=prev_doi,
                            placeholder="10.1038/s41586-021-03534-y")
        ratio = f2.text_input("지질 몰비", placeholder="50:10:38.5:1.5",
                              help="이온화 : 헬퍼 : 콜레스테롤 : PEG 순서")
        g1, g2 = st.columns(2)
        ee = g1.number_input("EE (%)", 0.0, 100.0, 90.0, 0.1)
        ion = g2.text_input("이온화지질 이름", placeholder="SM-102",
                            help="SMILES는 PubChem에서 자동 조회됩니다")

        st.subheader("권장")
        h1, h2, h3 = st.columns(3)
        cargo = h1.selectbox("cargo", ["", "mRNA", "siRNA", "saRNA", "pDNA",
                                       "ASO", "circRNA"])
        helper = h2.text_input("헬퍼 지질", placeholder="DSPC")
        peg = h3.text_input("PEG 지질", placeholder="DMG-PEG2000")
        i1, i2 = st.columns(2)
        npr = i1.number_input("N/P 비 (0=미기재)", 0.0, 60.0, 0.0, 0.5)
        ph = i2.number_input("완충액 pH (0=미기재)", 0.0, 9.0, 0.0, 0.1)

        with st.expander("선택 — 사후 측정값 (설계 예측에는 안 쓰임)"):
            j1, j2, j3 = st.columns(3)
            size = j1.number_input("입자 크기 (nm, 0=미기재)", 0.0, 500.0, 0.0, 1.0)
            pdi = j2.number_input("PDI (0=미기재)", 0.0, 1.0, 0.0, 0.01)
            zeta = j3.number_input("제타 (mV)", -100.0, 100.0, 0.0, 0.1)
            zeta_na = j3.checkbox("제타 미기재", value=True)
        note = st.text_input("메모", placeholder="Table 2, row 3")

        ok = st.form_submit_button("추가", type="primary")

    if ok:
        if not doi.strip():
            st.error("DOI가 필요합니다. 논문 단위 CV의 기준입니다.")
        elif not ratio.strip():
            st.error("몰비가 필요합니다.")
        else:
            e = LE.Entry()
            e.paper(doi.strip())
            e.add(ratio.strip(), ee,
                  ion=ion.strip() or None, helper=helper.strip() or None,
                  peg=peg.strip() or None, cargo=cargo or None,
                  np_ratio=npr or None, ph=ph or None,
                  size=size or None, pdi=pdi or None,
                  zeta=None if zeta_na else zeta,
                  note=note.strip() or None)
            new = e.to_frame()
            w = LP.chemistry_check(ratio.strip()) if PDF_OK else []
            with st.spinner("SMILES 조회 중..."):
                new = LE.resolve_smiles(new, verbose=False)
            add_rows(new)
            got = new["ionizable_lipid_smiles"].notna().any()
            st.success(f"추가했습니다." + ("  SMILES 자동 확보 완료." if got else ""))
            for msg in w:
                st.warning(msg)
            if ion.strip() and not got:
                st.info(f"'{ion}' 의 SMILES를 PubChem에서 못 찾았습니다. "
                        f"데이터 관리 탭에서 직접 넣을 수 있습니다.")


# ==========================================================================

# ==========================================================================

# ==========================================================================
# ==========================================================================
# 탭 3 — 데이터 관리
# ==========================================================================
with tab_data:
    st.header("데이터 관리")

    up2 = st.file_uploader("기존 CSV 불러오기 (표준 형식으로 자동 정렬 및 정제)", type=["csv"], key="csvup")
    if up2 is not None:
        raw = up2.read()
        for enc in ("utf-8-sig", "cp949", "latin-1"):
            try:
                d_raw = pd.read_csv(io.BytesIO(raw), encoding=enc)
                break
            except: continue
        
        # 1. 표준 컬럼에 맞춰 데이터 정제 및 정렬
        d_clean = pd.DataFrame(columns=df.columns)
        for col in df.columns:
            if col in d_raw.columns:
                d_clean[col] = d_raw[col]
            else:
                d_clean[col] = np.nan
        
      # 2. EE 값 자동 정제 (문자열 -> 숫자) - 💡 패치 1 적용 (손실 방지)
        ee_col = "encapsulation_efficiency_percent_std_num"
        d_clean[ee_col] = d_clean[ee_col].map(P.robust_ee)
        
        # 3. 숫자형 컬럼 타입 변환
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
            valid_d = P.dedupe(valid_d)  # 💡 패치 2 적용 (중복 제거)
            add_rows(valid_d)
            st.success("✅ 중복 제거 후 안전하게 추가되었습니다.")
            st.rerun()
            
        if c2.button("기존 데이터를 이 파일로 교체"):
            valid_d = d_clean.dropna(subset=[ee_col])
            valid_d = P.dedupe(valid_d)  # 💡 패치 2 적용 (중복 제거)
            st.session_state.df = valid_d
            save_disk()
            st.success("✅ 파일이 중복 제거된 표준 형식으로 교체되었습니다.")
            st.rerun()

    # =========================================================
    # 복붙 전용 입력창 (기존과 동일)
    # =========================================================
    st.divider()
    st.subheader("📝 엑셀에서 바로 복사/붙여넣기")
    # (이하 복붙 입력창과 전체 데이터 편집창 코드는 기존 그대로 두셔도 됩니다)
    
    # ... [중간 복붙 및 전체 데이터 편집 부분] ...

    # 전체 데이터 편집 및 저장 버튼 있는 구간부터 다시 붙여넣으세요
    if len(df) == 0:
        st.info("아직 데이터가 없습니다.")
    else:
        ed = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="main_edit", height=380)
        
        b1, b2, b3 = st.columns(3)
        if b1.button("편집 내용 저장", type="primary"):
            if "lipid_molar_ratio" in ed.columns:
                valid_ed = ed.dropna(subset=["lipid_molar_ratio"])
                valid_ed = valid_ed[valid_ed["lipid_molar_ratio"].astype(str).str.strip() != ""]
            else:
                valid_ed = ed

            if pd.to_numeric(valid_ed["encapsulation_efficiency_percent_std_num"], errors='coerce').isna().any():
                st.error("🚨 저장 실패! EE(%) 칸을 확인하세요.")
            else:
                st.session_state.df = valid_ed
                save_disk()
                st.success("✅ 편집 내용이 성공적으로 저장되었습니다.")

        if b2.button("빠진 SMILES 채우기"):
            with st.spinner("PubChem 조회 중..."):
                st.session_state.df = LE.resolve_smiles(ed, verbose=False)
            save_disk()
            st.success("✅ 빠진 SMILES 데이터가 채워졌습니다.")
            st.rerun()
            
        if b3.button("검증 실행"):
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                LE.validate(ed)
            st.code(buf.getvalue(), language=None)
# ==========================================================================
# 탭 4 — 모델 실행
# ==========================================================================
with tab_model:
    st.header("모델 실행")
    st.caption("논문 단위 교차검증으로 평가합니다. 무작위 분할은 성능을 "
               "부풀리므로 쓰지 않습니다.")

    if n_pap < 5:
        st.warning(f"논문이 {n_pap}편입니다. 논문 단위 5-fold CV에는 최소 5편, "
                   f"의미 있는 결론에는 20편 이상이 필요합니다.")

    if st.button("평가 실행", type="primary", disabled=(n_pap < 3)):
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from scipy.stats import spearmanr

        d = st.session_state.df.copy()
        y = pd.to_numeric(d["encapsulation_efficiency_percent_std_num"],
                          errors="coerce")
        frac = (y > 0) & (y <= 1)
        y = y.where(~frac, y * 100)
        keep = y.notna() & (y > 0) & (y <= 100)
        d, y = d[keep].reset_index(drop=True), y[keep].reset_index(drop=True)
        g = d["reference_doi"].astype(str).str.strip().str.lower()

        # 특징 — 조성 + 공정 + 범주 (사후 측정값 제외)
        R = pd.DataFrame(
            d["lipid_molar_ratio"].astype(str).str.replace(r"[\/\-,;|]", ":", regex=True)
            .str.split(":", expand=True).apply(pd.to_numeric, errors="coerce"))
        R = R.iloc[:, :4]
        R.columns = ["ionizable", "helper", "chol", "peg"][:R.shape[1]]
        tot = R.sum(axis=1, min_count=1).replace(0, np.nan)
        R = R.div(tot, axis=0) * 100
        proc = [c for c in ("np_ratio_std_num", "buffer_ph_std_num") if c in d]
        X = pd.concat([R, d[proc].apply(pd.to_numeric, errors="coerce")], axis=1)
        cats = [c for c in ("cargo_type", "helper_lipid_name", "ionizable_lipid_name")
                if c in d and 1 < d[c].nunique() <= 40]
        for c in cats:
            X[c] = d[c].astype(str)

        num = [c for c in X.columns if c not in cats]
        pre = ColumnTransformer([
            ("n", Pipeline([("i", SimpleImputer(strategy="median")),
                            ("s", StandardScaler())]), num),
            ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                            ("o", OneHotEncoder(handle_unknown="ignore",
                                                min_frequency=2))]), cats)]
            if cats else
            [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler())]), num)])
        model = Pipeline([("pre", pre),
                          ("m", RandomForestRegressor(
                              n_estimators=400, min_samples_leaf=3,
                              max_features=0.5, random_state=42, n_jobs=-1))])

        k = min(5, g.nunique())
        cv = GroupKFold(n_splits=k)
        with st.spinner("교차검증 중..."):
            pm = cross_val_predict(model, X, y, cv=cv, groups=g)
            pb = cross_val_predict(DummyRegressor(strategy="mean"), X, y,
                                   cv=cv, groups=g)

        mae_m, mae_b = mean_absolute_error(y, pm), mean_absolute_error(y, pb)
        gain = (mae_b - mae_m) / mae_b * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("모델 MAE", f"{mae_m:.2f} %p")
        c2.metric("baseline MAE", f"{mae_b:.2f} %p")
        c3.metric("개선율", f"{gain:+.1f} %",
                  delta=f"{'유의미' if gain > 5 else '미미'}")

        # 논문 간 vs 논문 내 분산
        gm = y.groupby(g).transform("mean")
        icc = np.var(gm, ddof=0) / np.var(y, ddof=0) if np.var(y) > 0 else np.nan
        st.write(f"**논문 간 분산 비중(ICC) = {icc:.2f}** — "
                 f"EE 변동의 {icc*100:.0f}%가 조성이 아니라 어느 논문인지에서 옵니다. "
                 f"이 값이 높으면 절대값 예측은 원리적으로 어렵습니다.")

        # 논문 내 순위
        rhos = []
        for gid, idx in y.groupby(g).groups.items():
            if len(idx) >= 4:
                r, _ = spearmanr(y.loc[idx], pd.Series(pm, index=y.index).loc[idx])
                if not np.isnan(r):
                    rhos.append(r)
        if rhos:
            st.success(
                f"**논문 내 순위 상관 중앙값 rho = {np.median(rhos):.2f}** "
                f"({sum(r > 0 for r in rhos)}/{len(rhos)}편에서 양수)\n\n"
                f"절대값은 못 맞춰도 '어느 조성이 더 나은지'는 맞춥니다. "
                f"스크리닝 용도로는 이 값이 MAE보다 중요합니다.")
        else:
            st.info("논문당 처방이 4개 이상인 논문이 없어 순위 평가를 못 했습니다. "
                    "한 논문에서 조성을 여러 개 담으면 이 지표를 볼 수 있습니다.")

        res = pd.DataFrame({"실측 EE": y, "예측 EE": pm, "논문": g})
        st.scatter_chart(res, x="실측 EE", y="예측 EE")
        st.dataframe(res.head(50), use_container_width=True)

# ==========================================
# 🌐 인터넷 대량 수집 탭
# ==========================================
with st.expander("🌐 인터넷 대량 수집 (PMC 자동 검색)", expanded=False):
    st.markdown("PubMed Central(PMC)에서 LNP 논문을 자동으로 검색하고 성분과 EE 데이터를 추출합니다.")
    
    # 몇 편을 검색할지 정하는 칸
    retmax_input = st.number_input("최대 검색 논문 수 (기본 50, 최대 200 추천)", min_value=10, max_value=500, value=50, step=10)
    
    # 수집 시작 버튼
    if st.button("🚀 자동 수집 시작"):
        # 진행 상황을 알려주는 스피너(로딩바) 표시
        with st.spinner(f"PMC에서 최대 {retmax_input}편의 논문을 수집 중입니다. 이 작업은 1~2분 정도 걸릴 수 있습니다..."):
            try:
                # 실제로 lnp_harvest.py의 기능을 작동시키는 부분
                df_harvested, status = lnp_harvest.harvest(retmax=retmax_input, lnp_pdf_mod=lnp_pdf)
                
                if not df_harvested.empty:
                    st.success(f"수집 완료! 총 {len(df_harvested)}개의 유효한 데이터를 찾았습니다.")
                    
                    # 수집된 데이터를 화면에 표로 보여줌
                    st.dataframe(df_harvested)
                    
                    # CSV 다운로드 버튼
                    csv_data = df_harvested.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 수집된 데이터 CSV로 다운로드",
                        data=csv_data,
                        file_name="lnp_harvested_candidates.csv",
                        mime="text/csv",
                    )
                    st.info("💡 위 버튼을 눌러 CSV를 다운로드한 후, 원문을 보고 EE 값 등을 검토하세요. 완료된 파일은 '📊 데이터 관리' 탭에서 기존 데이터와 합치면 됩니다.")
                else:
                    st.warning("조건에 맞는 데이터를 찾지 못했습니다.")
            except Exception as e:
                st.error(f"수집 중 오류가 발생했습니다. (오류 내용: {e})")


# ==========================================

# ==========================================================================
# ==========================================================================
# ⚓ 앵커링 (실측 영점 조절) 기능 탭 (k=3 패치 적용 완료)
# ==========================================================================
with st.expander("⚓ 앵커링 (영점 조절) 기반 정밀 예측", expanded=False):
    st.markdown("AI가 추천하는 조성으로 실험하거나, 원하는 조성을 직접 입력하여 영점을 조절합니다.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🧪 1. 실험할 조성 선정")
        
        # 1-1. AI 추천 버튼 (최종 수정본 - 패치 3 적용)
        if st.button("AI 앵커 추천받기 (k=3)"):
            try:
                if 'df' not in locals() or df.empty:
                    st.error("먼저 '데이터 관리' 탭에서 학습용 CSV 파일을 업로드해주세요.")
                else:
                    X, num_cols, cat_cols = v3.build_features(df)
                    m = lnp_anchor.AnchoredEEPredictor(v3, num_cols, cat_cols)
                    m.fit(X, df["encapsulation_efficiency_percent_std_num"])
                    
                    # 💡 패치 3 적용: k=2를 k=3으로 변경하여 안정성 극대화
                    picks = m.suggest_anchors(X, k=3) 
                    
                    st.session_state['anchored_model'] = m
                    st.session_state['anchor_picks'] = picks
                    st.session_state['X_data'] = X
                    
                    st.success(f"🤖 AI가 추천한 앵커: {picks[0]}번, {picks[1]}번, {picks[2]}번")
                    
                    st.divider()
                    st.write("📋 **추천된 앵커 상세 정보**")
                    for i, idx in enumerate(picks):
                        rec = df.iloc[idx]
                        st.markdown(f"**앵커 {i+1} (데이터 {idx}번)**")
                        st.write(f"- 비율: {rec.get('lipid_molar_ratio', '기록없음')}")
                        st.write(f"- 이온화지질: {rec.get('ionizable_lipid_name', '기록없음')}")
                        st.write(f"- 문헌 EE: {rec.get('encapsulation_efficiency_percent_std_num', 0):.1f}%")
            except Exception as e:
                st.error(f"오류: {e}")

        # 💡 패치 3 적용: 정직한 앵커 홀드아웃 평가 리포트 버튼 추가
        if st.button("📊 앵커링 정직 평가 리포트 실행"):
            with st.spinner("논문 단위 홀드아웃 검증 중... (잠시만 기다려주세요)"):
                try:
                    R, stats = P.anchor_report(df, v3, lnp_anchor, k=3)
                    if "note" in stats:
                        st.warning(stats["note"])
                    else:
                        st.success(f"✅ 검증 완료! (총 {stats['papers']}편의 논문으로 테스트)")
                        st.metric("앵커 3개 적용 시 평균 MAE", 
                                  f"{stats['mae_anchored']:.1f} %p",
                                  f"{stats['gain_pct']:+.1f}% (오차 감소)")
                        st.caption(f"개선 성공 확률: {stats['papers']}편 중 {stats['improved']}편 개선 (p-value: {stats['p_value']:.4f})")
                except Exception as e:
                    st.error(f"평가 중 오류: {e}")

        # 1-2. 직접 지정하기 (입력칸 3개로 확장)
        st.write("---")
        st.caption("또는, 데이터 관리 표에서 확인한 '행 번호'를 직접 입력하세요.")
        manual_1 = st.number_input("실험 1의 데이터 행 번호", min_value=0, max_value=n_rows-1 if n_rows>0 else 0, value=0)
        manual_2 = st.number_input("실험 2의 데이터 행 번호", min_value=0, max_value=n_rows-1 if n_rows>0 else 0, value=1 if n_rows>1 else 0)
        manual_3 = st.number_input("실험 3의 데이터 행 번호", min_value=0, max_value=n_rows-1 if n_rows>0 else 0, value=2 if n_rows>2 else 0)
        
        if st.button("직접 지정한 데이터로 앵커 설정"):
            if 'df' in locals() and not df.empty:
                X, num_cols, cat_cols = v3.build_features(df)
                m = lnp_anchor.AnchoredEEPredictor(v3, num_cols, cat_cols)
                m.fit(X, df["encapsulation_efficiency_percent_std_num"])
                
                st.session_state['anchored_model'] = m
                st.session_state['anchor_picks'] = [manual_1, manual_2, manual_3]
                st.session_state['X_data'] = X
                st.success(f"✅ 앵커 설정됨: {manual_1}번, {manual_2}번, {manual_3}번")

        # 1-3. 앵커 정보 출력 (자동 대응)
        if 'anchor_picks' in st.session_state:
            st.divider()
            st.write("📋 **현재 설정된 앵커 정보**")
            for i, idx in enumerate(st.session_state['anchor_picks']):
                rec = df.iloc[idx]
                st.markdown(f"**앵커 {i+1} (데이터 {idx}번)**")
                st.write(f"- 비율: {rec.get('lipid_molar_ratio', '기록없음')}")
                st.write(f"- 이온화지질: {rec.get('ionizable_lipid_name', '기록없음')}")
                st.write(f"- 문헌 EE: {rec.get('encapsulation_efficiency_percent_std_num', 0):.1f}%")

    with col2:
        st.info("📊 2. 실측 결과 입력 및 예측")
        # 실측값 입력칸 3개로 확장
        anchor_1_ee = st.number_input("실험 1의 실제 측정 EE (%)", min_value=0.0, max_value=100.0, value=0.0)
        anchor_2_ee = st.number_input("실험 2의 실제 측정 EE (%)", min_value=0.0, max_value=100.0, value=0.0)
        anchor_3_ee = st.number_input("실험 3의 실제 측정 EE (%)", min_value=0.0, max_value=100.0, value=0.0)
        
        if st.button("영점 조절 후 전체 예측 실행"):
            if 'anchored_model' in st.session_state and anchor_1_ee > 0 and anchor_2_ee > 0 and anchor_3_ee > 0:
                m = st.session_state['anchored_model']
                picks = st.session_state['anchor_picks']
                X = st.session_state['X_data']
                
                with st.spinner("정밀 예측 중..."):
                    # 앵커 3개 값 전달
                    final_preds = m.predict(X, anchor_idx=picks, anchor_y=[anchor_1_ee, anchor_2_ee, anchor_3_ee])
                    st.success("🎯 3개의 실측값을 바탕으로 보정된 정밀 예측값이 산출되었습니다.")
                    
                    result_df = df.copy()
                    result_df['보정된_예측_EE'] = final_preds
                    st.dataframe(result_df[['reference_doi', 'lipid_molar_ratio', 'encapsulation_efficiency_percent_std_num', '보정된_예측_EE']])
                    
                    csv_data = result_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("📥 결과 CSV 다운로드", csv_data, "lnp_anchored_predictions_k3.csv", "text/csv", use_container_width=True)
            else:
                st.warning("먼저 앵커를 추천받거나 지정하고, 3개의 실측값을 모두 0보다 크게 입력하세요.")
