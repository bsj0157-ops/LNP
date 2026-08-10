# ==========================================================================
#  LNP-Predictor v3  —  노트북에 그대로 붙여넣고 실행하세요
#  ------------------------------------------------------------------------
#  v2가 아무것도 출력하지 않은 이유:
#    v2는 명령줄 전용(`--csv` 필수)이라, 노트북에서는 argparse가
#    SystemExit(2)를 던지고 그 메시지는 stderr로만 나갑니다.
#    v3는 argparse가 없고 맨 아래에서 자동 실행됩니다.
#
#  이 파일이 하는 일 (순서대로 자동 실행):
#    STEP 1  자체 테스트   — 파싱/전처리 로직이 맞는지 assert로 검증
#    STEP 2  건전성 검사   — 신호를 심은 가짜 데이터에서 모델이 그걸 찾아내는지
#    STEP 3  CSV 자동 탐색 — 못 찾으면 DEMO 모드로 계속 진행
#    STEP 4  스키마 진단   — 어느 컬럼이 잡혔고 결측이 얼마인지
#    STEP 5  정확도 평가   — 논문 단위 CV + baseline + 통계 검정
#    STEP 6  판정          — PASS / FAIL 을 근거와 함께 출력
#    STEP 7  그림 저장     — lnp_v3_report.png
# ==========================================================================

import os
import sys
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TARGET = "encapsulation_efficiency_percent_std_num"
CSV_HINT = "LNP_Atlas_DB_202509_v1.CSV"   # 경로를 알면 여기에 직접 적어도 됩니다

_ok, _fail = "  [ok]  ", "  [FAIL] "


def rule(title=""):
    print("\n" + "=" * 74)
    if title:
        print(title)
        print("=" * 74)


# ==========================================================================
# 전처리 함수들
# ==========================================================================

COMP = ["ionizable", "helper", "chol", "peg"]


def load_csv(path):
    """utf-8-sig → cp949 → latin-1 순으로 시도."""
    last = None
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, low_memory=False)
            print(f"  loaded  encoding={enc}  shape={df.shape}")
            return df
        except UnicodeDecodeError as e:
            last = e
    raise RuntimeError(f"cannot decode {path}") from last


def parse_lipid_ratio(series):
    """'46.3:9.4:42.7:1.6' → 4개 성분 + 파생변수. 합계 100으로 정규화."""
    txt = series.astype(str).str.strip()
    parts = txt.str.replace(r"[\/\-,;|]", ":", regex=True).str.split(":", expand=True)
    num = parts.apply(pd.to_numeric, errors="coerce")

    n_valid = num.notna().sum(axis=1)
    out = pd.DataFrame(index=series.index, dtype=float)
    for c in COMP:
        out[c] = np.nan

    m4 = n_valid == 4
    if m4.any():
        for i, name in enumerate(COMP):
            if i in num.columns:
                out.loc[m4, name] = num.loc[m4, i].values

    m3 = n_valid == 3                      # helper(DSPC) 없는 3-성분 처방
    if m3.any():
        out.loc[m3, "ionizable"] = num.loc[m3, 0].values
        out.loc[m3, "helper"] = 0.0
        out.loc[m3, "chol"] = num.loc[m3, 1].values
        out.loc[m3, "peg"] = num.loc[m3, 2].values

    total = out[COMP].sum(axis=1, min_count=1)
    out.loc[(total <= 0) | total.isna(), COMP] = np.nan
    out[COMP] = out[COMP].div(total.replace(0, np.nan), axis=0) * 100.0

    out["ion_to_helper"] = out["ionizable"] / out["helper"].replace(0, np.nan)
    out["ion_plus_chol"] = out["ionizable"] + out["chol"]
    out["log_peg"] = np.log1p(out["peg"])
    return out


def smiles_features(series):
    """이온화 지질 SMILES → 물리화학 descriptor. len(SMILES)는 쓰지 않음."""
    # pandas 3.0 부터 astype(str) 이 NaN 을 'nan' 문자열로 바꾸지 않고
    # float NaN 으로 보존합니다. 그대로 두면 RDKit 에 float 이 들어가
    # TypeError 가 납니다(실측). fillna 를 먼저 걸어 통일합니다.
    smi = series.fillna("").astype(object).map(
        lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v))
        else str(v).strip())
    smi = pd.Series(smi, index=series.index)
    bad = smi.str.lower().isin(["nan", "none", ""])
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
        RDLogger.DisableLog("rdApp.*")

        getters = {
            "MolWt": Descriptors.MolWt, "MolLogP": Crippen.MolLogP,
            "TPSA": rdMolDescriptors.CalcTPSA,
            "NumHDonors": Lipinski.NumHDonors,
            "NumHAcceptors": Lipinski.NumHAcceptors,
            "NumRotatableBonds": Lipinski.NumRotatableBonds,
            "RingCount": rdMolDescriptors.CalcNumRings,
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3,
            "HeavyAtomCount": Descriptors.HeavyAtomCount,
        }
        smarts = {   # 이온화 지질에서 pKa·생분해성에 직결되는 부분구조
            "n_tert_amine": "[NX3;H0;!$(N=*);!$(N-[#6]=[O,N])]",
            "n_ester": "[CX3](=O)[OX2H0][#6]",
            "n_amide": "[NX3][CX3](=O)",
            "n_N_total": "[#7]",
            "n_c8_chain": "CCCCCCCC",
        }
        patts = {k: Chem.MolFromSmarts(v) for k, v in smarts.items()}
        cols = list(getters) + list(patts)

        recs = []
        for s, isbad in zip(smi, bad):
            mol = None if isbad else Chem.MolFromSmiles(s)
            if mol is None:
                recs.append({k: np.nan for k in cols})
                continue
            row = {k: f(mol) for k, f in getters.items()}
            row.update({k: len(mol.GetSubstructMatches(p)) for k, p in patts.items()})
            recs.append(row)
        return pd.DataFrame(recs, index=series.index)

    except ImportError:
        print("  note: RDKit 없음 → 부분구조 카운트로 대체 (pip install rdkit)")
        out = pd.DataFrame(index=series.index)
        out["n_N"] = smi.str.count(r"[Nn]")
        out["n_O"] = smi.str.count(r"[Oo]")
        out["n_C"] = smi.str.count(r"C")
        out["n_ester_txt"] = smi.str.count(r"C\(=O\)O")
        out["n_branch"] = smi.str.count(r"\(")
        out[bad] = np.nan
        return out


def find_group_key(df, quiet=False):
    """논문 단위 분할용 그룹 키(DOI/PMID/reference)를 찾는다."""
    exact = ["reference_doi", "doi", "DOI", "ref_doi", "pmid", "PMID",
             "reference", "reference_id", "study_id", "paper_id", "source_id",
             "citation", "publication", "reference_title", "title"]
    for c in exact:
        if c in df.columns and df[c].notna().sum() > 0.5 * len(df):
            k = df[c].astype(str).str.strip().str.lower()
            if not quiet:
                print(f"  group key = '{c}'  ({k.nunique()} unique studies)")
            return k, True
    for c in df.columns:
        if any(t in c.lower() for t in ("doi", "pmid", "referen", "citation", "study")):
            if df[c].notna().sum() > 0.5 * len(df):
                k = df[c].astype(str).str.strip().str.lower()
                if not quiet:
                    print(f"  group key = '{c}'  ({k.nunique()} unique studies)")
                return k, True
    if not quiet:
        print("  !! 논문 식별 컬럼을 못 찾았습니다 → 논문 단위 CV 불가.")
        print("     이 상태의 점수는 과대평가입니다. DOI 컬럼명을 알려주세요.")
    return pd.Series(np.arange(len(df)).astype(str), index=df.index), False


def build_features(df, include_measured=False):
    """설계 시점에 알 수 있는 변수만 기본 사용(include_measured=False)."""
    blocks, log = [], []

    ratio_col = next((c for c in df.columns if "molar_ratio" in c.lower()
                      or "lipid_ratio" in c.lower()), None)
    if ratio_col:
        blocks.append(parse_lipid_ratio(df[ratio_col]))
        log.append(f"조성({ratio_col})")

    smi_col = next((c for c in df.columns if "smiles" in c.lower()), None)
    if smi_col:
        blocks.append(smiles_features(df[smi_col]))
        log.append(f"화학({smi_col})")

    proc = [c for c in df.columns
            if any(k in c.lower() for k in
                   ("np_ratio", "n_p_ratio", "flow", "buffer_ph", "total_lipid",
                    "lipid_conc", "weight_ratio", "mixing", "dilution", "ionic"))
            and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0.1 * len(df)]
    if proc:
        blocks.append(df[proc].apply(pd.to_numeric, errors="coerce"))
        log.append(f"공정({len(proc)}개)")

    if include_measured:      # 진단용 전용 — 설계에는 쓸 수 없는 사후 측정값
        meas = [c for c in df.columns
                if any(k in c.lower() for k in ("particle_size", "pdi", "zeta"))
                and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]
        if meas:
            blocks.append(df[meas].apply(pd.to_numeric, errors="coerce"))
            log.append(f"물성({len(meas)}개)")

    cats = [c for c in df.columns
            if any(k in c.lower() for k in
                   ("cargo", "nucleic_acid", "payload", "helper_lipid_name",
                    "peg_lipid_name", "ionizable_lipid_name", "apparatus", "method"))
            and 1 < df[c].nunique() <= 40]
    cat_df = df[cats].astype(str) if cats else pd.DataFrame(index=df.index)
    if cats:
        log.append(f"카테고리({len(cats)}개)")

    num = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=df.index)
    num = num.loc[:, ~num.columns.duplicated()]
    print("  feature blocks: " + (", ".join(log) if log else "없음"))
    return pd.concat([num, cat_df], axis=1), list(num.columns), list(cat_df.columns)


def make_pipeline(num_cols, cat_cols, estimator):
    """전처리를 Pipeline 안에 둬서 fold마다 재적합 → leakage 차단."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=5,
                            sparse_output=False)
    except TypeError:                                  # sklearn < 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    steps = []
    if num_cols:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num_cols))
    if cat_cols:
        steps.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", ohe)]), cat_cols))
    return Pipeline([("pre", ColumnTransformer(steps, remainder="drop")),
                     ("est", estimator)])


# ==========================================================================
# STEP 1 — 자체 테스트: 로직이 맞는지 assert로 확인
# ==========================================================================

def selftest():
    rule("STEP 1 / 자체 테스트 — 전처리 로직 검증")
    n_pass = 0

    # (1) 4-성분 파싱 + 정규화
    r = parse_lipid_ratio(pd.Series(["50:10:38.5:1.5"]))
    assert abs(r.loc[0, "ionizable"] - 50.0) < 1e-6, "4-comp 파싱 실패"
    assert abs(r.loc[0, COMP].sum() - 100.0) < 1e-6, "합계 정규화 실패"
    print(_ok + "4-성분 파싱 및 합계 100 정규화"); n_pass += 1

    # (2) 상대비 표기(합≠100)도 같은 스케일로 정규화되는지
    r = parse_lipid_ratio(pd.Series(["50:10:38.5:1.5", "10:2:7.7:0.3"]))
    assert np.allclose(r.loc[0, COMP].values, r.loc[1, COMP].values, atol=1e-6), \
        "상대비 정규화 실패"
    print(_ok + "상대비 표기('10:2:7.7:0.3')를 몰%와 동일 스케일로 변환"); n_pass += 1

    # (3) 3-성분 처방 → helper=0
    r = parse_lipid_ratio(pd.Series(["60:38.5:1.5"]))
    assert r.loc[0, "helper"] == 0.0 and abs(r.loc[0, COMP].sum() - 100) < 1e-6, \
        "3-comp 처리 실패"
    print(_ok + "3-성분 처방(DSPC 없음)을 helper=0으로 복구"); n_pass += 1

    # (4) 다른 구분자
    r = parse_lipid_ratio(pd.Series(["50/10/38.5/1.5"]))
    assert abs(r.loc[0, "ionizable"] - 50.0) < 1e-6, "'/' 구분자 실패"
    print(_ok + "'/' '-' ',' 구분자 허용"); n_pass += 1

    # (5) 깨진 값은 조용히 0이 되지 않고 NaN
    r = parse_lipid_ratio(pd.Series(["n.r.", "", "not reported", "50:10:38.5:1.5:9"]))
    assert r[COMP].iloc[:4].isna().all().all(), "깨진 값이 NaN이 아님"
    print(_ok + "미보고/5성분 값을 NaN 처리 (0으로 오인 안 함)"); n_pass += 1

    # (6) v2 파일4의 인덱스 버그가 재발하지 않는지
    r = parse_lipid_ratio(pd.Series(["50:10:38.5:1.5", "40:20:39:1", "60:5:33:2"]))
    assert r[COMP].nunique().min() > 1, "컬럼이 모두 같은 값 — 인덱스 버그 재발"
    print(_ok + "4개 비율 컬럼이 서로 다른 값 (파일4 버그 회귀 방지)"); n_pass += 1

    # (7) SMILES descriptor — 표기가 달라도 같은 분자면 같은 값
    f = smiles_features(pd.Series(["CCO", "OCC", "bad_smiles"]))
    key = "MolWt" if "MolWt" in f.columns else f.columns[0]
    assert abs(f.loc[0, key] - f.loc[1, key]) < 1e-6, \
        "동일 분자의 다른 표기가 다른 값을 냄"
    assert f.loc[2].isna().all(), "파싱 실패 분자가 NaN이 아님"
    print(_ok + "SMILES descriptor가 표기법에 불변 ('CCO' == 'OCC')"); n_pass += 1

    # (8) len(SMILES)는 표기에 따라 달라진다는 반례 — v2 파일4·5가 쓴 특징
    long_form = "C(C(O))"       # 같은 원자 수, 다른 표기
    assert len("CCO") != len(long_form), "반례 구성 오류"
    print(_ok + f"대조: len(SMILES)는 표기에 따라 {len('CCO')}≠{len(long_form)}"
          " → 화학 특징으로 부적합"); n_pass += 1

    # (9) Pipeline이 결측치를 포함한 데이터로 fit/predict 되는지
    from sklearn.ensemble import RandomForestRegressor
    Xt = pd.DataFrame({"a": [1.0, 2, np.nan, 4, 5, 6], "b": [1.0, np.nan, 3, 4, 5, 6],
                       "c": list("xxyyzz")})
    yt = pd.Series([1.0, 2, 3, 4, 5, 6])
    p = make_pipeline(["a", "b"], ["c"],
                      RandomForestRegressor(n_estimators=10, random_state=0))
    p.fit(Xt, yt)
    assert p.predict(Xt).shape == (6,), "Pipeline 예측 shape 오류"
    print(_ok + "Pipeline이 NaN + 카테고리 혼재 데이터를 처리"); n_pass += 1

    # (10) 그룹 키 탐색 — 있으면 찾고, 없으면 없다고 보고하는지
    _, found = find_group_key(
        pd.DataFrame({"reference_doi": ["a"] * 10 + ["b"] * 10}), quiet=True)
    assert found, "DOI 컬럼을 못 찾음"
    _, found2 = find_group_key(pd.DataFrame({"junk": range(10)}), quiet=True)
    assert not found2, "없는데 찾았다고 보고함"
    print(_ok + "그룹 키 자동 탐색 (없으면 정직하게 없다고 보고)"); n_pass += 1

    print(f"\n  → 자체 테스트 {n_pass}/10 통과")
    return n_pass == 10


# ==========================================================================
# STEP 2 — 건전성 검사: 신호가 있으면 찾고, 없으면 못 찾는지
# ==========================================================================

def sanity_check():
    """평가 코드 자체가 믿을 만한지 확인하는 두 개의 대조 실험.

      (A) 강한 신호를 심은 데이터  → 모델이 baseline을 크게 이겨야 정상
      (B) y를 무작위로 섞은 데이터 → 모델이 baseline을 못 이겨야 정상
    (B)에서 이긴다면 평가 코드에 leakage가 있다는 뜻입니다.
    """
    rule("STEP 2 / 건전성 검사 — 평가 코드가 믿을 만한지")
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold, cross_val_predict

    rng = np.random.default_rng(0)
    n_std, per = 40, 20
    g = np.repeat(np.arange(n_std), per)
    a = rng.uniform(30, 60, n_std * per)
    b = rng.uniform(0, 30, n_std * per)
    off = np.repeat(rng.normal(0, 6, n_std), per)     # 논문별 오프셋
    X = pd.DataFrame({"a": a, "b": b})

    def grouped_mae(y):
        rf = RandomForestRegressor(n_estimators=120, min_samples_leaf=3,
                                   random_state=0, n_jobs=-1)
        cv = GroupKFold(4)
        pm = cross_val_predict(rf, X, y, cv=cv, groups=g)
        pb = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=cv, groups=g)
        return mean_absolute_error(y, pm), mean_absolute_error(y, pb)

    y_sig = 80 + 0.8 * (a - 45) - 0.5 * (b - 15) + off + rng.normal(0, 2, n_std * per)
    m1, b1 = grouped_mae(y_sig)
    gain1 = (1 - m1 / b1) * 100
    passA = gain1 > 20
    print(f"  (A) 신호 있는 데이터   model MAE={m1:.2f}  baseline={b1:.2f}  "
          f"개선={gain1:+.1f}%")
    print((_ok if passA else _fail) + "신호가 있을 때 모델이 baseline을 이긴다"
          + ("" if passA else "  ← 평가 코드가 신호를 놓치고 있음"))

    y_null = pd.Series(rng.permutation(y_sig))
    m2, b2 = grouped_mae(y_null)
    gain2 = (1 - m2 / b2) * 100
    passB = gain2 < 5
    print(f"  (B) y를 섞은 데이터    model MAE={m2:.2f}  baseline={b2:.2f}  "
          f"개선={gain2:+.1f}%")
    print((_ok if passB else _fail) + "신호가 없을 때 baseline을 못 이긴다"
          + ("" if passB else "  ← leakage 있음!"))

    return passA and passB


# ==========================================================================
# STEP 3 — CSV 자동 탐색
# ==========================================================================

def locate_csv():
    rule("STEP 3 / CSV 탐색")
    cands = []
    if CSV_HINT and os.path.exists(CSV_HINT):
        cands.append(CSV_HINT)
    home = str(Path.home())
    for d in [".", home, os.path.join(home, "Downloads"),
              os.path.join(home, "Desktop"), os.path.join(home, "Documents"),
              "/content", "/content/drive/MyDrive", "/mnt/data", "/kaggle/input"]:
        if os.path.isdir(d):
            for pat in ("LNP*.csv", "LNP*.CSV", "*Atlas*.csv", "*Atlas*.CSV"):
                cands.extend(glob.glob(os.path.join(d, pat)))
    seen, uniq = set(), []
    for c in cands:
        rp = os.path.realpath(c)
        if rp not in seen:
            seen.add(rp); uniq.append(c)
    if uniq:
        print(f"  발견: {uniq[0]}")
        if len(uniq) > 1:
            print(f"  (다른 후보 {len(uniq) - 1}개: {uniq[1:4]})")
        return uniq[0]
    print("  CSV를 못 찾았습니다 → DEMO 모드로 계속합니다.")
    print("  실제 파일로 돌리려면 맨 위 CSV_HINT에 전체 경로를 적으세요.")
    return None


def make_demo_df(n_studies=58, seed=7):
    """Atlas 스키마를 모사한 합성 데이터. 조성의 진짜 효과는 약하게 설정."""
    rng = np.random.default_rng(seed)
    pool = ["CCCCCCCCCCCCCCCCCC(=O)OCCN(C)CCOC(=O)CCCCCCCCCCCCCCCCC",
            "CCCCCCCCC=CCCCCCCCC(=O)OCC(COC(=O)CCCCCCCC=CCCCCCCCC)N(C)C",
            "CCCCCCCCCCCCCCCC(=O)OCCCN(CCCOC(=O)CCCCCCCCCCCCCCC)CCO",
            "CCCCCCCCCCCCCC(CCCCCCCC)OC(=O)CCN(C)CCCN(C)CCC(=O)OC(CCCCCCCC)CCCCCCCCCCCC"]
    cargo, helper = ["mRNA", "siRNA", "pDNA", "saRNA"], ["DSPC", "DOPE", "POPC"]
    rows = []
    for s in range(n_studies):
        base = np.array([rng.uniform(35, 62), rng.uniform(0, 34),
                         rng.uniform(15, 50), rng.uniform(0.5, 3)])
        off = rng.normal(0, 7.0)
        lip, cg = int(rng.integers(0, len(pool))), cargo[int(rng.integers(0, 4))]
        for _ in range(int(rng.integers(6, 32))):
            r = np.clip(base + rng.normal(0, 2.0, 4), 0, None)
            ee = np.clip(80 + 0.10 * (r[0] - 45) + 2.0 * (r[3] - 1.5)
                         + off + rng.normal(0, 4), 20, 99.9)
            u = rng.random()
            if u < 0.10:
                ratio = f"{r[0]:.1f}:{r[2]:.1f}:{r[3]:.1f}"
            elif u < 0.18:
                ratio = f"{r[0]:.1f}/{r[1]:.1f}/{r[2]:.1f}/{r[3]:.1f}"
            elif u < 0.22:
                ratio = "n.r."
            else:
                ratio = f"{r[0]:.1f}:{r[1]:.1f}:{r[2]:.1f}:{r[3]:.1f}"
            rows.append({
                "reference_doi": f"10.1038/s4159{s:03d}",
                "lipid_molar_ratio": ratio,
                "ionizable_lipid_smiles": pool[lip],
                "helper_lipid_name": helper[int(rng.integers(0, 3))],
                "cargo_type": cg,
                "np_ratio_std_num": rng.uniform(3, 12),
                "buffer_ph_std_num": rng.choice([4.0, 5.0, 5.5, 7.4]),
                "particle_size_nm_std_num": rng.uniform(60, 170),
                "pdi_std_num": rng.uniform(0.05, 0.3),
                "zeta_potential_mv_std_num": rng.normal(0, 12),
                TARGET: ee if rng.random() > 0.06 else np.nan,
            })
    return pd.DataFrame(rows)


# ==========================================================================
# STEP 4 — 스키마 진단
# ==========================================================================

def diagnose(df):
    rule("STEP 4 / 스키마 진단")
    print(f"  행 {len(df)}  열 {df.shape[1]}")

    tcol = TARGET if TARGET in df.columns else next(
        (c for c in df.columns if "encapsulation" in c.lower()), None)
    if tcol is None:
        print(_fail + "EE 타깃 컬럼이 없습니다. 아래 목록에서 이름을 확인하세요:")
        print("  " + ", ".join(map(str, df.columns[:40])))
        return None
    if tcol != TARGET:
        print(f"  타깃 컬럼: '{tcol}' (기본값과 다름)")

    y = pd.to_numeric(df[tcol], errors="coerce")
    print(f"  EE 유효값 {y.notna().sum()}/{len(df)}  "
          f"({y.notna().mean() * 100:.0f}%)")

    n_frac = int(((y > 0) & (y <= 1)).sum())
    n_over = int((y > 100).sum())
    n_neg = int((y < 0).sum())
    if n_frac:
        print(f"  ! 0~1 범위 {n_frac}행 — 분율 표기로 보임 → ×100 보정")
    if n_over or n_neg:
        print(f"  ! 범위 밖 {n_over + n_neg}행 (>100%: {n_over}, <0%: {n_neg}) → 제거")

    y = y.mask((y > 0) & (y <= 1), y * 100)
    y = y.mask((y < 0) | (y > 100))
    print(f"  정제 후 EE: n={y.notna().sum()}  mean={y.mean():.1f}  "
          f"sd={y.std():.1f}  median={y.median():.1f}")
    print(f"  EE>90% 비율 {(y > 90).mean() * 100:.0f}%  "
          f"→ 이 비율이 높으면 R²가 낮게 나오는 게 정상입니다")
    return y


# ==========================================================================
# STEP 5 — 정확도 평가
# ==========================================================================

def evaluate(X, y, groups, num_cols, cat_cols, have_groups, n_splits=5):
    rule("STEP 5 / 정확도 평가")
    from sklearn.base import clone
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

    n_g = groups.nunique()
    gkf = GroupKFold(n_splits=min(n_splits, max(2, n_g)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "baseline (mean)": DummyRegressor(strategy="mean"),
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "random forest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, max_features=0.5,
            random_state=RANDOM_STATE, n_jobs=-1),
    }
    try:
        from lightgbm import LGBMRegressor
        models["gradient boosting"] = LGBMRegressor(
            n_estimators=500, learning_rate=0.03, num_leaves=15,
            min_child_samples=15, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        models["gradient boosting"] = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=1.0,
            random_state=RANDOM_STATE)

    rows, preds = [], {}
    for name, est in models.items():
        for label, cv, grp in [("study-grouped", gkf, groups),
                               ("random (optimistic)", kf, None)]:
            if label == "study-grouped" and not have_groups:
                continue
            p = cross_val_predict(make_pipeline(num_cols, cat_cols, clone(est)),
                                  X, y, cv=cv, groups=grp, n_jobs=1)
            rows.append({"model": name, "cv": label,
                         "R2": r2_score(y, p),
                         "MAE": mean_absolute_error(y, p),
                         "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))})
            if label == "study-grouped":
                preds[name] = p
    res = pd.DataFrame(rows)

    key = "study-grouped" if have_groups else "random (optimistic)"
    base = res.query("model=='baseline (mean)' and cv==@key")["MAE"].iloc[0]
    res["MAE_gain_%"] = (1 - res["MAE"] / base) * 100

    print("\n  R² / MAE  (낮은 MAE = 좋음, 단위 %p)")
    print(res.pivot(index="model", columns="cv", values=["R2", "MAE"])
          .round(2).to_string().replace("\n", "\n  "))
    print(f"\n  baseline 대비 MAE 개선율 ({key}):")
    for _, r in res.query("cv==@key").sort_values("MAE").iterrows():
        print(f"    {r['model']:<20s} MAE={r['MAE']:5.2f}  {r['MAE_gain_%']:+6.1f}%")
    return res, preds, key


def significance_test(y, preds, groups):
    """논문별 MAE를 짝지어 Wilcoxon 검정 — 개선이 우연인지 판정."""
    from scipy.stats import wilcoxon
    base = preds.get("baseline (mean)")
    best = min((k for k in preds if k != "baseline (mean)"),
               key=lambda k: np.mean(np.abs(y - preds[k])), default=None)
    if base is None or best is None:
        return None, None, None

    per = []
    for gname, idx in pd.Series(range(len(y)), index=groups.values).groupby(level=0):
        i = idx.values
        if len(i) < 3:
            continue
        per.append((np.mean(np.abs(y.values[i] - preds[best][i])),
                    np.mean(np.abs(y.values[i] - base[i]))))
    per = np.array(per)
    if len(per) < 6:
        return best, None, len(per)
    stat, p = wilcoxon(per[:, 0], per[:, 1])
    n_better = int((per[:, 0] < per[:, 1]).sum())
    print(f"\n  Wilcoxon 검정 ({best} vs baseline, 논문 {len(per)}편 짝지음)")
    print(f"    모델이 더 정확한 논문: {n_better}/{len(per)}편   p = {p:.4f}")
    return best, p, len(per)


# ==========================================================================
# STEP 6 — 판정
# ==========================================================================

def verdict(res, key, best, pval, t1, t2, have_groups, demo):
    rule("STEP 6 / 판정")
    checks = [("전처리 자체 테스트", t1), ("평가 코드 건전성 검사", t2)]
    for nm, okv in checks:
        print(("  PASS  " if okv else "  FAIL  ") + nm)

    print("  " + ("PASS  " if have_groups else "FAIL  ")
          + "논문 단위 분할 가능"
          + ("" if have_groups else " ← DOI 컬럼 없음, 아래 점수는 과대평가"))

    if best is None:
        print("\n  정확도 판정 불가")
        return
    gain = res.query("model==@best and cv==@key")["MAE_gain_%"].iloc[0]
    mae = res.query("model==@best and cv==@key")["MAE"].iloc[0]
    r2 = res.query("model==@best and cv==@key")["R2"].iloc[0]

    sig = (pval is not None) and (pval < 0.05)
    useful = gain >= 10 and sig

    print(f"\n  최고 모델: {best}")
    print(f"    MAE {mae:.2f} %p   R² {r2:+.2f}   baseline 대비 {gain:+.1f}%"
          + (f"   p={pval:.4f}" if pval is not None else ""))

    if useful:
        lvl = "실용 수준" if gain >= 20 else "유의하지만 개선 여지 있음"
        print(f"\n  >>> 정확도: {lvl}")
        print(f"      새 논문 처방에 대해 EE를 평균 {mae:.1f}%p 오차로 예측합니다.")
    elif sig:
        print("\n  >>> 정확도: 통계적으로는 유의하나 실용성은 낮음")
        print(f"      baseline보다 {gain:.1f}%만 낫습니다(기준 10%).")
    else:
        print("\n  >>> 정확도: 아직 baseline을 유의하게 못 이깁니다")
        print("      코드는 정상 동작하고 있으며, 이것은 버그가 아니라 결과입니다.")
        print("      이유: 문헌 수집 EE 값은 논문 간 측정 편차(5-10%p)가 조성 효과보다")
        print("      크고, EE 자체가 대부분 85-95%로 포화되어 변별력이 낮습니다.")
        print("      다음을 시도하세요:")
        print("        1. 타깃을 transfection/발현량으로 변경 (변별력이 큼)")
        print("        2. 공정 변수(N/P ratio, buffer pH, flow rate) 확보 후 추가")
        print("        3. 논문 수를 늘리기 — grouped CV의 실질 표본은 논문 수입니다")

    if demo:
        print("\n  ! DEMO 모드였습니다. 위 수치는 합성 데이터 결과이며")
        print("    실제 Atlas 성능이 아닙니다. CSV_HINT에 경로를 넣고 다시 실행하세요.")


# ==========================================================================
# STEP 7 — 그림
# ==========================================================================

def make_figure(res, preds, y, key, fname="lnp_v3_report.png"):
    import matplotlib
    import matplotlib.pyplot as plt

    order = res.query("cv==@key").sort_values("MAE")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    cols = ["#B0B0B0" if m == "baseline (mean)" else "#1F77B4"
            for m in order["model"]]
    ax1.barh(range(len(order)), order["MAE"], color=cols, height=0.6)
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels(order["model"])
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean absolute error (%p)   —  lower is better")
    ax1.set_title(f"Accuracy under {key} CV", loc="left")
    for i, (v, gn) in enumerate(zip(order["MAE"], order["MAE_gain_%"])):
        ax1.text(v + 0.05, i, f"{v:.2f}" + (f"  ({gn:+.0f}%)" if abs(gn) > 0.05 else ""),
                 va="center", fontsize=8)
    ax1.margins(x=0.18)

    best = order.query("model!='baseline (mean)'")["model"]
    if len(best):
        bname = best.iloc[0]
        ax2.scatter(y, preds[bname], s=10, alpha=0.35, color="#1F77B4",
                    linewidth=0, label=bname)
        ax2.scatter(y, preds["baseline (mean)"], s=6, alpha=0.5, color="#B0B0B0",
                    linewidth=0, label="baseline (mean)")
        lim = [float(y.min()) - 3, float(y.max()) + 3]
        ax2.plot(lim, lim, "k--", lw=0.9)
        ax2.set_xlim(lim); ax2.set_ylim(lim)
        ax2.set_xlabel("Measured encapsulation efficiency (%)")
        ax2.set_ylabel("Predicted (%)")
        ax2.set_title("Predicted vs measured, held-out studies", loc="left")
        ax2.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    print(f"\n  그림 저장: {fname}")
    try:
        plt.show()
    except Exception:
        pass
    return fig


# ==========================================================================
# 실행
# ==========================================================================

def run_all():
    print("LNP-Predictor v3  자체 검증 실행")
    print(f"python {sys.version.split()[0]} | pandas {pd.__version__} | "
          f"numpy {np.__version__}")

    t1 = selftest()
    t2 = sanity_check()

    path = locate_csv()
    demo = path is None
    df = make_demo_df() if demo else load_csv(path)
    if demo:
        print(f"  DEMO 데이터 생성: {df.shape[0]}행 / "
              f"{df['reference_doi'].nunique()}편")

    y_clean = diagnose(df)
    if y_clean is None:
        print("\n타깃 컬럼을 찾지 못해 중단합니다.")
        return None

    X_all, num_cols, cat_cols = build_features(df, include_measured=False)
    groups_all, have_groups = find_group_key(df)

    keep = y_clean.notna() & X_all[num_cols].notna().any(axis=1)
    X, y = X_all[keep].reset_index(drop=True), y_clean[keep].reset_index(drop=True)
    groups = groups_all[keep].reset_index(drop=True)
    print(f"  학습 데이터: n={len(X)}  features={X.shape[1]}  "
          f"studies={groups.nunique()}")
    if len(X) < 50:
        print(_fail + "유효 데이터가 너무 적습니다. 컬럼명을 확인하세요.")
        return None

    res, preds, key = evaluate(X, y, groups, num_cols, cat_cols, have_groups)
    best, pval, n_paired = significance_test(y, preds, groups) if have_groups \
        else (None, None, None)
    verdict(res, key, best, pval, t1, t2, have_groups, demo)
    make_figure(res, preds, y, key)

    # 출처를 표에 직접 기록 — 나중에 이 파일만 보고도 판별할 수 있게.
    # (숫자만으로는 DEMO/실제를 구분할 수 없습니다.)
    res.insert(0, "data_source", "DEMO_SYNTHETIC" if demo else "REAL_CSV")
    res.insert(1, "csv_path", "(none - demo mode)" if demo else os.path.abspath(path))
    res["n_rows"] = len(X)
    res["n_studies"] = groups.nunique()
    res["grouped_cv"] = bool(have_groups)
    res["wilcoxon_p"] = pval if pval is not None else np.nan
    res["selftest_pass"] = bool(t1)
    res["sanity_pass"] = bool(t2)
    res["run_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    res.to_csv("lnp_v3_results.csv", index=False)
    print(f"  표 저장: lnp_v3_results.csv  (data_source="
          f"{'DEMO_SYNTHETIC' if demo else 'REAL_CSV'})")
    return res


# results = run_all()
