# ==========================================================================
#  LNP-Entry  —  논문에서 데이터를 입력하는 도구
#  ------------------------------------------------------------------------
#  "논문에서 무슨 데이터를 어떻게 넣지?" 를 해결합니다.
#
#    1) make_template()      입력 서식 CSV 생성 (엑셀로 열어서 채우기)
#    2) add(...)             노트북에서 한 줄씩 추가 (SMILES 자동 조회)
#    3) resolve_smiles()     지질 이름 → SMILES 자동 변환 (PubChem)
#    4) validate()           입력 오류를 저장 전에 잡아냄
#    5) save()               v3/v4가 바로 읽는 CSV로 저장
#
#  핵심: SMILES를 손으로 옮기지 마세요. 지질 이름만 적으면 됩니다.
#        (DLin-MC3-DMA, ALC-0315, SM-102 등 12개 중 11개 자동 조회 확인)
# ==========================================================================

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# --- v3/v4가 읽는 컬럼 이름 (이대로 쓰면 자동 인식됩니다) ----------------
COLS = [
    # [필수] 이 4개가 없으면 모델을 못 돌립니다
    "reference_doi",                            # 논문 DOI — 논문 단위 CV의 핵심
    "lipid_molar_ratio",                        # "50:10:38.5:1.5" (ion:helper:chol:peg)
    "ionizable_lipid_name",                     # "DLin-MC3-DMA" → SMILES 자동 조회
    "encapsulation_efficiency_percent_std_num",  # EE (%)  — 예측 타깃
    # [권장] 있으면 정확도가 오릅니다
    "np_ratio_std_num",                         # N/P 비
    "buffer_ph_std_num",                        # 혼합 시 완충액 pH (보통 4.0)
    "cargo_type",                               # mRNA / siRNA / pDNA / saRNA
    "helper_lipid_name",                        # DSPC / DOPE / ...
    "peg_lipid_name",                           # DMG-PEG2000 / ALC-0159 / ...
    # [선택] 사후 측정값 — 설계 예측에는 안 쓰이고 진단용입니다
    "particle_size_nm_std_num",
    "pdi_std_num",
    "zeta_potential_mv_std_num",
    # [자동] resolve_smiles()가 채웁니다 — 비워 두세요
    "ionizable_lipid_smiles",
    # [메모] 나중에 원본을 찾아갈 때 쓰는 자유 기입란
    "source_note",                              # "Table 2, row 3" 등
]

REQUIRED = ["reference_doi", "lipid_molar_ratio",
            "encapsulation_efficiency_percent_std_num"]

CARGO_OK = {"mrna", "sirna", "pdna", "sarna", "asomrna", "aso", "circrna", "dna", "rna"}

# 자주 쓰는 지질의 SMILES — 네트워크 없이도 바로 동작하는 캐시
LIPID_CACHE = {}
_CACHE_FILE = "lnp_lipid_smiles_cache.json"


# ==========================================================================
# 1. 입력 서식 만들기
# ==========================================================================

def make_template(path="lnp_input_template.csv", n_rows=10, with_example=True):
    """엑셀로 열어서 채울 빈 서식 CSV를 만든다.

    첫 줄에 예시 한 줄이 들어갑니다(지우고 쓰세요).
    utf-8-sig 로 저장하므로 한글 엑셀에서 안 깨집니다.
    """
    df = pd.DataFrame({c: [""] * n_rows for c in COLS})
    if with_example:
        ex = {
            "reference_doi": "10.1038/s41586-021-03534-y",
            "lipid_molar_ratio": "50:10:38.5:1.5",
            "ionizable_lipid_name": "SM-102",
            "encapsulation_efficiency_percent_std_num": "94.2",
            "np_ratio_std_num": "6",
            "buffer_ph_std_num": "4.0",
            "cargo_type": "mRNA",
            "helper_lipid_name": "DSPC",
            "peg_lipid_name": "DMG-PEG2000",
            "particle_size_nm_std_num": "82",
            "pdi_std_num": "0.12",
            "zeta_potential_mv_std_num": "-2.1",
            "ionizable_lipid_smiles": "",
            "source_note": "<-- 예시입니다. 지우고 쓰세요. Table 2 row 1",
        }
        for k, v in ex.items():
            df.loc[0, k] = v
    df.to_csv(path, index=False, encoding="utf-8-sig")

    print(f"서식 생성: {os.path.abspath(path)}")
    print("\n논문에서 찾아야 할 것 — 보통 Methods 와 Table 1~2 에 있습니다:")
    print("  [필수] DOI            : 논문 첫 페이지. 같은 논문 행은 DOI가 같아야 합니다")
    print("  [필수] 몰비            : 'lipid:DSPC:cholesterol:PEG = 50:10:38.5:1.5'")
    print("                          → 그대로 '50:10:38.5:1.5' 로 적으세요")
    print("  [필수] EE (%)         : 'encapsulation efficiency 94%' → 94")
    print("  [권장] 이온화지질 이름 : 'SM-102' — SMILES는 자동 조회됩니다")
    print("  [권장] N/P 비, pH     : Methods 의 mixing 조건")
    print("  [권장] cargo          : mRNA / siRNA / pDNA")
    print("\n한 논문에서 처방을 여러 개 보고했으면 각각 한 줄씩, DOI는 동일하게.")
    print("모르는 값은 비워 두세요 (0 이나 'N/A' 로 채우지 마세요 — 왜곡됩니다).")
    return path


# ==========================================================================
# 2. 노트북에서 한 줄씩 추가
# ==========================================================================

class Entry:
    """논문을 읽으며 한 줄씩 쌓는 수집기.

        e = Entry()
        e.paper("10.1038/xxx")                     # 이후 행에 DOI 자동 적용
        e.add("50:10:38.5:1.5", 94.2, ion="SM-102", np_ratio=6, cargo="mRNA")
        e.add("35:16:46.5:2.5", 88.0, ion="SM-102", np_ratio=6, cargo="mRNA")
        e.save()                                   # 검증 후 CSV 저장
    """

    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self._doi = None
        self._defaults = {}

    def paper(self, doi, **defaults):
        """이 논문의 DOI와, 이후 행에 공통 적용할 값을 설정."""
        self._doi = str(doi).strip()
        self._defaults = defaults
        print(f"논문 설정: {self._doi}" +
              (f"  공통값 {defaults}" if defaults else ""))
        return self

    def add(self, ratio, ee, ion=None, helper=None, peg=None, cargo=None,
            np_ratio=None, ph=None, size=None, pdi=None, zeta=None,
            doi=None, note=None, **extra):
        """처방 한 줄 추가. 모르는 값은 그냥 생략하세요."""
        r = {c: np.nan for c in COLS}
        r.update(self._defaults)
        r["reference_doi"] = doi or self._doi
        if not r["reference_doi"]:
            raise ValueError("DOI가 없습니다. e.paper('10.xxxx/yyy') 를 먼저 부르세요.")
        r["lipid_molar_ratio"] = str(ratio).strip()
        r["encapsulation_efficiency_percent_std_num"] = ee
        for k, v in [("ionizable_lipid_name", ion), ("helper_lipid_name", helper),
                     ("peg_lipid_name", peg), ("cargo_type", cargo),
                     ("np_ratio_std_num", np_ratio), ("buffer_ph_std_num", ph),
                     ("particle_size_nm_std_num", size), ("pdi_std_num", pdi),
                     ("zeta_potential_mv_std_num", zeta), ("source_note", note)]:
            if v is not None:
                r[k] = v
        r.update(extra)
        self.rows.append(r)
        print(f"  + [{len(self.rows):3d}] {r['lipid_molar_ratio']:<22s} EE={ee}"
              f"{'  ' + str(ion) if ion else ''}")
        return self

    def to_frame(self):
        return pd.DataFrame(self.rows, columns=COLS + [
            c for r in self.rows for c in r if c not in COLS])

    def save(self, path="lnp_my_data.csv", lookup=True, strict=False):
        df = self.to_frame()
        if lookup:
            df = resolve_smiles(df)
        ok = validate(df, strict=strict)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n저장: {os.path.abspath(path)}   ({len(df)}행 / "
              f"{df['reference_doi'].nunique()}편)")
        if not ok:
            print("! 경고가 있습니다. 위 [검증] 내용을 확인하세요.")
        return path


# ==========================================================================
# 3. 지질 이름 → SMILES 자동 조회
# ==========================================================================

def _load_cache():
    global LIPID_CACHE
    if not LIPID_CACHE and os.path.exists(_CACHE_FILE):
        try:
            LIPID_CACHE = json.load(open(_CACHE_FILE, encoding="utf-8"))
        except Exception:
            LIPID_CACHE = {}
    return LIPID_CACHE


def _save_cache():
    try:
        json.dump(LIPID_CACHE, open(_CACHE_FILE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=0)
    except Exception:
        pass


def pubchem_lookup(name, timeout=20):
    """지질 이름 → (CID, SMILES). PubChem 응답 키 변경에 무관하게 동작.

    2025년 PubChem이 CanonicalSMILES 를 ConnectivitySMILES 로 개명해서,
    키 이름을 고정하면 조용히 실패합니다. 여러 키를 순서대로 시도합니다.
    """
    q = urllib.parse.quote(str(name).strip())
    for path in (f"compound/name/{q}", f"compound/fastformula/{q}"):
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/{path}"
               f"/property/SMILES,MolecularFormula/JSON")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                p = json.load(r)["PropertyTable"]["Properties"][0]
            smi = next((p[k] for k in ("SMILES", "ConnectivitySMILES",
                                       "IsomericSMILES", "CanonicalSMILES")
                        if k in p and p[k]), None)
            if smi:
                return p.get("CID"), smi
        except Exception:
            continue
    return None, None


def resolve_smiles(df, name_col="ionizable_lipid_name",
                   smi_col="ionizable_lipid_smiles", pause=0.25, verbose=True):
    """이름 컬럼을 읽어 SMILES 컬럼을 채운다. 이미 있는 값은 건드리지 않음.

    같은 이름은 한 번만 조회하고 캐시에 남깁니다(재실행 시 네트워크 불필요).
    """
    df = df.copy()
    if name_col not in df.columns:
        if verbose:
            print(f"[SMILES] '{name_col}' 컬럼이 없어 건너뜁니다.")
        return df
    # 빈 컬럼이 float64로 만들어지면 문자열 대입이 막히므로 object로 고정
    if smi_col not in df.columns:
        df[smi_col] = pd.Series([np.nan] * len(df), index=df.index, dtype=object)
    else:
        df[smi_col] = df[smi_col].astype(object)

    cache = _load_cache()
    need = sorted({str(n).strip() for n, s in
                   zip(df[name_col], df[smi_col])
                   if str(n).strip() not in ("", "nan", "None")
                   and (pd.isna(s) or str(s).strip() == "")})
    if not need:
        if verbose:
            print("[SMILES] 조회할 항목 없음 (이미 모두 채워짐)")
        return df

    if verbose:
        print(f"[SMILES] {len(need)}개 이름 조회 중...")
    miss = []
    for n in need:
        if n in cache:
            continue
        cid, smi = pubchem_lookup(n)
        if smi:
            cache[n] = smi
            if verbose:
                print(f"   OK   {n:<18s} CID={cid}")
        else:
            miss.append(n)
            if verbose:
                print(f"   MISS {n:<18s} <- 직접 넣거나 이름 표기를 바꿔보세요")
        time.sleep(pause)
    _save_cache()

    fill = df[name_col].astype(str).str.strip().map(cache)
    cur = df[smi_col]
    blank = cur.isna() | (cur.astype(str).str.strip().isin(["", "nan", "None"]))
    df[smi_col] = np.where(blank & fill.notna(), fill, cur).astype(object)
    got = df[smi_col].notna().sum()
    if verbose:
        print(f"[SMILES] 완료: {got}/{len(df)}행 확보"
              + (f"   미해결 이름: {miss}" if miss else ""))
        if miss:
            print("   → PubChem 검색창에 이름을 넣어 정식 명칭을 확인하거나,")
            print("     논문 SI의 구조식을 보고 SMILES를 직접 넣으세요.")
            print("     LIPID_CACHE['이름'] = 'SMILES...' 로 등록하면 다음부터 자동입니다.")
    return df


# ==========================================================================
# 4. 검증 — 저장 전에 실수를 잡는다
# ==========================================================================

def validate(df, strict=False):
    """입력 오류를 찾아 보고한다. 통과 여부를 bool 로 반환."""
    print("\n[검증]")
    ok = True

    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        print(f"  FAIL  필수 컬럼 없음: {miss}")
        return False

    # 필수값 결측
    for c in REQUIRED:
        n = df[c].isna().sum() + (df[c].astype(str).str.strip() == "").sum()
        if n:
            print(f"  FAIL  '{c}' 비어 있는 행 {n}개")
            ok = False

    # 몰비 파싱
    parts = (df["lipid_molar_ratio"].astype(str).str.strip()
             .str.replace(r"[\/\-,;|]", ":", regex=True).str.split(":", expand=True))
    num = parts.apply(pd.to_numeric, errors="coerce")
    ncomp = num.notna().sum(axis=1)
    bad = df.index[~ncomp.isin([3, 4])]
    if len(bad):
        print(f"  FAIL  몰비를 못 읽은 행 {len(bad)}개 → "
              f"{list(df.loc[bad, 'lipid_molar_ratio'].head(3))}")
        print("        형식: '50:10:38.5:1.5' (이온화:헬퍼:콜레스테롤:PEG)")
        ok = False
    tot = num.sum(axis=1, min_count=1)
    odd = df.index[(ncomp == 4) & ((tot < 80) | (tot > 120))]
    if len(odd):
        print(f"  경고  몰비 합이 100에서 먼 행 {len(odd)}개 (합계 예: "
              f"{[round(x,1) for x in tot[odd].head(3)]}) — 자동 정규화되지만 확인 권장")

    # EE 범위
    ee = pd.to_numeric(df["encapsulation_efficiency_percent_std_num"], errors="coerce")
    n_nan = ee.isna().sum()
    n_frac = int(((ee > 0) & (ee <= 1)).sum())
    n_out = int(((ee < 0) | (ee > 100)).sum())
    if n_nan:
        print(f"  FAIL  EE를 숫자로 못 읽은 행 {n_nan}개 ('94%' 말고 94 로 적으세요)")
        ok = False
    if n_frac:
        print(f"  경고  EE가 0~1 범위인 행 {n_frac}개 — 분율 표기로 보입니다 "
              f"(0.94 → 94). v3가 ×100 보정합니다")
    if n_out:
        print(f"  FAIL  EE 범위 밖 {n_out}행 (<0 또는 >100)")
        ok = False

    # 논문 수 — 그룹 CV의 실질 표본
    k = df["reference_doi"].astype(str).str.strip().str.lower().nunique()
    per = df.groupby(df["reference_doi"].astype(str).str.strip().str.lower()).size()
    print(f"  정보  {len(df)}행 / {k}편   논문당 중앙값 {per.median():.0f}행 "
          f"(최소 {per.min()}, 최대 {per.max()})")
    if k < 10:
        print(f"  경고  논문이 {k}편뿐입니다. 논문 단위 5-fold CV에는 "
              f"최소 10편, 안정적으로는 20편 이상 필요합니다.")
        print("        (그룹 CV에서 실질 표본 수는 '행 수'가 아니라 '논문 수'입니다)")
        ok = ok and not strict

    # DOI 형식
    doi = df["reference_doi"].astype(str).str.strip()
    weird = doi[~doi.str.match(r"^(10\.\d{4,9}/|PMID|pmid|https?://)")]
    if len(weird):
        print(f"  경고  DOI 형식이 아닌 값 {len(weird)}개: {list(weird.unique()[:3])}")
        print("        같은 논문끼리 문자열이 완전히 같기만 하면 동작은 합니다.")

    # cargo 표기 흔들림
    if "cargo_type" in df.columns:
        cg = df["cargo_type"].dropna().astype(str).str.strip()
        cg = cg[cg != ""]
        unknown = sorted({c for c in cg.unique()
                          if c.lower().replace("-", "").replace(" ", "") not in CARGO_OK})
        if unknown:
            print(f"  경고  낯선 cargo 표기: {unknown} — 표기를 통일하세요 "
                  f"('mRNA' 와 'mrna' 는 다른 범주로 취급됩니다)")

    # SMILES
    if "ionizable_lipid_smiles" in df.columns:
        s = df["ionizable_lipid_smiles"].astype(str).str.strip()
        n_s = int(((s != "") & (s.str.lower() != "nan")).sum())
        print(f"  정보  SMILES 확보 {n_s}/{len(df)}행"
              + ("  (resolve_smiles() 로 채우세요)" if n_s < len(df) else ""))

    print("  " + ("PASS  저장 가능합니다." if ok else
                  "FAIL  위 FAIL 항목을 고치고 다시 저장하세요."))
    return ok


# ==========================================================================
# 5. 기존 파일 이어붙이기 / 컬럼 이름 자동 정렬
# ==========================================================================

ALIASES = {
    "doi": "reference_doi", "reference": "reference_doi", "paper": "reference_doi",
    "ratio": "lipid_molar_ratio", "molar_ratio": "lipid_molar_ratio",
    "lipid_ratio": "lipid_molar_ratio", "composition": "lipid_molar_ratio",
    "ee": "encapsulation_efficiency_percent_std_num",
    "ee_percent": "encapsulation_efficiency_percent_std_num",
    "encapsulation": "encapsulation_efficiency_percent_std_num",
    "encapsulation_efficiency": "encapsulation_efficiency_percent_std_num",
    "ionizable": "ionizable_lipid_name", "ionizable_lipid": "ionizable_lipid_name",
    "lipid": "ionizable_lipid_name", "smiles": "ionizable_lipid_smiles",
    "helper": "helper_lipid_name", "peg": "peg_lipid_name",
    "cargo": "cargo_type", "payload": "cargo_type",
    "np": "np_ratio_std_num", "n_p": "np_ratio_std_num", "np_ratio": "np_ratio_std_num",
    "ph": "buffer_ph_std_num", "buffer_ph": "buffer_ph_std_num",
    "size": "particle_size_nm_std_num", "diameter": "particle_size_nm_std_num",
    "pdi": "pdi_std_num", "zeta": "zeta_potential_mv_std_num",
}


def normalize_columns(df, verbose=True):
    """느슨하게 적은 컬럼 이름을 v3/v4가 아는 이름으로 바꾼다."""
    ren = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_").replace("(%)", "").strip("_")
        if c in COLS:
            continue
        if key in ALIASES:
            ren[c] = ALIASES[key]
    if ren and verbose:
        print("[컬럼 정렬]", {k: v for k, v in ren.items()})
    return df.rename(columns=ren)


def merge_files(paths, out="lnp_merged.csv", lookup=True):
    """여러 사람이 나눠 입력한 파일을 합친다. 중복 행은 제거."""
    frames = []
    for p in paths:
        for enc in ("utf-8-sig", "cp949", "latin-1"):
            try:
                d = pd.read_csv(p, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        d = normalize_columns(d)
        d["_source_file"] = os.path.basename(p)
        frames.append(d)
        print(f"  읽음 {p}: {len(d)}행")
    df = pd.concat(frames, ignore_index=True)

    key = ["reference_doi", "lipid_molar_ratio",
           "encapsulation_efficiency_percent_std_num"]
    key = [k for k in key if k in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="first")
    if before != len(df):
        print(f"  중복 제거: {before - len(df)}행")

    if lookup:
        df = resolve_smiles(df)
    validate(df)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n병합 저장: {os.path.abspath(out)}  ({len(df)}행 / "
          f"{df['reference_doi'].nunique()}편)")
    return df
