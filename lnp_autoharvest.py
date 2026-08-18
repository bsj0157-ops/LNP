"""lnp_autoharvest — PMC 자동 수집 파이프라인 (앱 내장용)

설계 근거는 6·7차 실측입니다.

  * 1차 LLM 추출만 쓰면 몰비 14%가 원문에 없는 값(환각)이었습니다.
    → 원문 대조 게이트(gate_row) 없이는 절대 투입하지 않습니다.
  * 2차 감사의 교정값은 99%가 원문에 존재했습니다.
    → 추출과 감사를 분리한 2단 구조가 실효가 있습니다.
  * 표 추출은 이 분야에서 거의 실패합니다(426편 중 몰비+EE 동시 표재 6편).
    → 본문(Methods) 경로가 주력입니다.
  * 최종 채택률은 추출 227행 기준 61%, high 등급은 32%입니다.
    → 100행을 얻으려면 약 300편을 봐야 합니다. 진행률 표시가 필요합니다.

앱에서의 사용:

    import lnp_autoharvest as AH
    job = AH.HarvestJob(existing_df=work_df, llm=host.llm, target_rows=40)
    for ev in job.run(max_papers=120):
        st.progress(ev.frac, ev.message)      # 제너레이터 — UI 갱신 가능
    new = job.accepted            # 스키마에 맞는 DataFrame
    rej = job.rejected            # 기각 사유가 붙은 DataFrame

이 모듈은 Streamlit 에 의존하지 않습니다. 진행 상황을 이벤트로 내보내므로
CLI 로도 돌아갑니다.
"""
from __future__ import annotations

import re
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional

import numpy as np
import pandas as pd

# ── 스키마 ──────────────────────────────────────────────────────────────
SCHEMA = [
    "reference_doi", "pmcid", "ionizable_lipid_name", "ionizable_lipid_smiles",
    "lipid_molar_ratio", "helper_lipid_name", "peg_lipid_name", "cargo_type",
    "encapsulation_efficiency_percent_std_num", "np_ratio_std_num",
    "buffer_ph_std_num", "particle_size_nm_std_num", "pdi_std_num",
    "zeta_potential_mv_std_num", "ee_is_approximate", "confidence",
    "evidence", "source", "source_note", "repair_note", "chem_warnings",
    "generic_name",
]

UNNAMED = "Custom lipid"

# 값 범위 — 이 밖은 무조건 기각합니다
RANGES = {
    "encapsulation_efficiency_percent_std_num": (0.0, 100.0),
    "np_ratio_std_num": (0.5, 50.0),
    "buffer_ph_std_num": (3.0, 9.0),
    "particle_size_nm_std_num": (20.0, 500.0),
    "pdi_std_num": (0.0, 1.0),
    "zeta_potential_mv_std_num": (-60.0, 60.0),
}

# 몰비 토큰 — 꼬리가 (?!\d)(?!\.\d) 인 것이 중요합니다.
# 기존 lnp_pdf.RATIO_RE 의 (?![\d.]) 는 문장 끝 마침표에 걸려
# "50:1.5:10:38.5. T-SGR" 에서 마지막 성분 38.5 를 잘라냈습니다.
RATIO_TOKEN = re.compile(
    r"(?<![\d.])(\d{1,3}(?:[.,]\d{1,2})?)"
    r"\s*[:/]\s*(\d{1,3}(?:[.,]\d{1,2})?)"
    r"(?:\s*[:/]\s*(\d{1,3}(?:[.,]\d{1,2})?))?"
    r"(?:\s*[:/]\s*(\d{1,3}(?:[.,]\d{1,2})?))?"
    r"(?:\s*[:/]\s*(\d{1,3}(?:[.,]\d{1,2})?))?"
    r"(?!\d)(?!\.\d)"
)

EE_PAT = re.compile(
    r"(?:encapsulation\s+efficienc\w*|\bEE%?\b|entrapment\s+efficienc\w*)"
    r"[^.\n]{0,120}?(\d{1,3}(?:\.\d{1,2})?)\s*%", re.I)
APPROX_PAT = re.compile(r"(?:>|≥|above|over|exceed\w*|approximately|about|~|nearly|almost)", re.I)

# 총칭 — 이름이 아니므로 UNNAMED 로 접습니다. 모델이 "ionizable lipid",
# "cationic lipid", "lipid A" 같은 총칭을 이름 칸에 넣는 일이 잦습니다.
GENERIC_NAMES = {
    "ionizable lipid", "ionisable lipid", "cationic lipid", "ionizable lipids",
    "novel lipid", "lipid", "lipids", "custom", "custom lipid", "unnamed",
    "not specified", "n/a", "na", "none", "unknown", "test lipid",
    "lead lipid", "candidate lipid", "amino lipid", "ionizable amino lipid",
}

# EE 정의가 핵산과 다른 화물 — 같은 표에 섞으면 라벨이 오염됩니다
NUCLEIC_CARGO = {"mRNA", "siRNA", "DNA", "ASO", "saRNA", "circRNA"}

# 이온화지질이 아닌 것 — 이 이름이 나오면 지질명으로 채택하지 않습니다
NOT_IONIZABLE = {
    "dspc", "dope", "dopc", "dppc", "dsPE", "dspe", "cholesterol", "chol",
    "lecithin", "phosphatidylcholine", "dmg-peg2000", "dmg-peg", "peg-dmg",
    "alc-0159", "c14-peg2000", "peg-lipid", "dmpe-peg2000", "peg-c-dma",
    "mrna", "sirna", "dna", "pdna", "asо", "aso", "sarna", "circrna",
}

# 화물 카테고리 — 기존 데이터가 쓰는 4종 + 신규 3종
def cargo_category(s: str) -> str:
    t = str(s or "").lower()
    if not t:
        return ""
    if "sarna" in t or "self-ampl" in t or "srrna" in t:
        return "saRNA"
    if "circrna" in t:
        return "circRNA"
    if "sirna" in t:
        return "siRNA"
    if any(k in t for k in ("sgrna", "crispr", "cas9", "abe ")):
        return "mRNA"
    if any(k in t for k in ("mrna", "spike", "il-12", "il-33", "exotoxin")):
        return "mRNA"
    if "antisense" in t or re.search(r"\baso\b", t):
        return "ASO"
    if any(k in t for k in ("pdna", "plasmid", "dsrna", "odn", "poly")) or t == "dna":
        return "DNA"
    if any(k in t for k in ("protein", "peptide", "ovalbumin", "epo", "antibody",
                            "enzyme", "albumin", "rnp")):
        return "protein"
    if any(k in t for k in ("curcumin", "paclitaxel", "doxorubicin", "dexamethasone",
                            "small molecule", "drug")):
        return "small molecule"
    return "other"


# 이온화지질 표기 통일 — 기존 데이터가 SM-102 와 SM102 를 함께 쓰고 있어
# 같은 물질이 두 카테고리로 갈립니다(원핫 인코딩에서 서로 다른 지질이 됩니다).
ION_CANON = {
    "sm102": "SM-102", "sm-102": "SM-102",
    "alc0315": "ALC-0315", "alc-0315": "ALC-0315",
    "mc3": "DLin-MC3-DMA", "mc-3": "DLin-MC3-DMA", "dlinmc3dma": "DLin-MC3-DMA",
    "dlin-mc3-dma": "DLin-MC3-DMA", "dlinmc3": "DLin-MC3-DMA",
    "kc2": "DLin-KC2-DMA", "kc-2": "DLin-KC2-DMA", "dlin-kc2-dma": "DLin-KC2-DMA",
    "dlinkc2dma": "DLin-KC2-DMA",
    "c12200": "C12-200", "c12-200": "C12-200",
    "dodap": "DODAP", "dotap": "DOTAP", "dlindma": "DLin-DMA",
    "lipid5": "Lipid 5", "lipid 5": "Lipid 5",
    "306oi10": "306Oi10", "9a1p9": "9A1P9",
}


def resolve_cargo(claimed, art: dict) -> str:
    """화물 카테고리를 정합니다. 모델이 비워 보내면 논문에서 보완합니다.

    7차 시험에서 36편 중 12행이 cargo_type 누락으로 기각됐습니다. 이 분야는
    제목에 화물이 거의 명시되므로 제목 → 초록 → 본문 앞부분 순으로 봅니다.
    cargo_category 는 미상에 "other" 를 돌려주므로 or 연쇄로는 안 됩니다
    ("other" 가 truthy 라 첫 단계에서 멈춥니다).
    """
    c = cargo_category(claimed)
    if c and c != "other":
        return c
    for txt in (art.get("title", ""), str(art.get("abstract", ""))[:800],
                str(art.get("body", ""))[:2000]):
        c2 = cargo_category(txt)
        if c2 and c2 != "other":
            return c2
    return c or "other"


PEG_CANON = {
    "dmg-peg": "DMG-PEG2000", "peg-dmg": "DMG-PEG2000",
    "dmg-peg2k": "DMG-PEG2000", "dmg-peg 2000": "DMG-PEG2000",
    "peg2000-dmg": "DMG-PEG2000", "c14-peg 2000": "C14-PEG2000",
}


# ── 검증 게이트 ─────────────────────────────────────────────────────────
def _fnum(v):
    """숫자로 변환. 결측(None / "" / NaN)은 None 을 돌려줍니다.

    NaN 을 그대로 흘리면 하류의 범위 검사가 항상 실패해 결측 행이 전부
    기각됩니다(7차에서 np_ratio 결측 60행이 이 이유로 기각됐습니다).
    """
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        x = float(str(v).replace(",", "."))
        return None if x != x else x          # NaN 차단
    except (TypeError, ValueError):
        return None


def ratio_wellformed(s) -> tuple[bool, str]:
    """몰비 문자열이 형식적으로 성립하는가."""
    parts = [p for p in str(s or "").split(":") if p.strip()]
    if len(parts) < 3:
        return False, "성분 3개 미만"
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return False, "숫자 아님"
    if any(v <= 0 for v in vals):
        return False, "0 이하 값"
    if max(vals) / min(vals) > 500:
        return False, "비율이 극단적"
    return True, ""


def ratio_in_source(ratio: str, text: str) -> bool:
    """몰비의 값 집합이 원문의 어떤 몰비 표현과 일치하는가.

    순서 변경은 허용합니다(스키마 순서로 재배열하는 것이 정상 경로).
    값 자체가 바뀌면 기각합니다 — 이것이 환각을 잡는 핵심입니다.
    """
    want = sorted((p.strip() for p in str(ratio or "").split(":") if p.strip()),
                  key=lambda x: float(x))
    if len(want) < 3:
        return False
    for m in RATIO_TOKEN.finditer(text):
        got = [g.replace(",", ".") for g in m.groups() if g]
        gs = sorted(got, key=float)
        if len(got) == len(want) and gs == want:
            return True
        # 5성분 처방(두 번째 PEG 지질 등)의 앞 4성분만 저장한 경우를 허용합니다.
        # 7차에서 이 경우 6행이 부당하게 환각으로 기각됐습니다. 값 자체는
        # 원문에 있어야 하므로 부분집합만 허용하고, 값 변경은 여전히 기각합니다.
        if len(got) > len(want):
            pool = list(gs)
            ok = True
            for w in want:
                if w in pool:
                    pool.remove(w)
                else:
                    ok = False
                    break
            if ok:
                return True
    return False


def number_in_source(val, text: str) -> bool:
    v = _fnum(val)
    if v is None:
        return False
    for s in {f"{v:g}", f"{v:.1f}", f"{v:.0f}"}:
        if re.search(rf"(?<![\d.]){re.escape(s)}(?![\d])", text):
            return True
    return False


def name_in_source(name: str, text: str) -> bool:
    """지질명이 원문에 있는가 — 표기 변형을 허용합니다.

    논문은 SM-102 를 SM102, DLin-MC3-DMA 를 MC3 로 씁니다. 저장 시에는
    기존 데이터 표기로 통일하므로, 원문 대조에서는 하이픈·공백을 지운
    형태와 흔한 약칭까지 봐야 합니다. 7차에서 이 처리가 없어 정상 행
    11개가 부당하게 기각됐습니다.
    """
    n = str(name or "").strip().lower()
    if not n:
        return False
    hay = text.lower()
    hay_flat = hay.replace("-", "").replace(" ", "").replace("\u2010", "")
    cands = {n, n.replace("-", "").replace(" ", "")}
    # 접두 수식어를 뗀 약칭 (DLin-MC3-DMA → MC3)
    parts = [p for p in re.split(r"[-\s]", n) if len(p) >= 3]
    cands.update(parts)
    for c in cands:
        if len(c) < 3:
            continue
        if c in hay or c.replace("-", "").replace(" ", "") in hay_flat:
            return True
    return False


def reduce_to_four(ratio: str) -> tuple[str, str]:
    """5성분 이상 몰비를 스키마의 4성분으로 축약합니다.

    이것이 필요한 이유는 하류입니다. build_features 는 4성분만 파싱하므로
    "42:10:8:38:2" 같은 값이 들어오면 조성 특징 7개(ionizable, helper, chol,
    peg, ion_to_helper, ion_plus_chol, log_peg)가 전부 NaN 이 되고, 대치값으로
    채워져 조용히 틀린 입력이 됩니다. 7차 시험에서 2행이 이 경로였습니다.

    5성분은 대개 두 번째 PEG 지질(DSPE-PEG 등)이 붙은 경우이고 값이 작습니다.
    앞 4성분을 남기고 나머지를 주석으로 기록합니다.

    반환: (축약된 몰비, 기록 메모). 축약이 불가능하면 ("", 사유).
    """
    parts = [p.strip() for p in str(ratio or "").split(":") if p.strip()]
    if len(parts) <= 4:
        return ":".join(parts), ""
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return "", "숫자 아님"
    extra = vals[4:]
    # 남기는 4성분이 전체의 대부분이어야 축약이 정당합니다
    if sum(extra) > 0.15 * sum(vals):
        return "", f"{len(parts)}성분 처방(축약 불가)"
    return ":".join(f"{v:g}" for v in vals[:4]), \
           f"원문 {len(parts)}성분({ratio}) 중 앞 4성분만 저장; 제외 {extra}"


def gate_row(row: dict, source_text: str) -> tuple[bool, str]:
    """원문 대조 게이트. 통과하지 못한 행은 데이터베이스에 넣지 않습니다.

    반환: (통과여부, 기각사유)
    """
    ee = _fnum(row.get("encapsulation_efficiency_percent_std_num"))
    if ee is None:
        return False, "EE 없음"
    lo, hi = RANGES["encapsulation_efficiency_percent_std_num"]
    if not (lo <= ee <= hi):
        return False, "EE 범위 밖"
    if not number_in_source(ee, source_text):
        return False, "EE가 원문에 없음"

    # 축약 불가로 몰비가 비워진 경우 사유를 그대로 전달합니다
    note = str(row.get("repair_note") or "")
    if not str(row.get("lipid_molar_ratio") or "").strip() and "축약 불가" in note:
        return False, note.strip("() ")

    ok, why = ratio_wellformed(row.get("lipid_molar_ratio"))
    if not ok:
        return False, f"몰비 형식 불량({why})"
    if not ratio_in_source(row["lipid_molar_ratio"], source_text):
        return False, "몰비가 원문에 없음(환각 의심)"

    name = str(row.get("ionizable_lipid_name") or "").strip()
    if name and name != UNNAMED:
        if name.lower() in NOT_IONIZABLE:
            return False, "이온화지질이 아닌 이름"
        if not name_in_source(name, source_text):
            return False, "지질명이 원문에 없음"

    for col, (lo, hi) in RANGES.items():
        v = _fnum(row.get(col))
        if v is not None and not (lo <= v <= hi):
            return False, f"{col} 범위 밖"
    return True, ""


# ── 추출 프롬프트 ───────────────────────────────────────────────────────
EXTRACT_SYSTEM = """You extract LNP formulation rows from a paper excerpt.

Report every formulation for which the excerpt states BOTH a lipid molar ratio
AND an encapsulation efficiency. Skip formulations missing either.

Field rules
  ionizable_lipid_name  the paper's own designation, including codes for novel
        compounds ("3D-P-DMA", "KEL1", "CP-LC-1272"). Use "Custom lipid" ONLY
        when the paper gives no name at all. Never put a helper lipid (DSPC,
        DOPE), a PEG-lipid, cholesterol, a cargo, or a drug here.
  lipid_molar_ratio     "ionizable:helper:cholesterol:PEG". Papers often list a
        different order (commonly ionizable:cholesterol:helper:PEG) — reorder
        the paper's numbers to the schema order. Never change a value. If the
        paper gives 3 components because there is no helper lipid, report 3.
  encapsulation_efficiency_percent_std_num   the number, 0-100.
  ee_is_approximate     true if the paper only gives a bound ("above 80%",
        ">90%", "~95%"), else false.
  evidence              verbatim quote under 25 words containing the EE.

Hard rules
- Every number you report must appear in the excerpt. A downstream checker
  compares your ratios and EE against the source text and discards rows that
  do not match, so inventing a plausible composition (50:10:38.5:1.5) wastes
  the row.
- If a paper reports one ratio and several EEs, emit one row per EE with the
  same ratio. Do not vary the ratio to match.
- Return the tool input as a real JSON object, not as a string, and never write
  placeholder tokens such as <UNKNOWN>, TBD, or ... in any field. If you do not
  have a value, omit that optional field; if you lack the ratio or the EE, do
  not emit the row at all.
"""

EXTRACT_TOOL = {
    "name": "emit_rows",
    "input_schema": {
        "type": "object",
        "properties": {
            "rows": {"type": "array", "items": {"type": "object", "properties": {
                "ionizable_lipid_name": {"type": "string"},
                "lipid_molar_ratio": {"type": "string"},
                "encapsulation_efficiency_percent_std_num": {"type": "number"},
                "ee_is_approximate": {"type": "boolean"},
                "helper_lipid_name": {"type": "string"},
                "peg_lipid_name": {"type": "string"},
                "cargo_type": {"type": "string"},
                "np_ratio_std_num": {"type": ["number", "null"]},
                "buffer_ph_std_num": {"type": ["number", "null"]},
                "particle_size_nm_std_num": {"type": ["number", "null"]},
                "pdi_std_num": {"type": ["number", "null"]},
                "zeta_potential_mv_std_num": {"type": ["number", "null"]},
                "evidence": {"type": "string"},
            }, "required": ["lipid_molar_ratio",
                            "encapsulation_efficiency_percent_std_num",
                            "evidence"]}},
        },
        "required": ["rows"],
    },
}


def build_excerpt(body: str, tables: list, maxlen: int = 9000) -> str:
    """몰비와 EE 가 등장하는 구간 + 관련 표만 모아 발췌를 만듭니다.

    전체 본문(평균 5만자)을 그대로 넣지 않는 이유는 비용과 정확도 둘 다입니다.
    """
    spans = []
    for m in RATIO_TOKEN.finditer(body):
        if sum(1 for g in m.groups() if g) >= 3:
            spans.append((max(0, m.start() - 600), min(len(body), m.end() + 500)))
    for m in EE_PAT.finditer(body):
        spans.append((max(0, m.start() - 500), min(len(body), m.end() + 400)))
    if not spans:
        return ""
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + 80:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    txt = "\n[...]\n".join(body[s:e] for s, e in merged)

    tt = ""
    for tb in (tables or [])[:8]:
        cap = str(tb.get("caption", ""))[:160]
        rows = tb.get("rows") or []
        hdr = " ".join(str(x) for x in (rows[0] if rows else []))
        if any(k in (cap + hdr).lower() for k in
               ("encapsul", "molar", "formulation", "lipid", "composition")):
            tt += f"\n[TABLE] {cap}\n" + "\n".join(
                " | ".join(map(str, rr))[:280] for rr in rows[:16])
    return (txt + tt)[:maxlen]


# ── 중복 판정 ───────────────────────────────────────────────────────────
def row_key(row) -> str:
    """중복 판정 키. 몰비는 값 집합으로 정규화합니다.

    같은 처방을 50:10:38.5:1.5 와 50:38.5:10:1.5 로 저장하면 문자열이 달라
    중복을 놓칩니다(7차 시험에서 같은 논문·같은 EE 가 2행이 됐습니다).
    순서에 무관한 키를 씁니다.
    """
    def g(k):
        v = row.get(k) if isinstance(row, dict) else row[k]
        return str(v or "").strip().lower()
    ee = _fnum(row.get("encapsulation_efficiency_percent_std_num")
               if isinstance(row, dict)
               else row["encapsulation_efficiency_percent_std_num"])
    parts = [p.strip() for p in g("lipid_molar_ratio").split(":") if p.strip()]
    try:
        ratio_key = ":".join(f"{float(p):g}" for p in sorted(parts, key=float))
    except ValueError:
        ratio_key = g("lipid_molar_ratio")
    return "|".join([g("reference_doi"), g("ionizable_lipid_name"),
                     ratio_key, f"{ee:.1f}" if ee is not None else ""])


def existing_keys(df: pd.DataFrame) -> set:
    if df is None or len(df) == 0:
        return set()
    out = set()
    for _, r in df.iterrows():
        out.add(row_key(r.to_dict()))
    return out


def existing_pmcids(df: pd.DataFrame) -> set:
    if df is None or "pmcid" not in df.columns:
        return set()
    return {str(p).replace(".0", "").replace("PMC", "").strip()
            for p in df["pmcid"].dropna() if str(p).strip()}


# ── 진행 이벤트 ─────────────────────────────────────────────────────────
@dataclass
class Progress:
    stage: str
    frac: float
    message: str
    counts: dict = field(default_factory=dict)


@dataclass
class HarvestJob:
    """PMC 검색 → 본문 확보 → 추출 → 원문 대조 → 중복 제거 → 스키마 변환.

    existing_df : 지금 쓰고 있는 데이터. 중복 판정과 표기 통일의 기준입니다.
    llm         : host.llm 을 그대로 넘깁니다. 리스트 입력을 지원해야 합니다.
    target_rows : 이만큼 모이면 조기 종료합니다.
    require_named : True 면 이름이 확실한 행만 채택합니다(Custom lipid 배제).
    require_exact_ee : True 면 EE 가 근사표현인 행을 배제합니다.
    """
    existing_df: Optional[pd.DataFrame] = None
    llm: Optional[Callable] = None
    model: Optional[str] = None
    target_rows: int = 40
    require_named: bool = True
    require_exact_ee: bool = False
    nucleic_only: bool = True   # EE 정의가 다른 화물(단백질·소분자)을 배제
    max_concurrency: int = 6

    accepted: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=SCHEMA))
    rejected: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)

    def run(self, queries: Optional[Iterable[str]] = None,
            max_papers: int = 120, batch: int = 12) -> Iterator[Progress]:
        """제너레이터. 각 단계마다 Progress 를 내보냅니다."""
        import lnp_harvest as H

        have_pmc = existing_pmcids(self.existing_df)
        have_key = existing_keys(self.existing_df)

        yield Progress("search", 0.02, "PMC 검색 중…")
        ids = []
        for q in (queries or [None]):
            try:
                got = H.search_pmc(retmax=300) if q is None else H.search_pmc(term=q, retmax=300)
            except TypeError:
                got = H.search_pmc(retmax=300)
            ids += [str(i) for i in got]
        pool = [i for i in dict.fromkeys(ids)
                if i.replace("PMC", "").strip() not in have_pmc][:max_papers]
        yield Progress("search", 0.08,
                       f"신규 후보 {len(pool)}편 (기존 {len(have_pmc)}편 제외)",
                       {"candidates": len(pool)})
        if not pool:
            self.stats = {"candidates": 0}
            return

        rows, rej, seen_key = [], [], set(have_key)
        done = 0
        for k in range(0, len(pool), batch):
            chunk = pool[k:k + batch]
            try:
                xmls = H.fetch_xml(chunk, batch=batch)
            except Exception as e:
                rej.append({"pmcid": ",".join(chunk), "why": f"XML 수신 실패: {e}"})
                continue

            reqs, meta = [], []
            for pid, xs in xmls.items():
                try:
                    art = H.parse_article(xs)
                except Exception:
                    continue
                ex = build_excerpt(art.get("body", ""), art.get("tables"))
                if len(ex) < 300:
                    rej.append({"pmcid": pid, "why": "몰비·EE 구간 없음"})
                    continue
                reqs.append({"prompt": f"Paper: {art.get('title','')[:170]}\n\nExcerpt:\n{ex}",
                             "system": EXTRACT_SYSTEM, "tools": [EXTRACT_TOOL],
                             "tool_choice": {"type": "tool", "name": "emit_rows"},
                             "max_tokens": 2200,
                             **({"model": self.model} if self.model else {})})
                meta.append((pid, art, ex))

            if reqs:
                outs = self.llm(reqs, max_concurrency=self.max_concurrency)
                for (pid, art, ex), o in zip(meta, outs):
                    if not isinstance(o, dict) or "error" in o:
                        rej.append({"pmcid": pid, "why": "LLM 오류"})
                        continue
                    tu = o.get("tool_use") or {}
                    got = (tu.get("input") or {}).get("rows") or []
                    # rows 가 리스트가 아니면(문자열로 오는 경우가 있습니다)
                    # 글자 단위로 순회되어 기각 수천 건이 찍힙니다. 논문 단위로
                    # 한 번만 기각합니다.
                    if isinstance(got, str):
                        # rows 가 JSON 문자열로 오는 경우가 있습니다. 파싱해
                        # 살립니다 — 그냥 기각하면 논문 7편 중 1편꼴로 버립니다.
                        try:
                            got = json.loads(got)
                        except (json.JSONDecodeError, TypeError):
                            rej.append({"pmcid": pid, "why": "추출 형식 불량(rows)"})
                            continue
                    if not isinstance(got, list):
                        rej.append({"pmcid": pid, "why": "추출 형식 불량(rows)"})
                        continue
                    src_text = ex + " " + art.get("body", "")[:20000]
                    for g in got:
                        # 모델이 rows 에 문자열이나 리스트를 담아 보내는 경우가
                        # 있습니다. 스키마를 tool 로 강제해도 완전히 막히지는
                        # 않으므로 형식을 확인하고 넘깁니다.
                        if not isinstance(g, dict):
                            rej.append({"pmcid": pid, "why": "추출 형식 불량"})
                            continue
                        rec = self._to_schema(g, pid, art)
                        ok, why = gate_row(rec, src_text)
                        if not ok:
                            rej.append({"pmcid": pid, "why": why,
                                        "ratio": rec.get("lipid_molar_ratio"),
                                        "ee": rec.get("encapsulation_efficiency_percent_std_num")})
                            continue
                        kk = row_key(rec)
                        if kk in seen_key:
                            rej.append({"pmcid": pid, "why": "중복"})
                            continue
                        if self.require_named and rec["ionizable_lipid_name"] in ("", UNNAMED):
                            rej.append({"pmcid": pid, "why": "지질명 불분명"})
                            continue
                        if self.require_exact_ee and rec["ee_is_approximate"]:
                            rej.append({"pmcid": pid, "why": "EE 근사표현"})
                            continue
                        if self.nucleic_only and rec["cargo_type"] not in NUCLEIC_CARGO:
                            rej.append({"pmcid": pid,
                                        "why": f"핵산 아닌 화물({rec['cargo_type'] or '미상'})"})
                            continue
                        seen_key.add(kk)
                        rows.append(rec)

            done += len(chunk)
            frac = 0.08 + 0.9 * done / max(1, len(pool))
            yield Progress("extract", min(frac, 0.98),
                           f"{done}/{len(pool)}편 처리 · 채택 {len(rows)}행 · 기각 {len(rej)}행",
                           {"papers": done, "accepted": len(rows), "rejected": len(rej)})
            if len(rows) >= self.target_rows:
                yield Progress("extract", 0.98, f"목표 {self.target_rows}행 도달 — 조기 종료")
                break

        self.accepted = self._finalize(pd.DataFrame(rows))
        self.rejected = pd.DataFrame(rej)
        self.stats = {"candidates": len(pool), "papers_seen": done,
                      "accepted": len(self.accepted), "rejected": len(self.rejected)}
        yield Progress("done", 1.0,
                       f"완료 — 채택 {len(self.accepted)}행 / 기각 {len(self.rejected)}행",
                       self.stats)

    # ── 내부 ───────────────────────────────────────────────────────────
    def _to_schema(self, g: dict, pid: str, art: dict) -> dict:
        ev = str(g.get("evidence", ""))[:400]
        approx = bool(g.get("ee_is_approximate")) or bool(APPROX_PAT.search(ev))
        name = str(g.get("ionizable_lipid_name") or "").strip()
        # 총칭은 이름이 아닙니다 — UNNAMED 로 접어 require_named 에서 걸립니다
        if not name or name.lower() in GENERIC_NAMES:
            name = UNNAMED
        raw_ratio = str(g.get("lipid_molar_ratio", "")).replace(" ", "")
        ratio4, ratio_note = reduce_to_four(raw_ratio)
        rec = {c: "" for c in SCHEMA}
        rec.update({
            "reference_doi": art.get("doi", "") or f"PMC{pid}",
            "pmcid": str(pid),
            "ionizable_lipid_name": name,
            "lipid_molar_ratio": ratio4,
            "helper_lipid_name": str(g.get("helper_lipid_name") or "").strip(),
            "peg_lipid_name": str(g.get("peg_lipid_name") or "").strip(),
            "cargo_type": resolve_cargo(g.get("cargo_type"), art),
            "encapsulation_efficiency_percent_std_num": _fnum(
                g.get("encapsulation_efficiency_percent_std_num")),
            "ee_is_approximate": approx,
            "evidence": ev,
            "source": "PMC 자동 수집 (원문 대조 검증)",
            "source_note": f"PMC{pid}",
            "confidence": "low" if name in ("", UNNAMED) else ("medium" if approx else "high"),
            "repair_note": ratio_note,
        })
        for k, col in [("np_ratio_std_num", "np_ratio_std_num"),
                       ("buffer_ph_std_num", "buffer_ph_std_num"),
                       ("particle_size_nm_std_num", "particle_size_nm_std_num"),
                       ("pdi_std_num", "pdi_std_num"),
                       ("zeta_potential_mv_std_num", "zeta_potential_mv_std_num")]:
            v = _fnum(g.get(k))
            lo, hi = RANGES[col]
            rec[col] = v if (v is not None and lo <= v <= hi) else None
        return rec

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=SCHEMA)
        # 먼저 내장 사전으로 통일합니다. 기존 데이터가 같은 물질을 두 표기로
        # 갖고 있는 경우(SM-102 / SM102)가 있어 사전이 기준이 됩니다.
        def _flat(s):
            return str(s).strip().lower().replace("-", "").replace(" ", "")
        df["ionizable_lipid_name"] = df["ionizable_lipid_name"].map(
            lambda s: ION_CANON.get(str(s).strip().lower(),
                                    ION_CANON.get(_flat(s), str(s).strip())))
        # 그 다음 기존 데이터의 표기를 반영합니다(사전에 없는 이름만)
        if self.existing_df is not None and "ionizable_lipid_name" in self.existing_df:
            canon = {str(n).strip().lower(): str(n).strip()
                     for n in self.existing_df["ionizable_lipid_name"].dropna()
                     if str(n).strip()}
            fixed_by_dict = set(ION_CANON.values())
            df["ionizable_lipid_name"] = df["ionizable_lipid_name"].map(
                lambda s: str(s).strip() if str(s).strip() in fixed_by_dict
                else canon.get(str(s).strip().lower(), str(s).strip()))
            smi = {str(r["ionizable_lipid_name"]).strip().lower(): r["ionizable_lipid_smiles"]
                   for _, r in self.existing_df.iterrows()
                   if str(r.get("ionizable_lipid_smiles") or "").strip()}
            df["ionizable_lipid_smiles"] = df["ionizable_lipid_name"].map(
                lambda s: smi.get(str(s).strip().lower(), ""))
        df["peg_lipid_name"] = df["peg_lipid_name"].map(
            lambda s: PEG_CANON.get(str(s).strip().lower(), str(s).strip()))
        # 이름 미상에 SMILES 가 붙는 것을 막습니다
        df.loc[df["ionizable_lipid_name"].isin(["", UNNAMED]), "ionizable_lipid_smiles"] = ""
        for c in SCHEMA:
            if c not in df.columns:
                df[c] = ""
        return df[SCHEMA].reset_index(drop=True)


def audit_rows(df: pd.DataFrame, articles: dict, llm: Callable,
               model: Optional[str] = None, max_concurrency: int = 6) -> pd.DataFrame:
    """2차 감사 — 이미 저장된 행을 원문과 다시 대조합니다.

    7차에서 이 단계가 잡은 것: cargo 오분류 25건, 제타/pKa 혼동 11건,
    LNP 아닌 처방 1행, 5성분 처방 누락 2행.
    """
    from lnp_autoharvest import AUDIT_SYSTEM, AUDIT_TOOL  # 자기 참조 방지용 지연 임포트
    reqs, idxs = [], []
    for i, row in df.iterrows():
        pid = str(row.get("pmcid", "")).replace(".0", "").strip()
        art = articles.get(pid)
        if art is None:
            continue
        ex = build_excerpt(art.get("body", ""), art.get("tables"), maxlen=6500)
        if len(ex) < 300:
            continue
        claim = "\n".join(f"  {c}: {row[c]}" for c in SCHEMA if str(row.get(c, "")).strip())
        reqs.append({"prompt": f"Claimed row:\n{claim}\n\nExcerpt:\n{ex}",
                     "system": AUDIT_SYSTEM, "tools": [AUDIT_TOOL],
                     "tool_choice": {"type": "tool", "name": "audit_row"},
                     "max_tokens": 1600, **({"model": model} if model else {})})
        idxs.append(i)
    if not reqs:
        return pd.DataFrame()
    outs = llm(reqs, max_concurrency=max_concurrency)
    recs = []
    for i, o in zip(idxs, outs):
        if not isinstance(o, dict) or "error" in o:
            continue
        a = (o.get("tool_use") or {}).get("input")
        if isinstance(a, dict):
            a["row"] = i
            recs.append(a)
    return pd.DataFrame(recs)


AUDIT_SYSTEM = """You are auditing one row of an extracted LNP formulation database
against the source paper excerpt. CHECK, do not extract freely.

For each field decide whether the excerpt supports the claim, contradicts it, or
is silent, then give a corrected value where the excerpt supports a different one.

  ratio_verdict  "ok" if the claimed ratio matches the paper for THIS lipid
                 (reordering the paper's numbers into the schema order
                 ionizable:helper:cholesterol:PEG still counts as ok),
                 "wrong_order", "wrong_values", or "not_found".
  ee_verdict     "ok" | "wrong" | "not_found";  ee_is_range true for bounds.
  ion_verdict    "ok" | "wrong" | "unnamed" | "not_ionizable" (a drug, helper
                 lipid, PEG-lipid, cholesterol, or cargo in the lipid field).
  overall        "keep" | "fix" | "drop". Use "drop" when the row is not an LNP
                 with an ionizable lipid, or the excerpt contradicts it and you
                 cannot determine correct values.
  quote          verbatim snippet under 25 words justifying the verdict.

The excerpt is the only evidence. If silent, say "not_found" — never fall back
on what is typical for LNPs, and never invent a ratio."""

AUDIT_TOOL = {
    "name": "audit_row",
    "input_schema": {"type": "object", "properties": {
        "ratio_verdict": {"type": "string",
                          "enum": ["ok", "wrong_order", "wrong_values", "not_found"]},
        "ratio_correct": {"type": ["string", "null"]},
        "ee_verdict": {"type": "string", "enum": ["ok", "wrong", "not_found"]},
        "ee_correct": {"type": ["number", "null"]},
        "ee_is_range": {"type": "boolean"},
        "ion_verdict": {"type": "string",
                        "enum": ["ok", "wrong", "unnamed", "not_ionizable"]},
        "ion_correct": {"type": ["string", "null"]},
        "helper_correct": {"type": ["string", "null"]},
        "peg_correct": {"type": ["string", "null"]},
        "cargo_correct": {"type": ["string", "null"]},
        "size": {"type": ["number", "null"]}, "pdi": {"type": ["number", "null"]},
        "zeta": {"type": ["number", "null"]}, "np_ratio": {"type": ["number", "null"]},
        "ph": {"type": ["number", "null"]},
        "overall": {"type": "string", "enum": ["keep", "fix", "drop"]},
        "quote": {"type": "string"}, "note": {"type": "string"}},
        "required": ["ratio_verdict", "ee_verdict", "ion_verdict", "overall",
                     "quote", "note"]},
}
