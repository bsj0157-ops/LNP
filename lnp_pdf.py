# ==========================================================================
#  LNP-PDF  —  논문 PDF에서 처방 데이터 후보를 뽑아내는 모듈
#  ------------------------------------------------------------------------
#  ! 중요: 이 모듈은 "자동 입력기"가 아니라 "초안 작성기"입니다.
#    PDF에서 뽑은 값은 반드시 사람이 확인해야 합니다. LNP 논문의 조성은
#    본문 문장, 표, SI 에 흩어져 있고 표기도 제각각이라 완전 자동은
#    신뢰할 수 없습니다. 이 모듈의 목표는 '타이핑 양을 줄이는 것'이고,
#    각 후보에 근거 문장(evidence)과 페이지 번호를 붙여 검토를 돕습니다.
#
#  뽑는 것:
#    - DOI            : 정규식 (거의 항상 성공)
#    - 지질 몰비       : "50:10:38.5:1.5" 형태 + 주변 문맥
#    - EE (%)         : "encapsulation efficiency of 94%" 형태
#    - 지질 이름       : 알려진 이온화/헬퍼/PEG 지질 사전 매칭
#    - N/P 비, pH     : "N/P ratio of 6", "pH 4.0 citrate"
#    - 입자 크기/PDI   : "80 nm", "PDI 0.12"
# ==========================================================================

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

# --- 알려진 지질 사전 -----------------------------------------------------
IONIZABLE = [
    "DLin-MC3-DMA", "MC3", "DLin-KC2-DMA", "KC2", "DLinDMA", "SM-102", "SM102",
    "ALC-0315", "ALC0315", "C12-200", "cKK-E12", "OF-02", "306Oi10", "5A2-SC8",
    "Lipid 5", "L319", "TT3", "9A1P9", "FTT5", "BAMEA-O16B", "YSK05", "CL4H6",
    "306O13", "A6", "98N12-5", "7C1", "DODAP", "DODMA",
]
HELPER = ["DSPC", "DOPE", "POPC", "DPPC", "DOPC", "DSPE", "SOPC", "egg PC", "ESM"]
PEG_LIPID = [
    "DMG-PEG2000", "DMG-PEG 2000", "DMG-PEG", "ALC-0159", "ALC0159",
    "PEG-DMG", "PEG2000-DMG", "DSPE-PEG2000", "DSPE-PEG", "C16-PEG2000",
    "PEG-DSG", "PEG-c-DMA", "PEG-DMPE",
]
CARGO = {
    "mRNA": ["mrna", "messenger rna", "modrna", "nucleoside-modified rna"],
    "siRNA": ["sirna", "small interfering"],
    "saRNA": ["sarna", "self-amplifying", "replicon"],
    "pDNA": ["pdna", "plasmid dna", "plasmid"],
    "ASO": ["antisense oligo", "aso "],
    "circRNA": ["circrna", "circular rna"],
}

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+[A-Za-z0-9])")
# 3~4성분 몰비: 40:10:48:2  /  40/10/48/2  /  40 : 10 : 48 : 2
RATIO_RE = re.compile(
    r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*[:/]\s*(\d{1,3}(?:\.\d+)?)\s*[:/]\s*"
    r"(\d{1,3}(?:\.\d+)?)(?:\s*[:/]\s*(\d{1,3}(?:\.\d+)?))?(?![\d.])")
EE_RE = re.compile(
    r"(?:encapsulation\s+efficienc\w*|entrapment\s+efficienc\w*|\bEE%?\b|\bE\.E\.)"
    r"[^.\n]{0,80}?(\d{1,3}(?:\.\d+)?)\s*%", re.I)
EE_RE2 = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*%[^.\n]{0,40}?"
    r"(?:encapsulation|entrapment)\s+efficienc\w*", re.I)
NP_RE = re.compile(r"(?:N\s*[:/]\s*P|N/P|nitrogen[- ]to[- ]phosphate)\s*"
                   r"(?:ratio\s*)?(?:of\s*|=\s*|:\s*)?(\d{1,2}(?:\.\d+)?)", re.I)
PH_RE = re.compile(r"pH\s*(?:of\s*|=\s*)?(\d(?:\.\d+)?)", re.I)
SIZE_RE = re.compile(r"(\d{2,3}(?:\.\d+)?)\s*(?:±\s*[\d.]+\s*)?nm", re.I)
PDI_RE = re.compile(r"(?:PDI|polydispersity(?:\s+index)?)\s*(?:of\s*|=\s*|:\s*)?"
                    r"(0?\.\d+)", re.I)
ZETA_RE = re.compile(r"(?:zeta[- ]potential)\s*(?:of\s*|=\s*|:\s*)?"
                     r"(-?\d{1,2}(?:\.\d+)?)\s*mV", re.I)


# ==========================================================================
def read_pdf(file_or_bytes):
    """PDF → [(페이지번호, 텍스트), ...]. pdfplumber 우선, pypdf 대체."""
    pages = []
    try:
        import pdfplumber
        with pdfplumber.open(file_or_bytes) as pdf:
            for i, pg in enumerate(pdf.pages, 1):
                pages.append((i, pg.extract_text() or ""))
        return pages
    except Exception:
        pass
    try:
        from pypdf import PdfReader
        rd = PdfReader(file_or_bytes)
        for i, pg in enumerate(rd.pages, 1):
            pages.append((i, pg.extract_text() or ""))
    except Exception as e:
        raise RuntimeError(f"PDF를 읽지 못했습니다: {e}")
    return pages


def read_pdf_tables(file_or_bytes, max_pages=40):
    """표를 DataFrame 목록으로. 조성표가 표로 되어 있을 때 유용."""
    out = []
    try:
        import pdfplumber
        with pdfplumber.open(file_or_bytes) as pdf:
            for i, pg in enumerate(pdf.pages[:max_pages], 1):
                for t in (pg.extract_tables() or []):
                    if t and len(t) > 1:
                        df = pd.DataFrame(t[1:], columns=[str(c) for c in t[0]])
                        out.append((i, df))
    except Exception:
        pass
    return out


def _clean(txt):
    """줄바꿈으로 끊긴 단어를 잇고 공백을 정리."""
    txt = re.sub(r"-\n(\w)", r"\1", txt)      # 하이픈 줄바꿈 복원
    txt = re.sub(r"\n", " ", txt)
    return re.sub(r"\s{2,}", " ", txt)


def _context(text, m, width=110):
    a = max(0, m.start() - width)
    b = min(len(text), m.end() + width)
    return "…" + text[a:b].strip() + "…"


# --- 성분 순서 자동 인식 --------------------------------------------------
# 실제 논문 확인: "ALC-0315, DSPC, cholesterol, and DMG-PEG2000 at a molar
# ratio of 50:38.5:10:1.5" — 즉 논문이 나열한 순서가 우리 표준 순서
# (이온화:헬퍼:콜레스테롤:PEG)와 다를 수 있습니다. 숫자만 믿으면
# DSPC=38.5, chol=10 으로 뒤바뀌어 저장됩니다.
ROLE_WORDS = {
    "ionizable": ["dlin-mc3", "mc3", "kc2", "sm-102", "sm102", "alc-0315",
                  "alc0315", "c12-200", "ckk-e12", "lipid 5", "l319", "306oi10",
                  "ionizable", "cationic", "dodap", "dodma", "5a2-sc8"],
    "helper":    ["dspc", "dope", "popc", "dppc", "dopc", "phospholipid",
                  "helper", "sopc", "esm"],
    "chol":      ["cholesterol", "chol", "sterol", "beta-sitosterol"],
    "peg":       ["peg", "dmg-peg", "alc-0159", "dspe-peg", "peg-dmg", "peg-dsg"],
}
STD_ORDER = ["ionizable", "helper", "chol", "peg"]


def detect_order(context, n_comp=4):
    """근거 문장에서 성분이 나열된 순서를 읽는다.

    반환: (순서 리스트, 확신 여부). 예) ['ionizable','chol','helper','peg']
    못 읽으면 표준 순서와 False 를 돌려줍니다.
    """
    low = context.lower()
    found = []
    for role, words in ROLE_WORDS.items():
        pos = min((low.find(w) for w in words if low.find(w) >= 0), default=-1)
        if pos >= 0:
            found.append((pos, role))
    found.sort()
    order = [r for _, r in found]
    if len(order) == n_comp and len(set(order)) == n_comp:
        return order, True
    return STD_ORDER[:n_comp], False


def reorder_ratio(ratio_str, order):
    """논문 순서로 읽힌 몰비를 표준 순서(이온화:헬퍼:콜:PEG)로 재배열."""
    vals = [float(x) for x in re.split(r"[:/]", ratio_str)]
    if len(vals) != len(order):
        return ratio_str
    mapping = dict(zip(order, vals))
    std = [r for r in STD_ORDER if r in mapping]
    return ":".join(f"{mapping[r]:g}" for r in std)


def chemistry_check(ratio_str):
    """표준 순서로 배열된 몰비가 화학적으로 그럴듯한가.

    실제 논문에서 발견한 사례: 어떤 논문은 "ALC-0315, DSPC, cholesterol,
    DMG-PEG2000 = 50:38.5:10:1.5" 라고 적었는데, 이러면 DSPC=38.5,
    콜레스테롤=10 이 됩니다. 통상 조성(콜레스테롤 30~50%, 헬퍼 5~20%)과
    정반대라 논문 오기이거나 순서를 다르게 쓴 것입니다.
    자동으로 고치지 않고 경고만 합니다 — 원문 확인이 필요합니다.
    """
    try:
        v = [float(x) for x in re.split(r"[:/]", ratio_str)]
    except ValueError:
        return []
    warn = []
    if len(v) == 4:
        ion, helper, chol, peg = v
        if helper > chol:
            warn.append(
                f"헬퍼({helper:g}) > 콜레스테롤({chol:g}) — 통상과 반대입니다. "
                f"순서가 뒤바뀌었거나 논문 오기일 수 있습니다 "
                f"(뒤집으면 {ion:g}:{chol:g}:{helper:g}:{peg:g})")
        if peg > 6:
            warn.append(f"PEG({peg:g}%)가 통상 범위(0.5~5%)보다 큽니다")
        if not (20 <= ion <= 70):
            warn.append(f"이온화지질({ion:g}%)이 통상 범위(25~60%)를 벗어납니다")
    return warn


def _plausible_ratio(vals):
    """몰비로 그럴듯한가 — 합이 100 근처이고 PEG 성분이 작아야 한다.

    논문 PDF에는 '10:1', '2:1' 같은 무관한 비율과 페이지 범위가 흔해서
    이 필터가 없으면 오탐이 대부분을 차지합니다.
    """
    v = [x for x in vals if x is not None]
    if len(v) not in (3, 4):
        return False
    s = sum(v)
    if not (85 <= s <= 115):
        return False
    if len(v) == 4 and not (0 < v[3] <= 10):      # PEG 성분은 보통 0.5~5%
        return False
    if max(v) > 80 or min(v) <= 0:
        return False
    return True


# ==========================================================================
def extract(file_or_bytes, max_ratio=25):
    """PDF에서 입력 후보를 뽑는다.

    반환: dict — doi, ratios(근거 문장 포함), ee, lipids, np, ph, size, pdi, zeta
    각 항목에 페이지 번호와 근거 문장을 붙여 사람이 검증할 수 있게 합니다.
    """
    pages = read_pdf(file_or_bytes)
    full = _clean(" ".join(t for _, t in pages))
    out = {"n_pages": len(pages), "n_chars": len(full)}

    # --- DOI: 앞 3페이지에 있는 것이 이 논문의 DOI일 확률이 높다 ---------
    head = _clean(" ".join(t for p, t in pages if p <= 3))
    cands = DOI_RE.findall(head) or DOI_RE.findall(full)
    cands = [c.rstrip(".,;)") for c in cands]
    out["doi"] = cands[0] if cands else None
    out["doi_alternatives"] = list(dict.fromkeys(cands))[:6]

    # --- 몰비 ------------------------------------------------------------
    ratios = []
    for p, txt in pages:
        t = _clean(txt)
        for m in RATIO_RE.finditer(t):
            vals = [float(x) if x else None for x in m.groups()]
            if not _plausible_ratio(vals):
                continue
            v = [x for x in vals if x is not None]
            raw = ":".join(f"{x:g}" for x in v)
            ev = _context(t, m)
            order, sure = detect_order(ev, len(v))
            std = reorder_ratio(raw, order) if sure else raw
            ratios.append({
                "ratio": std, "ratio_as_written": raw,
                "order": order, "order_detected": sure,
                "reordered": sure and std != raw,
                "page": p, "sum": round(sum(v), 1), "n_comp": len(v),
                "chem_warnings": chemistry_check(std),
                "evidence": ev,
            })
    seen, uniq = set(), []
    for r in ratios:
        if r["ratio"] in seen:
            continue
        seen.add(r["ratio"])
        uniq.append(r)
    out["ratios"] = uniq[:max_ratio]

    # --- EE --------------------------------------------------------------
    # 일반론 문장 걸러내기 — 실제 논문에서 확인된 오탐:
    #   "In well-optimized formulations, EE is typically close to 100%"
    # 이런 문장은 이 논문의 측정값이 아니라 배경 서술입니다.
    GENERIC = re.compile(
        r"\b(typically|generally|usually|often|commonly|in general|approximately\s+"
        r"100|close to 100|exceed\w*|greater than|more than|less than|below|above|"
        r"reported to be|is known|literature|previous studies|well-optimized)\b", re.I)
    ee = []
    for p, txt in pages:
        t = _clean(txt)
        for rx in (EE_RE, EE_RE2):
            for m in rx.finditer(t):
                v = float(m.group(1))
                if not (0 < v <= 100):
                    continue
                ev = _context(t, m)
                ee.append({"ee": v, "page": p, "evidence": ev,
                           "generic": bool(GENERIC.search(ev))})
    # 구체적 측정값을 먼저, 일반론은 뒤로 (버리지는 않고 표시만)
    ee.sort(key=lambda d: (d["generic"], -d["ee"]))
    seen, ue = set(), []
    for r in ee:
        if r["ee"] in seen:
            continue
        seen.add(r["ee"])
        ue.append(r)
    out["ee"] = ue[:max_ratio]
    out["ee_specific"] = [d for d in ue if not d["generic"]]

    # --- 지질 이름 (등장 횟수 순) ----------------------------------------
    low = full.lower()
    def _find(names):
        hits = [(nm, low.count(nm.lower())) for nm in names
                if nm.lower() in low and len(nm) > 2]
        return [nm for nm, c in sorted(hits, key=lambda x: -x[1])]
    out["ionizable"] = _find(IONIZABLE)[:5]
    out["helper"] = _find(HELPER)[:4]
    out["peg"] = _find(PEG_LIPID)[:4]

    cargo_hits = Counter()
    for label, keys in CARGO.items():
        for k in keys:
            cargo_hits[label] += low.count(k)
    out["cargo"] = [c for c, n in cargo_hits.most_common(3) if n > 0]

    # --- 공정/물성 --------------------------------------------------------
    def _nums(rx, lo, hi, cap=8):
        vals = []
        for p, txt in pages:
            t = _clean(txt)
            for m in rx.finditer(t):
                try:
                    v = float(m.group(1))
                except ValueError:
                    continue
                if lo <= v <= hi:
                    vals.append({"value": v, "page": p, "evidence": _context(t, m, 70)})
        seen2, o = set(), []
        for d in vals:
            if d["value"] in seen2:
                continue
            seen2.add(d["value"])
            o.append(d)
        return o[:cap]

    out["np_ratio"] = _nums(NP_RE, 0.5, 60)
    out["ph"] = _nums(PH_RE, 2.5, 9.0)
    out["size"] = _nums(SIZE_RE, 30, 400)
    out["pdi"] = _nums(PDI_RE, 0.0, 0.9)
    out["zeta"] = _nums(ZETA_RE, -60, 60)
    return out


def to_draft_rows(ex, max_rows=12):
    """추출 결과를 입력 서식 행 초안으로. 사람이 고칠 것을 전제로 합니다.

    몰비와 EE의 개수가 같으면 순서대로 짝지어 보지만, 이 짝짓기는
    추측입니다 — 반드시 화면에서 확인하고 고치세요.
    """
    ratios = ex.get("ratios", [])
    ees = ex.get("ee", [])
    n = max(len(ratios), 1)
    rows = []
    for i in range(min(n, max_rows)):
        r = ratios[i] if i < len(ratios) else {}
        e = ees[i] if i < len(ees) and len(ees) == len(ratios) else {}
        rows.append({
            "reference_doi": ex.get("doi") or "",
            "lipid_molar_ratio": r.get("ratio", ""),
            "ionizable_lipid_name": (ex.get("ionizable") or [""])[0],
            "encapsulation_efficiency_percent_std_num": e.get("ee", ""),
            "np_ratio_std_num": (ex["np_ratio"][0]["value"] if ex.get("np_ratio") else ""),
            "buffer_ph_std_num": (ex["ph"][0]["value"] if ex.get("ph") else ""),
            "cargo_type": (ex.get("cargo") or [""])[0],
            "helper_lipid_name": (ex.get("helper") or [""])[0],
            "peg_lipid_name": (ex.get("peg") or [""])[0],
            "particle_size_nm_std_num": "",
            "pdi_std_num": "",
            "zeta_potential_mv_std_num": "",
            "ionizable_lipid_smiles": "",
            "source_note": f"PDF auto-draft p.{r.get('page','?')} — 검토 필요",
        })
    return pd.DataFrame(rows)
