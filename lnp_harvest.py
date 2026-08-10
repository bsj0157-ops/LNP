# ==========================================================================
#  LNP-Harvest  —  PMC 오픈액세스에서 논문을 자동으로 찾아 데이터 후보를 뽑는다
#  ------------------------------------------------------------------------
#  PDF보다 나은 경로: PMC 전문 XML 을 씁니다.
#    - 표(table-wrap)가 구조화되어 있어 조성표를 행 단위로 읽을 수 있음
#    - 섹션 구분(Methods / Results)이 있어 문맥 판단이 가능
#    - 그림 캡션도 텍스트로 옴
#
#  ! 한계 (측정된 사실):
#    - PMC 오픈액세스 논문만 됩니다. 구독 저널(JACS, Nano Lett 다수)은 불가.
#    - EE 값이 그림에만 있는 논문은 여전히 못 뽑습니다.
#    - 따라서 이건 "전자동 입력기"가 아니라 "후보 대량 수집기"입니다.
#      뽑힌 행은 confidence 등급이 붙고, 사람이 확인해야 합니다.
# ==========================================================================

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# 검색어 — LNP 처방 데이터를 보고할 법한 논문을 겨냥
DEFAULT_QUERY = (
    '("lipid nanoparticle"[Title/Abstract] OR "lipid nanoparticles"[Title/Abstract]) '
    'AND ("encapsulation efficiency"[Text Word]) '
    'AND ("molar ratio"[Text Word] OR "molar ratios"[Text Word]) '
    'AND (mRNA[Text Word] OR siRNA[Text Word] OR "nucleic acid"[Text Word]) '
    'AND open access[filter]'
)


def _get(url, timeout=60, retries=3, pause=0.4):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last = e
            time.sleep(pause * 2)
    raise last


def search_pmc(query=DEFAULT_QUERY, retmax=60, mindate=None, maxdate=None):
    """PMC 검색 → PMCID 목록."""
    u = (f"{EUTILS}/esearch.fcgi?db=pmc&retmode=json&retmax={retmax}"
         f"&term={urllib.parse.quote(query)}")
    if mindate:
        u += f"&mindate={mindate}&maxdate={maxdate or 3000}&datetype=pdat"
    d = json.loads(_get(u))
    return d["esearchresult"]["idlist"]


def fetch_xml(pmcids, batch=10, pause=0.4):
    """PMCID 목록 → {pmcid: XML문자열}. 배치로 받아 서버 부담을 줄입니다."""
    out = {}
    for i in range(0, len(pmcids), batch):
        chunk = pmcids[i:i + batch]
        u = (f"{EUTILS}/efetch.fcgi?db=pmc&retmode=xml&id="
             + ",".join(chunk))
        try:
            raw = _get(u).decode("utf-8", "replace")
        except Exception:
            continue
        # articleset 안에서 article 별로 쪼갠다
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            continue
        for art in root.findall(".//article"):
            # PMC 응답의 id 타입은 pmcid / pmcaid / pmc 로 판마다 다릅니다.
            # 하나로 못 박으면 조용히 0편이 나옵니다 (실제로 겪음).
            pid = None
            for aid in art.findall(".//article-id"):
                if aid.get("pub-id-type") in ("pmcaid", "pmcid", "pmc"):
                    pid = (aid.text or "").replace("PMC", "").strip()
                    if pid.isdigit():
                        break
            if pid:
                out[pid] = ET.tostring(art, encoding="unicode")
        time.sleep(pause)
    return out


# --------------------------------------------------------------------------
def _text(el):
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip() if el is not None else ""


def parse_article(xml_str):
    """전문 XML → {doi, title, sections, tables, fig_captions}"""
    art = ET.fromstring(xml_str)
    doi = None
    for aid in art.findall(".//article-id"):
        if aid.get("pub-id-type") == "doi":
            doi = (aid.text or "").strip()
    title = _text(art.find(".//article-title"))

    secs = []
    for sec in art.findall(".//body//sec"):
        st = _text(sec.find("./title"))
        secs.append((st, _text(sec)))
    body = _text(art.find(".//body"))

    tables = []
    for tw in art.findall(".//table-wrap"):
        cap = _text(tw.find(".//caption"))
        rows = []
        for tr in tw.findall(".//tr"):
            cells = [_text(td) for td in tr.findall("./td") + tr.findall("./th")]
            if cells:
                rows.append(cells)
        tables.append({"caption": cap, "rows": rows, "text": _text(tw)})

    figs = [_text(f.find(".//caption")) for f in art.findall(".//fig")]
    return {"doi": doi, "title": title, "sections": secs,
            "body": body, "tables": tables, "figures": figs}


# --------------------------------------------------------------------------
def extract_from_table(tb, lnp_pdf_mod):
    """조성표에서 행 단위로 (몰비, EE) 를 뽑는다.

    가장 신뢰도 높은 경로 — 표는 처방 하나가 한 행이라 짝짓기가 확실합니다.
    """
    rows = tb["rows"]
    if len(rows) < 2:
        return []
    header = [h.lower() for h in rows[0]]

    def _col(*keys):
        for i, h in enumerate(header):
            if any(k in h for k in keys):
                return i
        return None

    i_ratio = _col("molar ratio", "ratio", "composition", "formulation")
    i_ee = _col("encapsulation", "ee (%)", "ee(%)", "ee %", "entrapment")
    i_size = _col("size", "diameter", "z-ave")
    i_pdi = _col("pdi", "polydispers")
    i_zeta = _col("zeta")
    if i_ratio is None and i_ee is None:
        return []

    out = []
    for r in rows[1:]:
        if len(r) <= max([x for x in (i_ratio, i_ee) if x is not None]):
            continue
        ratio = r[i_ratio].strip() if i_ratio is not None else ""
        ee = r[i_ee].strip() if i_ee is not None else ""
        m = re.search(r"(\d{1,3}(?:\.\d+)?)", ee)
        ee_v = float(m.group(1)) if m else None
        rm = lnp_pdf_mod.RATIO_RE.search(ratio)
        if not rm:
            continue
        vals = [float(x) for x in rm.groups() if x]
        if not lnp_pdf_mod._plausible_ratio(vals + [None] * (4 - len(vals))):
            if not lnp_pdf_mod._plausible_ratio(vals):
                continue
        std = ":".join(f"{v:g}" for v in vals)
        rec = {"lipid_molar_ratio": std,
               "encapsulation_efficiency_percent_std_num": ee_v,
               "source": "table", "confidence": "high" if ee_v else "medium",
               "evidence": (tb["caption"][:120] + " | " + " ".join(r)[:160])}
        for key, idx in (("particle_size_nm_std_num", i_size),
                         ("pdi_std_num", i_pdi),
                         ("zeta_potential_mv_std_num", i_zeta)):
            if idx is not None and len(r) > idx:
                mm = re.search(r"(-?\d{1,3}(?:\.\d+)?)", r[idx])
                if mm:
                    rec[key] = float(mm.group(1))
        out.append(rec)
    return out


def harvest_one(xml_str, lnp_pdf_mod):
    """논문 하나에서 입력 후보 행들을 뽑는다.

    경로 우선순위:
      1. 표 (신뢰도 high)  — 처방 1개 = 1행, 짝짓기가 확실
      2. 본문 문장 (medium) — 조성 1개 + EE 1개가 명확할 때만
    """
    doc = parse_article(xml_str)
    body = doc["body"]

    # 지질 이름 / cargo
    low = body.lower()
    def _find(names):
        hits = [(nm, low.count(nm.lower())) for nm in names if nm.lower() in low]
        return [n for n, c in sorted(hits, key=lambda x: -x[1])]
    ion = _find(lnp_pdf_mod.IONIZABLE)
    helper = _find(lnp_pdf_mod.HELPER)
    peg = _find(lnp_pdf_mod.PEG_LIPID)
    cargo = None
    for lab, keys in lnp_pdf_mod.CARGO.items():
        if any(k in low for k in keys):
            cargo = lab
            break

    rows = []
    for tb in doc["tables"]:
        rows.extend(extract_from_table(tb, lnp_pdf_mod))

    # 본문 경로 — 표에서 못 얻었을 때만
    if not rows:
        ratios, ees = [], []
        for m in lnp_pdf_mod.RATIO_RE.finditer(body):
            vals = [float(x) if x else None for x in m.groups()]
            if lnp_pdf_mod._plausible_ratio(vals):
                v = [x for x in vals if x is not None]
                ev = body[max(0, m.start() - 130):m.end() + 130]
                order, sure = lnp_pdf_mod.detect_order(ev, len(v))
                raw = ":".join(f"{x:g}" for x in v)
                ratios.append({
                    "ratio": lnp_pdf_mod.reorder_ratio(raw, order) if sure else raw,
                    "evidence": ev, "order_detected": sure})
        GEN = re.compile(r"\b(typically|generally|usually|often|exceed\w*|"
                         r"greater than|more than|close to 100|well-optimized)\b", re.I)
        # 측정한 40편에서 확인한 실제 표기들:
        #   "encapsulation efficiency of 92.3%", "EE% of 85", "EE of ~90%",
        #   "encapsulation efficiencies were 88 and 91%", "%EE = 94"
        EE_EXTRA = [
            re.compile(r"(?:encapsulation efficien\w*|EE%?|%\s*EE)\s*"
                       r"(?:\([^)]{0,20}\))?\s*"
                       r"(?:of|was|were|is|are|:|=|reached|achiev\w+|up to)\s*"
                       r"(?:approximately|about|around|~|>)?\s*"
                       r"(\d{1,3}(?:\.\d+)?)\s*%?", re.I),
            re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%\s*(?:encapsulation|EE\b)", re.I),
        ]
        # 정확도 우선: EE 라는 말이 숫자 바로 옆(±90자)에 있어야 인정합니다.
        # 이 조건이 없으면 입자크기·TEM 문장의 숫자가 EE 로 둔갑합니다(실측).
        NEAR = re.compile(r"encapsulation efficien|entrapment efficien|\bEE\b|%\s*EE", re.I)
        # 계산식 문장 배제 — 실측된 최대 오탐원입니다.
        #   "EE% = 1 − (unencapsulated RNA / total RNA) × 100%"
        #   "EE (%) = (m₂ − m₁) / m₂ × 100%"
        # 여기의 100% 는 값이 아니라 백분율 환산 상수입니다. 40편 중 3편이
        # 이것 때문에 EE=100 으로 잘못 들어갔습니다.
        FORMULA = re.compile(
            r"(calculated as|determined as|according to the (?:following )?"
            r"(?:formula|equation)|EE\s*\(?%\)?\s*=|=\s*\(|×\s*100|\\times|\\frac|"
            r"following equation|as follows)", re.I)
        for rx in (lnp_pdf_mod.EE_RE, lnp_pdf_mod.EE_RE2, *EE_EXTRA):
            for m in rx.finditer(body):
                v = float(m.group(1))
                ev = body[max(0, m.start() - 130):m.end() + 130]
                near = body[max(0, m.start() - 90):m.end() + 90]
                # 매치 문자열 자체 + 앞 문맥을 함께 본다. 매치가 문장 앞부분에서
                # 시작하면 앞 문맥에는 계산식 표현이 없어 그냥 통과합니다(실측).
                span = m.group(0) + " " + body[max(0, m.start() - 160):m.start()]
                if (0 < v <= 100 and v >= 20 and not GEN.search(ev)
                        and NEAR.search(near) and not FORMULA.search(span)):
                    # "above 90%", "80%−95%", "approximately 90%" 는 정확한
                    # 측정치가 아니라 경계·근사값입니다. 버리지 않고 표시합니다
                    # — 원문 확인 시 정확한 값으로 바꿔 넣으라는 뜻입니다.
                    bound = bool(re.search(
                        r"(above|below|over|under|at least|up to|approximately|"
                        r"about|around|~|−\s*\d|-\s*\d{2}\s*%)", near, re.I))
                    ees.append({"ee": v, "evidence": ev, "pos": m.start(),
                                "approx": bound})
        uniq_r = list({r["ratio"]: r for r in ratios}.values())
        uniq_e = list({e["ee"]: e for e in ees}.values())

        # 짝짓기 규칙 — 확신도에 따라 등급을 나눕니다.
        #   조성1 + EE1                      → medium (짝이 자명)
        #   조성1 + EE여러개                  → medium (그 논문의 대표 조성에
        #                                      EE 최댓값을 붙이지 않고, 본문에서
        #                                      가장 가까운 EE 를 붙임)
        #   조성여러개                        → low (EE 없이 조성만)
        if len(uniq_r) == 1 and uniq_e:
            if len(uniq_e) == 1:
                pick = uniq_e[0]
            else:
                # 조성 문장과 본문 위치가 가장 가까운 EE 를 고른다
                rpos = body.find(uniq_r[0]["evidence"][:60])
                pick = min(uniq_e, key=lambda e: abs(e.get("pos", 0) - rpos))
                # 너무 멀면 (다른 섹션) 짝짓지 않는다 — 실측상 무관한 숫자가 붙음
                if abs(pick.get("pos", 0) - rpos) > 4000:
                    pick = None
            if pick is None:
                rows.append({"lipid_molar_ratio": uniq_r[0]["ratio"],
                             "encapsulation_efficiency_percent_std_num": None,
                             "source": "text", "confidence": "low",
                             "evidence": uniq_r[0]["evidence"][:200]})
                pick = {"ee": None, "evidence": ""}
            else:
                rows.append({
                    "lipid_molar_ratio": uniq_r[0]["ratio"],
                    "encapsulation_efficiency_percent_std_num": pick["ee"],
                    "source": "text",
                    "confidence": "medium" if len(uniq_e) == 1 else "low",
                    "ee_is_approximate": bool(pick.get("approx")),
                    "evidence": (uniq_r[0]["evidence"][:150] + "  ⟶EE근거: "
                                 + pick["evidence"][:150])})
        elif len(uniq_r) >= 1:
            for r in uniq_r[:8]:
                rows.append({"lipid_molar_ratio": r["ratio"],
                             "encapsulation_efficiency_percent_std_num": None,
                             "source": "text", "confidence": "low",
                             "evidence": r["evidence"][:200]})

    for r in rows:
        r.update({"reference_doi": doc["doi"] or "",
                  "ionizable_lipid_name": ion[0] if ion else "",
                  "helper_lipid_name": helper[0] if helper else "",
                  "peg_lipid_name": peg[0] if peg else "",
                  "cargo_type": cargo or "",
                  "title": doc["title"][:120]})
        if r.get("lipid_molar_ratio"):
            r["chem_warnings"] = "; ".join(
                lnp_pdf_mod.chemistry_check(r["lipid_molar_ratio"]))
    return rows, doc


def harvest(query=DEFAULT_QUERY, retmax=40, verbose=True, lnp_pdf_mod=None):
    """검색 → 수집 → 후보 표 반환. 통계도 함께 돌려줍니다."""
    if lnp_pdf_mod is None:
        import lnp_pdf as lnp_pdf_mod
    ids = search_pmc(query, retmax=retmax)
    if verbose:
        print(f"[검색] PMC {len(ids)}편")
    xmls = fetch_xml(ids)
    if verbose:
        print(f"[수집] 전문 XML {len(xmls)}편 확보")

    all_rows, stats = [], []
    for pid, xs in xmls.items():
        try:
            rows, doc = harvest_one(xs, lnp_pdf_mod)
        except Exception as e:
            stats.append({"pmcid": pid, "status": f"error:{type(e).__name__}",
                          "n_rows": 0, "n_tables": 0})
            continue
        usable = [r for r in rows
                  if r.get("encapsulation_efficiency_percent_std_num") is not None]
        stats.append({"pmcid": pid, "doi": rows[0]["reference_doi"] if rows else "",
                      "title": (rows[0]["title"] if rows else ""),
                      "n_tables": len(doc["tables"]),
                      "n_rows": len(rows), "n_usable": len(usable),
                      "status": "ok" if usable else ("ratio_only" if rows else "none")})
        for r in rows:
            r["pmcid"] = pid
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    st = pd.DataFrame(stats)
    if verbose and len(st):
        print(f"\n[수율]")
        print(f"  EE까지 확보한 논문 : {(st.status=='ok').sum()}/{len(st)}편")
        print(f"  조성만 확보        : {(st.status=='ratio_only').sum()}편")
        print(f"  아무것도 못 뽑음    : {(st.status=='none').sum()}편")
        if len(df):
            print(f"  총 후보 행         : {len(df)}행 "
                  f"(EE 있는 행 {df['encapsulation_efficiency_percent_std_num'].notna().sum()}행)")
    return df, st
