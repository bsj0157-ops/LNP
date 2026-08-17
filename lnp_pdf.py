# ==========================================================================
# lnp_pdf.py
# ==========================================================================
# LNP 논문 PDF에서 formulation 데이터를 추출하는 모듈
#
# 기존 app.py와 호환
#
# 핵심 수정:
#   - ratios 후보에 반드시 "sum" 필드 포함
#   - app.py에서 r["sum"]을 사용해도 KeyError가 발생하지 않음
#   - 기존 DATA_COLS 형식 유지
#   - 본문 + Table + Figure caption 검색
#   - 몰비 / EE / 성분 / N/P / pH / size / PDI / zeta 추출
#   - 표준 몰비 순서:
#       ionizable : helper : cholesterol : PEG
# ==========================================================================

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd


# ==========================================================================
# 1. 기존 데이터 형식 — 절대 변경하지 않음
# ==========================================================================

DATA_COLS = [
    "reference_doi",
    "lipid_molar_ratio",
    "ionizable_lipid_name",
    "encapsulation_efficiency_percent_std_num",
    "np_ratio_std_num",
    "buffer_ph_std_num",
    "cargo_type",
    "helper_lipid_name",
    "peg_lipid_name",
    "particle_size_nm_std_num",
    "pdi_std_num",
    "zeta_potential_mv_std_num",
    "ionizable_lipid_smiles",
    "source_note",
    "source",
    "confidence",
    "evidence",
    "title",
    "chem_warnings",
    "pmcid",
    "ee_is_approximate",
    "repair_note",
]


# ==========================================================================
# 2. 지질 사전
# ==========================================================================

IONIZABLE = [
    "DLin-MC3-DMA",
    "DLin-KC2-DMA",
    "DLinDMA",
    "SM-102",
    "SM102",
    "ALC-0315",
    "ALC0315",
    "C12-200",
    "cKK-E12",
    "OF-02",
    "306Oi10",
    "5A2-SC8",
    "Lipid 5",
    "L319",
    "TT3",
    "9A1P9",
    "FTT5",
    "BAMEA-O16B",
    "YSK05",
    "CL4H6",
    "306O13",
    "A6",
    "98N12-5",
    "7C1",
    "DODAP",
    "DODMA",
]


HELPER = [
    "DSPC",
    "DOPE",
    "POPC",
    "DPPC",
    "DOPC",
    "DSPE",
    "SOPC",
    "egg PC",
    "ESM",
]


PEG_LIPID = [
    "DMG-PEG2000",
    "DMG-PEG 2000",
    "DMG-PEG",
    "ALC-0159",
    "ALC0159",
    "PEG-DMG",
    "PEG2000-DMG",
    "DSPE-PEG2000",
    "DSPE-PEG",
    "C16-PEG2000",
    "PEG-DSG",
    "PEG-c-DMA",
    "PEG-DMPE",
]


CARGO = {
    "mRNA": [
        "mrna",
        "messenger rna",
        "modrna",
        "nucleoside-modified rna",
    ],
    "siRNA": [
        "sirna",
        "small interfering",
    ],
    "saRNA": [
        "sarna",
        "self-amplifying",
        "replicon",
    ],
    "pDNA": [
        "pdna",
        "plasmid dna",
        "plasmid",
    ],
    "ASO": [
        "antisense oligo",
        "aso ",
    ],
    "circRNA": [
        "circrna",
        "circular rna",
    ],
}


DOI_RE = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+[A-Za-z0-9])",
    re.I,
)

PMCID_RE = re.compile(
    r"\b(PMC\d{4,12})\b",
    re.I,
)


# ==========================================================================
# 3. PDF 읽기
# ==========================================================================

def _reset_file(file_or_bytes):
    if hasattr(file_or_bytes, "seek"):
        try:
            file_or_bytes.seek(0)
        except Exception:
            pass


def read_pdf(file_or_bytes):
    """
    PDF -> page별 text

    반환:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    _reset_file(file_or_bytes)

    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError(
            "pdfplumber가 필요합니다.\n"
            "터미널에서 다음을 실행하세요:\n"
            "pip install pdfplumber"
        ) from e

    pages = []

    with pdfplumber.open(file_or_bytes) as pdf:

        for i, page in enumerate(
            pdf.pages,
            start=1,
        ):

            try:
                text = page.extract_text(
                    x_tolerance=2,
                    y_tolerance=3,
                ) or ""
            except Exception:
                text = ""

            pages.append({
                "page": i,
                "text": text,
            })

    return pages


# ==========================================================================
# 4. PDF 표 읽기
# ==========================================================================

def read_pdf_tables(
    file_or_bytes,
    max_pages=100,
):
    """
    PDF 내부 Table 추출.

    반환:
        [
            {
                "page": 3,
                "table_index": 0,
                "df": DataFrame,
                "text": "..."
            }
        ]
    """

    _reset_file(file_or_bytes)

    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError(
            "pdfplumber가 필요합니다.\n"
            "pip install pdfplumber"
        ) from e

    result = []

    with pdfplumber.open(file_or_bytes) as pdf:

        for page_no, page in enumerate(
            pdf.pages[:max_pages],
            start=1,
        ):

            try:
                tables = page.extract_tables()
            except Exception:
                tables = []

            for table_index, table in enumerate(
                tables
            ):

                if not table:
                    continue

                rows = []

                for row in table:

                    if not row:
                        continue

                    cleaned = [
                        ""
                        if x is None
                        else str(x).strip()
                        for x in row
                    ]

                    if any(cleaned):
                        rows.append(cleaned)

                if not rows:
                    continue

                width = max(
                    len(x)
                    for x in rows
                )

                rows = [
                    x + [""] * (
                        width - len(x)
                    )
                    for x in rows
                ]

                header = rows[0]

                header_text = (
                    " ".join(header)
                    .lower()
                )

                if (
                    len(header_text) < 3
                    or sum(
                        ch.isdigit()
                        for ch in header_text
                    )
                    >
                    sum(
                        ch.isalpha()
                        for ch in header_text
                    )
                ):

                    header = [
                        f"col_{i + 1}"
                        for i in range(width)
                    ]

                    data = rows

                else:

                    data = rows[1:]

                try:
                    df = pd.DataFrame(
                        data,
                        columns=header,
                    )
                except Exception:
                    df = pd.DataFrame(
                        data
                    )

                table_text = "\n".join(
                    " | ".join(row)
                    for row in rows
                )

                result.append({
                    "page": page_no,
                    "table_index": table_index,
                    "df": df,
                    "text": table_text,
                })

    return result


# ==========================================================================
# 5. 문자열 유틸리티
# ==========================================================================

def _clean(x):

    if x is None:
        return ""

    x = str(x)

    x = x.replace(
        "\u00a0",
        " ",
    )

    x = x.replace(
        "\u2013",
        "-",
    )

    x = x.replace(
        "\u2014",
        "-",
    )

    x = x.replace(
        "\u2212",
        "-",
    )

    x = re.sub(
        r"[ \t]+",
        " ",
        x,
    )

    return x.strip()


def _norm(x):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        _clean(x).lower(),
    )


def _unique(items):

    out = []

    for x in items:

        x = _clean(x)

        if x and x not in out:
            out.append(x)

    return out


def _context(
    text,
    start,
    end,
    width=300,
):

    left = max(
        0,
        start - width,
    )

    right = min(
        len(text),
        end + width,
    )

    return _clean(
        text[left:right]
    )


# ==========================================================================
# 6. 숫자 추출
# ==========================================================================

def _numbers(
    pattern,
    text,
):

    vals = []

    for m in re.finditer(
        pattern,
        text,
        re.I,
    ):

        try:
            vals.append(
                float(m.group(1))
            )
        except Exception:
            pass

    return vals


def _first(
    pattern,
    text,
):

    m = re.search(
        pattern,
        text,
        re.I,
    )

    if not m:
        return None

    try:
        return float(
            m.group(1)
        )
    except Exception:
        return None


# ==========================================================================
# 7. Cargo 탐색
# ==========================================================================

def _find_cargo(text):

    low = text.lower()

    for cargo, words in CARGO.items():

        for word in words:

            if word.lower() in low:
                return cargo

    return None


# ==========================================================================
# 8. 지질 이름 탐색
# ==========================================================================

def _find_component(
    text,
    names,
):

    low = text.lower()

    found = []

    names = sorted(
        names,
        key=len,
        reverse=True,
    )

    for name in names:

        if name.lower() in low:

            if name not in found:
                found.append(name)

    return found


# ==========================================================================
# 9. 성분 순서 감지
# ==========================================================================

def detect_order(text):
    """
    표준 순서:
        ionizable
        helper
        cholesterol
        peg
    """

    low = text.lower()

    positions = {}

    for name in IONIZABLE:

        p = low.find(
            name.lower()
        )

        if p >= 0:
            positions.setdefault(
                "ionizable",
                p,
            )

    for name in HELPER:

        p = low.find(
            name.lower()
        )

        if p >= 0:
            positions.setdefault(
                "helper",
                p,
            )

    for name in PEG_LIPID:

        p = low.find(
            name.lower()
        )

        if p >= 0:
            positions.setdefault(
                "peg",
                p,
            )

    chol_patterns = [
        "cholesterol",
        "chol ",
        "chol.",
    ]

    for pattern in chol_patterns:

        p = low.find(
            pattern
        )

        if p >= 0:

            positions.setdefault(
                "cholesterol",
                p,
            )

            break

    return [
        k
        for k, _ in sorted(
            positions.items(),
            key=lambda x: x[1],
        )
    ]


# ==========================================================================
# 10. 몰비 파싱
# ==========================================================================

# 💡 [패치] 문장 끝 마침표로 인해 마지막 소수점 성분이 잘리는 문제 해결
RATIO_RE = re.compile(
    r"""
    (?<!\d)
    (
        \d+(?:\.\d+)?
        \s*[:/;,|-]\s*
        \d+(?:\.\d+)?
        (?:
            \s*[:/;,|-]\s*
            \d+(?:\.\d+)?
        ){2,3}
    )
    (?!\d)(?!\.\d)
    """,
    re.X,
)


def _parse_ratio(text):

    candidates = []

    for m in RATIO_RE.finditer(
        text
    ):

        raw = m.group(1)

        parts = re.split(
            r"\s*[:/;,|-]\s*",
            raw,
        )

        try:

            vals = [
                float(x)
                for x in parts
            ]

        except Exception:
            continue

        if len(vals) not in (
            3,
            4,
        ):
            continue

        if any(
            v < 0
            for v in vals
        ):
            continue

        total = sum(vals)

        if total <= 0:
            continue

        positive = [
            v
            for v in vals
            if v > 0
        ]

        if positive:

            if (
                max(vals)
                /
                max(
                    min(positive),
                    0.0001,
                )
                > 10000
            ):
                continue

        candidates.append({
            "raw": raw,
            "values": vals,
            "sum": total,
            "start": m.start(),
            "end": m.end(),
        })

    return candidates


# ==========================================================================
# 11. 몰비 표준화
# ==========================================================================

def _format_ratio(values):

    out = []

    for v in values:

        if abs(
            v - round(v)
        ) < 1e-8:

            out.append(
                str(
                    int(
                        round(v)
                    )
                )
            )

        else:

            out.append(
                f"{v:.4f}"
                .rstrip("0")
                .rstrip(".")
            )

    return ":".join(out)


def reorder_ratio(
    values,
    order,
):
    """
    논문에 등장한 성분 순서를
    표준 순서로 변환.

    표준:
        ionizable : helper : cholesterol : PEG
    """

    if len(values) != 4:
        return _format_ratio(
            values
        )

    if len(order) != 4:
        return _format_ratio(
            values
        )

    mapping = dict(
        zip(
            order,
            values,
        )
    )

    wanted = [
        "ionizable",
        "helper",
        "cholesterol",
        "peg",
    ]

    if not all(
        k in mapping
        for k in wanted
    ):
        return _format_ratio(
            values
        )

    return _format_ratio([
        mapping[k]
        for k in wanted
    ])


def _ratio_sum(
    ratio_string
):
    """
    표준화된 ratio 문자열의 합계를 계산.

    예:
        "50:10:38.5:1.5"
        -> 100.0
    """

    if ratio_string is None:
        return None

    try:

        parts = re.split(
            r"[:/;,|-]",
            str(ratio_string),
        )

        values = [
            float(x.strip())
            for x in parts
            if x.strip() != ""
        ]

        if not values:
            return None

        return round(
            sum(values),
            6,
        )

    except Exception:
        return None


# ==========================================================================
# 12. 화학적으로 이상한 몰비 검사
# ==========================================================================

def chemistry_check(
    ratio_str
):

    warnings = []

    try:

        vals = [
            float(x)
            for x in re.split(
                r"[:/;,|-]",
                str(ratio_str),
            )
        ]

    except Exception:

        return [
            "몰비 숫자를 해석하지 못했습니다."
        ]

    if len(vals) != 4:

        warnings.append(
            "표준 LNP 몰비는 4개 성분인지 확인하세요."
        )

        return warnings

    if any(
        v < 0
        for v in vals
    ):

        warnings.append(
            "몰비에 음수가 있습니다."
        )

    if vals[0] == 0:

        warnings.append(
            "이온화 지질 비율이 0입니다."
        )

    if vals[3] > 10:

        warnings.append(
            "PEG 지질 비율이 매우 높습니다. "
            "원문을 확인하세요."
        )

    if vals[1] > 50:

        warnings.append(
            "헬퍼 지질 비율이 매우 높습니다."
        )

    total = sum(vals)

    if total > 150:

        warnings.append(
            "몰비 합계가 매우 큽니다."
        )

    return warnings


# ==========================================================================
# 13. EE 탐색
# ==========================================================================

EE_RE = re.compile(
    r"""
    (?:
        encapsulation
        |
        entrapment
        |
        loading
        |
        encapsulated
        |
        entrapped
        |
        EE
    )
    [^.%]{0,100}?
    (\d+(?:\.\d+)?)
    \s*%
    """,
    re.I | re.X,
)


def _find_ee(text):

    found = []

    for m in EE_RE.finditer(
        text
    ):

        try:
            value = float(
                m.group(1)
            )

        except Exception:
            continue

        if not (
            0 < value <= 100
        ):
            continue

        evidence = _context(
            text,
            m.start(),
            m.end(),
            220,
        )

        found.append({
            "value": value,
            "evidence": evidence,
            "start": m.start(),
            "end": m.end(),
        })

    alt = re.compile(
        r"(\d+(?:\.\d+)?)\s*%"
        r"\s*(?:EE|encapsulation efficiency|encapsulation)",
        re.I,
    )

    for m in alt.finditer(
        text
    ):

        try:
            value = float(
                m.group(1)
            )
        except Exception:
            continue

        if 0 < value <= 100:

            found.append({
                "value": value,
                "evidence": _context(
                    text,
                    m.start(),
                    m.end(),
                    220,
                ),
                "start": m.start(),
                "end": m.end(),
            })

    return found


# ==========================================================================
# 14. N/P, pH, size, PDI, zeta
# ==========================================================================

def _find_process_values(
    text
):

    result = {
        "np_ratio": None,
        "ph": None,
        "size": None,
        "pdi": None,
        "zeta": None,
    }

    result["np_ratio"] = _first(
        r"N\s*/\s*P"
        r"[^0-9]{0,20}"
        r"(\d+(?:\.\d+)?)",
        text,
    )

    result["ph"] = _first(
        r"\bpH"
        r"[^0-9]{0,20}"
        r"(\d+(?:\.\d+)?)",
        text,
    )

    result["size"] = _first(
        r"(?:particle size|diameter|size)"
        r"[^0-9]{0,30}"
        r"(\d+(?:\.\d+)?)"
        r"\s*nm",
        text,
    )

    result["pdi"] = _first(
        r"\bPDI"
        r"[^0-9]{0,20}"
        r"(\d+(?:\.\d+)?)",
        text,
    )

    result["zeta"] = _first(
        r"(?:zeta|ζ)"
        r"[^0-9\-+]{0,20}"
        r"([\-+]?\d+(?:\.\d+)?)"
        r"\s*mV",
        text,
    )

    return result


# ==========================================================================
# 15. DOI / 제목 / PMCID
# ==========================================================================

def _find_metadata(
    pages
):

    all_text = "\n".join(
        p["text"]
        for p in pages
    )

    dois = []

    for m in DOI_RE.finditer(
        all_text
    ):

        doi = m.group(
            1
        ).rstrip(
            ".,);]"
        )

        if doi not in dois:
            dois.append(
                doi
            )

    pmcids = []

    for m in PMCID_RE.finditer(
        all_text
    ):

        x = m.group(
            1
        ).upper()

        if x not in pmcids:
            pmcids.append(
                x
            )

    title = ""

    if pages:

        lines = [
            _clean(x)
            for x in pages[
                "text"
            ].splitlines()
            if _clean(x)
        ]

        for line in lines[:20]:

            low = line.lower()

            if (
                len(line) >= 20
                and "abstract"
                not in low
                and "doi"
                not in low
                and "copyright"
                not in low
            ):

                title = line
                break

    return {
        "doi": (
            dois[0]
            if dois
            else ""
        ),
        "doi_alternatives": dois,
        "pmcid": (
            pmcids[0]
            if pmcids
            else ""
        ),
        "title": title,
    }


# ==========================================================================
# 16. 후보 공통 생성 함수
# ==========================================================================

def _make_candidate(
    ratio,
    context,
    page,
    source,
    base_confidence=0.40,
    ee=None,
):
    """
    모든 후보가 동일한 구조를 가지도록 만드는 함수.

    중요:
        "sum"을 반드시 포함한다.
    """

    order = detect_order(
        context
    )

    ratio_std = reorder_ratio(
        ratio["values"],
        order,
    )

    ratio_sum = _ratio_sum(
        ratio_std
    )

    ion = _find_component(
        context,
        IONIZABLE,
    )

    helper = _find_component(
        context,
        HELPER,
    )

    peg = _find_component(
        context,
        PEG_LIPID,
    )

    proc = _find_process_values(
        context
    )

    confidence = base_confidence

    if ee is not None:
        confidence += 0.25

    if ion:
        confidence += 0.10

    if helper:
        confidence += 0.05

    if peg:
        confidence += 0.05

    if len(order) == 4:
        confidence += 0.10

    confidence = min(
        confidence,
        0.99,
    )

    warnings = chemistry_check(
        ratio_std
    )

    ee_value = None
    ee_evidence = ""

    if isinstance(
        ee,
        dict,
    ):

        ee_value = ee.get(
            "value"
        )

        ee_evidence = (
            ee.get(
                "evidence"
            )
            or ""
        )

    elif ee is not None:

        ee_value = ee

    return {
        # --------------------------------------------------------------
        # 핵심 필드
        # --------------------------------------------------------------

        "ratio": ratio_std,

        # 원문에 쓰여 있던 몰비
        "ratio_as_written": ratio[
            "raw"
        ],

        # ★ 오류 수정 핵심
        "sum": ratio_sum,

        # 원본 숫자값
        "ratio_values": ratio[
            "values"
        ],

        "ee": ee_value,

        "ee_evidence": ee_evidence,

        "page": page,

        "evidence": context,

        "ionizable": (
            ion[0]
            if ion
            else None
        ),

        "helper": (
            helper[0]
            if helper
            else None
        ),

        "peg": (
            peg[0]
            if peg
            else None
        ),

        "order": order,

        "order_detected": (
            len(order) == 4
        ),

        "np_ratio": proc[
            "np_ratio"
        ],

        "ph": proc["ph"],

        "size": proc[
            "size"
        ],

        "pdi": proc[
            "pdi"
        ],

        "zeta": proc[
            "zeta"
        ],

        "cargo": _find_cargo(
            context
        ),

        "confidence": round(
            confidence,
            3,
        ),

        "chem_warnings": warnings,

        "source": source,
    }


# ==========================================================================
# 17. 본문 후보 추출
# ==========================================================================

def _text_candidates(
    pages
):

    candidates = []

    for page in pages:

        text = page[
            "text"
        ]

        ratios = _parse_ratio(
            text
        )

        if not ratios:
            continue

        for ratio in ratios:

            context = _context(
                text,
                ratio["start"],
                ratio["end"],
                500,
            )

            local_ees = _find_ee(
                context
            )

            ee = None

            if local_ees:

                # ratio 주변에서 가장 가까운 EE
                ratio_center = (
                    len(context) // 2
                )

                local_ees.sort(
                    key=lambda x:
                    abs(
                        (
                            x["start"]
                            + x["end"]
                        )
                        / 2
                        - ratio_center
                    )
                )

                ee = local_ees[0]

            candidate = _make_candidate(
                ratio=ratio,
                context=context,
                page=page["page"],
                source="PDF text",
                base_confidence=0.40,
                ee=ee,
            )

            candidates.append(
                candidate
            )

    return candidates


# ==========================================================================
# 18. Table 후보 추출
# ==========================================================================

def _table_candidates(
    tables
):

    candidates = []

    for table in tables:

        df = table[
            "df"
        ]

        table_text = table[
            "text"
        ]

        table_ratios = _parse_ratio(
            table_text
        )

        if not table_ratios:
            continue

        for _, row in df.iterrows():

            row_text = " | ".join(
                _clean(x)
                for x in row.tolist()
            )

            ratios = _parse_ratio(
                row_text
            )

            if not ratios:
                continue

            ees = _find_ee(
                row_text
            )

            ee = (
                ees[0]
                if ees
                else None
            )

            for ratio in ratios:

                candidate = _make_candidate(
                    ratio=ratio,
                    context=row_text,
                    page=table[
                        "page"
                    ],
                    source=(
                        f"PDF table p."
                        f"{table['page']}"
                    ),
                    base_confidence=0.55,
                    ee=ee,
                )

                candidates.append(
                    candidate
                )

    return candidates


# ==========================================================================
# 19. Figure caption 후보
# ==========================================================================

def _caption_candidates(
    pages
):

    candidates = []

    for page in pages:

        text = page[
            "text"
        ]

        chunks = re.split(
            r"(?=(?:Figure|Fig\.)\s*\d+)",
            text,
            flags=re.I,
        )

        for chunk in chunks:

            if not re.match(
                r"(?:Figure|Fig\.)\s*\d+",
                chunk,
                re.I,
            ):
                continue

            if len(chunk) > 1500:
                chunk = chunk[
                    :1500
                ]

            ratios = _parse_ratio(
                chunk
            )

            if not ratios:
                continue

            ees = _find_ee(
                chunk
            )

            for ratio in ratios:

                ee = (
                    ees[0]
                    if ees
                    else None
                )

                candidate = _make_candidate(
                    ratio=ratio,
                    context=_clean(
                        chunk
                    ),
                    page=page[
                        "page"
                    ],
                    source=(
                        f"Figure caption p."
                        f"{page['page']}"
                    ),
                    base_confidence=(
                        0.45
                    ),
                    ee=ee,
                )

                candidates.append(
                    candidate
                )

    return candidates


# ==========================================================================
# 20. 후보 중복 제거
# ==========================================================================

def _deduplicate(
    candidates
):

    result = []

    seen = set()

    candidates = sorted(
        candidates,
        key=lambda x: (
            x.get(
                "confidence",
                0
            ),
            x.get(
                "ee"
            )
            is not None,
            str(
                x.get(
                    "source",
                    ""
                )
            ).startswith(
                "PDF table"
            ),
        ),
        reverse=True,
    )

    for c in candidates:

        key = (
            c.get(
                "ratio"
            ),
            c.get(
                "ee"
            ),
            c.get(
                "ionizable"
            ),
        )

        if key in seen:
            continue

        seen.add(key)

        # 혹시 외부에서 들어온 후보가
        # sum을 빠뜨렸어도 여기서 보완
        if c.get(
            "sum"
        ) is None:

            c["sum"] = _ratio_sum(
                c.get(
                    "ratio"
                )
            )

        result.append(c)

    return result


# ==========================================================================
# 21. 최종 extract()
# ==========================================================================

def extract(
    file_or_bytes,
    max_ratio=25,
):
    """
    app.py에서 사용하는 메인 함수.

    기존 앱과 호환되는 반환 구조.
    """

    if isinstance(
        file_or_bytes,
        (
            bytes,
            bytearray,
        ),
    ):

        file_or_bytes = io.BytesIO(
            file_or_bytes
        )

    _reset_file(
        file_or_bytes
    )

    pages = read_pdf(
        file_or_bytes
    )

    _reset_file(
        file_or_bytes
    )

    tables = read_pdf_tables(
        file_or_bytes
    )

    meta = _find_metadata(
        pages
    )

    text_candidates = (
        _text_candidates(
            pages
        )
    )

    table_candidates = (
        _table_candidates(
            tables
        )
    )

    caption_candidates = (
        _caption_candidates(
            pages
        )
    )

    candidates = _deduplicate(
        table_candidates
        + text_candidates
        + caption_candidates
    )

    candidates = [
        x
        for x in candidates
        if x.get(
            "ratio"
        )
    ]

    candidates = sorted(
        candidates,
        key=lambda x: (
            x.get(
                "confidence",
                0
            ),
            x.get(
                "ee"
            )
            is not None,
        ),
        reverse=True,
    )[:max_ratio]

    # 모든 후보에 sum 보장
    for c in candidates:

        if c.get(
            "sum"
        ) is None:

            c["sum"] = _ratio_sum(
                c.get(
                    "ratio"
                )
            )

    # ------------------------------------------------------------------
    # 전체 PDF에서 별도로 찾는 값
    # ------------------------------------------------------------------

    all_text = "\n".join(
        p["text"]
        for p in pages
    )

    ionizable = _find_component(
        all_text,
        IONIZABLE,
    )

    helper = _find_component(
        all_text,
        HELPER,
    )

    peg = _find_component(
        all_text,
        PEG_LIPID,
    )

    cargo = _find_cargo(
        all_text
    )

    proc = _find_process_values(
        all_text
    )

    # ------------------------------------------------------------------
    # EE 후보
    # ------------------------------------------------------------------

    ee_specific = []

    for c in candidates:

        if c.get(
            "ee"
        ) is not None:

            ee_specific.append({
                "ee": c[
                    "ee"
                ],
                "page": c[
                    "page"
                ],
                "evidence": (
                    c.get(
                        "ee_evidence"
                    )
                    or c.get(
                        "evidence",
                        ""
                    )
                ),
                "generic": False,
            })

    # 본문에서 직접 찾은 EE
    for page in pages:

        for ee in _find_ee(
            page["text"]
        ):

            ee_specific.append({
                "ee": ee[
                    "value"
                ],
                "page": page[
                    "page"
                ],
                "evidence": ee[
                    "evidence"
                ],
                "generic": False,
            })

    # EE 중복 제거
    seen_ee = set()

    ee_specific_clean = []

    for x in ee_specific:

        key = (
            x["ee"],
            x["page"],
        )

        if key in seen_ee:
            continue

        seen_ee.add(
            key
        )

        ee_specific_clean.append(
            x
        )

    return {
        "n_pages": len(
            pages
        ),

        "doi": meta[
            "doi"
        ],

        "doi_alternatives": meta[
            "doi_alternatives"
        ],

        "title": meta[
            "title"
        ],

        "pmcid": meta[
            "pmcid"
        ],

        "ratios": candidates,

        "ee": ee_specific_clean,

        "ee_specific": ee_specific_clean,

        "ionizable": ionizable,

        "helper": helper,

        "peg": peg,

        "cargo": (
            [cargo]
            if cargo
            else []
        ),

        "np_ratio": (
            [
                {
                    "value":
                    proc[
                        "np_ratio"
                    ]
                }
            ]
            if proc[
                "np_ratio"
            ] is not None
            else []
        ),

        "ph": (
            [
                {
                    "value":
                    proc["ph"]
                }
            ]
            if proc["ph"]
            is not None
            else []
        ),

        "size": (
            [
                {
                    "value":
                    proc["size"]
                }
            ]
            if proc["size"]
            is not None
            else []
        ),

        "pdi": (
            [
                {
                    "value":
                    proc["pdi"]
                }
            ]
            if proc["pdi"]
            is not None
            else []
        ),

        "zeta": (
            [
                {
                    "value":
                    proc["zeta"]
                }
            ]
            if proc["zeta"]
            is not None
            else []
        ),

        "pages": pages,

        "tables": tables,
    }


# ==========================================================================
# 22. 빈 데이터 행
# ==========================================================================

def _blank_row():

    return {
        col: None
        for col in DATA_COLS
    }


# ==========================================================================
# 23. 후보 -> 기존 데이터 형식
# ==========================================================================

def _candidate_to_row(
    c,
    ex,
):

    row = _blank_row()

    row[
        "reference_doi"
    ] = (
        ex.get(
            "doi"
        )
        or ""
    )

    row[
        "lipid_molar_ratio"
    ] = (
        c.get(
            "ratio"
        )
        or ""
    )

    row[
        "ionizable_lipid_name"
    ] = (
        c.get(
            "ionizable"
        )
        or ""
    )

    row[
        "encapsulation_efficiency_percent_std_num"
    ] = c.get(
        "ee"
    )

    row[
        "np_ratio_std_num"
    ] = c.get(
        "np_ratio"
    )

    row[
        "buffer_ph_std_num"
    ] = c.get(
        "ph"
    )

    row[
        "cargo_type"
    ] = (
        c.get(
            "cargo"
        )
        or (
            ex.get(
                "cargo"
            )
            or [None]
        )[0]
    )

    row[
        "helper_lipid_name"
    ] = (
        c.get(
            "helper"
        )
        or ""
    )

    row[
        "peg_lipid_name"
    ] = (
        c.get(
            "peg"
        )
        or ""
    )

    row[
        "particle_size_nm_std_num"
    ] = c.get(
        "size"
    )

    row[
        "pdi_std_num"
    ] = c.get(
        "pdi"
    )

    row[
        "zeta_potential_mv_std_num"
    ] = c.get(
        "zeta"
    )

    # SMILES는 추측하지 않음
    row[
        "ionizable_lipid_smiles"
    ] = None

    row[
        "source_note"
    ] = (
        f"PDF p."
        f"{c.get('page')}"
    )

    row[
        "source"
    ] = (
        c.get(
            "source"
        )
        or "PDF"
    )

    row[
        "confidence"
    ] = c.get(
        "confidence"
    )

    evidence = c.get(
        "evidence",
        "",
    )

    if c.get(
        "ee_evidence"
    ):

        evidence += (
            "\n\n[EE evidence]\n"
            + c[
                "ee_evidence"
            ]
        )

    row[
        "evidence"
    ] = evidence

    row[
        "title"
    ] = (
        ex.get(
            "title"
        )
        or ""
    )

    row[
        "chem_warnings"
    ] = "; ".join(
        c.get(
            "chem_warnings",
            [],
        )
    )

    row[
        "pmcid"
    ] = (
        ex.get(
            "pmcid"
        )
        or ""
    )

    row[
        "ee_is_approximate"
    ] = False

    repair_notes = []

    if c.get(
        "ratio_as_written"
    ):

        if (
            c[
                "ratio_as_written"
            ]
            != c.get(
                "ratio"
            )
        ):

            repair_notes.append(
                "논문 표기 몰비를 표준 순서 "
                "ionizable:helper:cholesterol:PEG "
                "형식으로 변환함"
            )

    if not c.get(
        "order_detected"
    ):

        repair_notes.append(
            "성분 순서를 명확하게 확인하지 못함"
        )

    if c.get(
        "ee"
    ) is None:

        repair_notes.append(
            "EE 값이 PDF 본문에서 확인되지 않음"
        )

    row[
        "repair_note"
    ] = " | ".join(
        repair_notes
    )

    return row


# ==========================================================================
# 24. Draft 생성
# ==========================================================================

def to_draft_rows(
    ex,
    max_rows=12,
):
    """
    Streamlit st.data_editor()에 바로 넣을 DataFrame.

    기존 앱과 동일한 DATA_COLS 반환.
    """

    candidates = ex.get(
        "ratios",
        [],
    )

    rows = []

    for c in candidates[
        :max_rows
    ]:

        row = _candidate_to_row(
            c,
            ex,
        )

        rows.append(
            row
        )

    if not rows:

        return pd.DataFrame(
            columns=DATA_COLS
        )

    df = pd.DataFrame(
        rows,
        columns=DATA_COLS,
    )

    numeric_cols = [
        "encapsulation_efficiency_percent_std_num",
        "np_ratio_std_num",
        "buffer_ph_std_num",
        "particle_size_nm_std_num",
        "pdi_std_num",
        "zeta_potential_mv_std_num",
        "confidence",
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df


# ==========================================================================
# 25. 후보 요약
# ==========================================================================

def get_formulation_candidates(
    ex
):

    return ex.get(
        "ratios",
        [],
    )


def candidate_summary(
    ex
):

    candidates = ex.get(
        "ratios",
        [],
    )

    return [
        {
            "page": c.get(
                "page"
            ),

            "ratio": c.get(
                "ratio"
            ),

            # ★ app.py에서 사용하는 sum 포함
            "sum": c.get(
                "sum"
            ),

            "EE": c.get(
                "ee"
            ),

            "ionizable": c.get(
                "ionizable"
            ),

            "helper": c.get(
                "helper"
            ),

            "PEG": c.get(
                "peg"
            ),

            "confidence": c.get(
                "confidence"
            ),
        }

        for c in candidates
    ]
