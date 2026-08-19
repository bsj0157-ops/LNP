# -*- coding: utf-8 -*-
"""사이드바 정확도 문구를 현재 데이터로 계산합니다.

앱은 구간별 정확도를 문자열에 박아 두고 있습니다("7.0 %p / 44.3 %p").
데이터가 늘면 그 숫자는 조용히 틀려집니다 — 측정 시점에 683행 기준으로
7.8 / 43.6 이었고, 표시된 값과 어긋났습니다.

사용법
------
    import lnp_app_livenote as LN
    st.sidebar.markdown(LN.accuracy_note(work_df, oof))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EE = "encapsulation_efficiency_percent_std_num"

BANDS = ((85.0, 101.0, "85% 이상"),
         (70.0, 85.0, "70~85%"),
         (50.0, 70.0, "50~70%"),
         (0.0, 50.0, "50% 미만"))


def band_table(df: pd.DataFrame, oof: pd.Series) -> pd.DataFrame:
    """실측 EE 구간별 out-of-fold MAE 를 계산합니다."""
    if oof is None or len(oof) == 0 or EE not in df:
        return pd.DataFrame(columns=["구간", "행 수", "MAE (%p)"])
    idx = oof.index.intersection(df.index)
    y = pd.to_numeric(df.loc[idx, EE], errors="coerce")
    p = oof.reindex(idx)
    err = (y - p).abs()
    rows = []
    for lo, hi, lab in BANDS:
        m = (y >= lo) & (y < hi) & err.notna()
        if int(m.sum()):
            rows.append({"구간": lab, "행 수": int(m.sum()),
                         "MAE (%p)": round(float(err[m].mean()), 1)})
    return pd.DataFrame(rows)


def accuracy_note(df: pd.DataFrame, oof: pd.Series, base_note: str = "") -> str:
    """사이드바용 마크다운을 만듭니다. 수치는 모두 현재 데이터로 계산합니다."""
    t = band_table(df, oof)
    if t.empty:
        return base_note + "\n\n_구간별 정확도: 예측을 계산할 수 없습니다(논문 2편 미만)._"

    idx = oof.index.intersection(df.index)
    y = pd.to_numeric(df.loc[idx, EE], errors="coerce")
    err = (y - oof.reindex(idx)).abs()
    overall = float(err.mean())

    worst = t.loc[t["MAE (%p)"].idxmax()]
    best = t.loc[t["MAE (%p)"].idxmin()]

    lines = [base_note, "",
             f"**구간별 정확도 (현재 {len(idx)}행 실측)**", "",
             "| 실측 EE 구간 | 행 수 | MAE |", "|---|---|---|"]
    for _, r in t.iterrows():
        lines.append(f"| {r['구간']} | {int(r['행 수'])} | {r['MAE (%p)']:.1f} %p |")
    lines += ["", f"전체 MAE **{overall:.1f} %p**. "
              f"{best['구간']} 구간이 가장 정확하고({best['MAE (%p)']:.1f} %p), "
              f"{worst['구간']} 구간은 {worst['MAE (%p)']:.1f} %p 로 "
              "예측을 신뢰하기 어렵습니다."]
    return "\n".join(lines)
