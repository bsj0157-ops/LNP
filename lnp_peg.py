# -*- coding: utf-8 -*-
"""PEG 비율만 변경했을 때의 EE 변화 예측.

네 성분 중 PEG 만 별도 기능으로 분리한 이유가 있습니다. 다른 세 성분은
EE 를 설명한다는 증거가 없었지만(음성대조군 대비 1.4~1.8배), PEG 는
논문 고정효과 회귀에서 견고한 기울기를 보였습니다. 실측 결과입니다.

--------------------------------------------------------------------------
검증 결과 (553행 / 91편, 논문 단위 5-fold CV)
--------------------------------------------------------------------------
[1] 논문 내 PEG 기울기는 실재합니다 — 단 구간에 따라 부호가 뒤집힙니다.

    논문 고정효과 회귀 (EE ~ PEG + 논문더미), 전 구간:
        계수 -2.50 %p EE / PEG 1 %p    95% CI [-3.23, -1.77]   p=4.4e-11
        논문 하나씩 빼고 91회 재추정: -3.29 ~ -0.51, 100% 음수

    구간을 나누면 이야기가 달라집니다:
        PEG >= 2.5% :  계수 -2.89  p<0.001  n=269/25편
                       LOPO 25회: -3.49 ~ -0.84, 100% 음수  <- 견고
        PEG <  2.5% :  계수 +3.74  p=0.25   n=284/66편
                       LOPO 80회: +1.76 ~ +7.48, 0% 음수    <- 부호 반대

    즉 "PEG 를 낮추면 EE 가 오른다"는 **PEG 2.5% 이상에서만** 맞습니다.
    2.5% 미만에서는 오히려 반대 방향이며 유의하지도 않습니다.
    이차항은 유의하지 않았습니다 (p=0.76) — 구간 내에서는 선형입니다.

[2] 방향 예측 정확도 (논문 단위 CV, 학습 데이터에 없는 논문으로 검증)

    PEG 가 지배적으로 다른 논문 내 처방 쌍(ΔPEG > 나머지 세 성분 변화 합)
    94개를 대상으로 측정했습니다.

        PEG >= 2.5% 구간   85쌍 / 6편   방향 적중 83.5%   p<0.0001
              논문별: 100, 100, 100, 78, 67, 33 %
        PEG <  2.5% 구간    9쌍 / 2편   방향 적중 22.2%   p=0.18
              -> 이 구간은 예측을 제공하지 않습니다

    실측 데이터만으로도 같은 방향입니다: PEG 지배 쌍 94개 중
    'PEG 증가 -> EE 감소' 가 75.5% (p<0.0001).

[3] 크기는 신뢰할 수 없습니다.

    변화량 MAE 33.1 %p (실제 변화량 평균 크기 36.3 %p).
    예측 변화량과 실제 변화량의 순위 상관은 rho=+0.64 (선형 기울기 방식)
    이지만, 절대 크기는 절반 이상 틀립니다. 방향과 대략적 규모까지만
    읽으십시오.

[4] 주의 — 표본이 편중되어 있습니다.

    94쌍 중 46쌍이 한 논문(10.1016/j.ijpharm.2023.123050)에서 나옵니다.
    그 논문을 제외하면 방향 적중률이 52~56% 로 떨어집니다(p>0.4).
    기울기 추정 자체는 91편 전체에서 LOPO 100% 음수로 견고하지만,
    방향 적중률 83.5% 라는 수치는 소수 논문에 의존합니다.

--------------------------------------------------------------------------
왜 RandomForest 대신 회귀 기울기를 쓰는가
--------------------------------------------------------------------------
같은 94쌍에서 두 방식을 비교했습니다.
    RF 방식      방향 83.5% (구간분리 후), 변화량 MAE 34.3, rho=+0.43
    회귀 기울기  방향 83.5%,               변화량 MAE 33.1, rho=+0.64
회귀가 크기 상관에서 낫고, 무엇보다 계수와 신뢰구간을 그대로 보여줄 수
있어 사용자가 근거를 확인할 수 있습니다. RF 는 왜 그런 값이 나왔는지
설명할 수 없습니다.
"""
import numpy as np
import pandas as pd

# PEG 기울기의 부호가 뒤집히는 경계 — 구간별 LOPO 로 확인한 값
PEG_BREAK = 2.5

# 전 구간 참고용 (구간 분리 전): -2.50 %p EE / PEG 1 %p, p=4.4e-11
SLOPE_ALL = -2.502

EE_COL = "encapsulation_efficiency_percent_std_num"
RATIO_COL = "lipid_molar_ratio"


# --------------------------------------------------------------------------
def fit_peg_slope(df, ee_col=EE_COL, paper_col="reference_doi"):
    """논문 고정효과 회귀로 PEG 기울기를 구간별로 추정합니다.

    논문더미를 넣는 이유: PEG 와 EE 의 전체 상관 rho=-0.345 중 상당 부분이
    논문 간 차이입니다(논문 평균끼리 rho=-0.265, p=0.011). 논문더미 없이
    회귀하면 'PEG 낮은 논문이 EE 를 높게 보고했다'를 'PEG 를 낮추면 EE 가
    오른다'로 잘못 읽습니다.

    반환: {"high": {...}, "low": {...}} — 각 구간의 slope/se/ci/p/n/n_papers,
          그리고 resid_sd (개별 예측 불확실성).
    """
    import statsmodels.formula.api as smf

    d = pd.DataFrame({
        "ee": pd.to_numeric(df[ee_col], errors="coerce"),
        "peg": _peg_of(df),
        "g": df[paper_col].astype(str).str.strip().str.lower(),
    }).dropna()
    d = d[(d.ee > 0) & (d.ee <= 100)]

    out = {}
    for name, (lo, hi) in [("high", (PEG_BREAK, np.inf)),
                           ("low", (0.0, PEG_BREAK))]:
        s = d[(d.peg >= lo) & (d.peg < hi)]
        if s.g.nunique() < 3 or len(s) < 20:
            out[name] = None
            continue
        f = smf.ols("ee ~ peg + C(g)", data=s).fit()
        ci = f.conf_int().loc["peg"]
        out[name] = {
            "slope": float(f.params["peg"]),
            "se": float(f.bse["peg"]),
            "ci": (float(ci[0]), float(ci[1])),
            "p": float(f.pvalues["peg"]),
            "n": int(len(s)),
            "n_papers": int(s.g.nunique()),
            "resid_sd": float(np.sqrt(f.mse_resid)),
        }
    out["_data"] = {"n": int(len(d)), "n_papers": int(d.g.nunique()),
                    "peg_range": (float(d.peg.min()), float(d.peg.max()))}
    return out


def _pos_of(df, row_idx):
    """row_idx 를 위치(iloc) 로 정규화합니다.

    이 함수가 필요한 이유: app.py 의 selectbox 는 df.index 라벨을 넘기지만
    내부 계산은 .iloc 을 씁니다. df 가 필터링·정렬을 거쳐 인덱스가 연속이
    아니면 둘이 어긋나 **다른 처방의 PEG 를 읽고도 오류 없이 값을 돌려줍니다**.
    라벨이 인덱스에 있으면 라벨로, 없으면 위치로 해석합니다.
    """
    idx = df.index
    if row_idx in idx:
        locs = np.flatnonzero(idx == row_idx)
        if len(locs) > 1:
            raise ValueError(f"인덱스 {row_idx} 가 중복됩니다 — "
                             f"df.reset_index(drop=True) 후 사용하십시오.")
        return int(locs[0])
    if isinstance(row_idx, (int, np.integer)) and 0 <= int(row_idx) < len(df):
        return int(row_idx)
    raise KeyError(f"{row_idx} 를 df 에서 찾을 수 없습니다.")


def _peg_of(df):
    """처방 문자열에서 PEG 몰비(합 100 정규화)를 뽑습니다."""
    if RATIO_COL not in df.columns:
        return pd.Series(np.nan, index=df.index)
    txt = df[RATIO_COL].astype(str).str.strip()
    parts = (txt.str.replace(r"[\/\-,;|]", ":", regex=True)
             .str.split(":", expand=True).apply(pd.to_numeric, errors="coerce"))
    nv = parts.notna().sum(axis=1)
    peg = pd.Series(np.nan, index=df.index, dtype=float)
    tot = pd.Series(np.nan, index=df.index, dtype=float)
    for k in (3, 4):                       # 3성분(헬퍼 없음) / 4성분 모두 PEG 는 마지막
        m = nv == k
        if m.any() and (k - 1) in parts.columns:
            peg.loc[m] = parts.loc[m, k - 1].values
            tot.loc[m] = parts.loc[m, list(range(k))].sum(axis=1).values
    return peg / tot.replace(0, np.nan) * 100.0


# --------------------------------------------------------------------------
def predict_peg_change(df, row_idx, new_peg, fit=None,
                       ee_col=EE_COL, paper_col="reference_doi"):
    """row_idx 처방의 PEG 만 new_peg 로 바꿨을 때 EE 예측.

    나머지 세 성분은 서로의 비율을 유지한 채 합이 100 이 되도록
    재정규화합니다(PEG 만 바꾸는 것이 물리적으로 의미하는 바입니다).

    반환 dict 의 'usable' 이 False 면 그 예측을 쓰지 마십시오 —
    검증에서 방향 적중률이 22% 였던 구간이라는 뜻입니다.
    """
    if fit is None:
        fit = fit_peg_slope(df, ee_col, paper_col)

    pos = _pos_of(df, row_idx)
    peg_now = float(_peg_of(df).iloc[pos])
    if not np.isfinite(peg_now):
        raise ValueError(f"[{row_idx}] 행의 PEG 비율을 읽을 수 없습니다.")
    new_peg = float(new_peg)

    # 어느 구간의 기울기를 쓸지: 변경 전후 중 높은 쪽이 경계를 넘으면 high
    seg = "high" if max(peg_now, new_peg) >= PEG_BREAK else "low"
    f = fit.get(seg)
    # 목표값이 검증 구간(>=2.5%) 밖이면 high 기울기로 외삽하게 됩니다.
    # 예: 3.0% -> 0.5% 를 -2.89 기울기로 계산하면 EE +7.2 %p 라는 근거 없는
    # 값이 나옵니다. 목표값 자체가 검증 구간 안에 있어야 합니다.
    target_in_range = new_peg >= PEG_BREAK
    if f is None:
        raise ValueError(f"{seg} 구간을 추정할 데이터가 부족합니다.")

    measured = pd.to_numeric(df[ee_col], errors="coerce").iloc[pos] \
        if ee_col in df.columns else np.nan
    d_peg = new_peg - peg_now
    delta = f["slope"] * d_peg
    d_lo = f["ci"][0] * d_peg
    d_hi = f["ci"][1] * d_peg
    if d_lo > d_hi:
        d_lo, d_hi = d_hi, d_lo

    base = float(measured) if np.isfinite(measured) else np.nan
    pred = np.clip(base + delta, 0, 100) if np.isfinite(base) else np.nan
    # EE 상한 100 에 부딪히는지 (실측 10.5% 가 95 초과)
    clipped = bool(np.isfinite(base) and (base + delta > 100 or base + delta < 0))

    usable = (seg == "high") and target_in_range
    if seg == "high" and not target_in_range:
        note = (f"목표 PEG {new_peg:.2f}% 가 검증 구간(2.5% 이상) 밖입니다. "
                f"검증 구간의 기울기(-2.89)를 그 아래로 연장한 값이므로 "
                f"외삽이며 근거가 없습니다. 2.5% 미만 구간의 실제 기울기는 "
                f"부호가 반대(+3.74)였고 유의하지 않았습니다(p=0.40).")
    elif not usable:
        note = ("PEG 2.5% 미만 구간입니다. 이 구간의 기울기는 부호가 "
                "불안정하고(LOPO 100% 양수, p=0.40) 방향 적중률이 22% "
                "였습니다. 예측을 사용하지 마십시오.")
    else:
        note = ("검증 구간입니다 (PEG >= 2.5%, 방향 적중 83.5%, n=85/6편).")
    return {
        "row_idx": row_idx,
        "row_pos": pos,
        "peg_before": peg_now,
        "peg_after": new_peg,
        "d_peg": d_peg,
        "segment": seg,
        "slope": f["slope"],
        "slope_ci": f["ci"],
        "slope_p": f["p"],
        "measured_ee": None if not np.isfinite(measured) else float(measured),
        "delta_ee": float(delta),
        "delta_ci": (float(d_lo), float(d_hi)),
        "pred_ee": None if not np.isfinite(pred) else float(pred),
        "pred_sd": f["resid_sd"],
        "clipped_at_bound": clipped,
        "usable": usable,
        "target_in_range": bool(target_in_range),
        "direction": ("상승" if delta > 0 else "하락") if abs(delta) > 1e-9 else "변화 없음",
        "note": note,
    }


def peg_curve(df, row_idx, fit=None, peg_min=0.5, peg_max=8.0, n=31,
              ee_col=EE_COL, paper_col="reference_doi"):
    """PEG 를 훑으며 예측 EE 곡선을 냅니다 (그래프용).

    구간 경계에서 기울기가 바뀌므로 곡선이 꺾입니다 — 이는 결함이 아니라
    실측 결과입니다(PEG>=2.5 에서 -2.89, 미만에서 +3.74).
    usable=False 구간은 lo/hi 를 NaN 으로 두어 그래프에서 구별됩니다.
    """
    if fit is None:
        fit = fit_peg_slope(df, ee_col, paper_col)
    rows = []
    for p in np.linspace(peg_min, peg_max, n):
        try:
            r = predict_peg_change(df, row_idx, p, fit=fit,
                                   ee_col=ee_col, paper_col=paper_col)
        except ValueError:
            continue
        rows.append({
            "peg": p, "pred_ee": r["pred_ee"],
            "lo": (r["pred_ee"] - r["pred_sd"]
                   if r["pred_ee"] is not None else np.nan),
            "hi": (r["pred_ee"] + r["pred_sd"]
                   if r["pred_ee"] is not None else np.nan),
            "usable": r["usable"], "segment": r["segment"],
        })
    return pd.DataFrame(rows)


def recommend_peg(df, row_idx, fit=None, ee_col=EE_COL,
                  paper_col="reference_doi"):
    """이 처방에서 PEG 를 어느 방향으로 움직여야 하는지 한 줄 권고.

    데이터가 지지하는 범위(PEG >= 2.5%) 안에서만 권고합니다.
    이미 2.5% 미만이면 '근거 없음'을 돌려줍니다 — 더 낮추라고 하지 않습니다.
    """
    if fit is None:
        fit = fit_peg_slope(df, ee_col, paper_col)
    peg_now = float(_peg_of(df).iloc[_pos_of(df, row_idx)])
    if not np.isfinite(peg_now):
        return {"actionable": False, "text": "PEG 비율을 읽을 수 없습니다."}

    if peg_now < PEG_BREAK:
        return {
            "actionable": False, "peg_now": peg_now,
            "text": (f"현재 PEG {peg_now:.2f}% 는 이미 검증 구간(2.5% 이상) "
                     f"아래입니다. 이 구간에서 PEG 를 더 조절하는 근거는 "
                     f"데이터에 없습니다 (기울기 +3.74, p=0.25, 방향 적중 22%)."),
        }
    f = fit["high"]
    target = PEG_BREAK
    gain = f["slope"] * (target - peg_now)
    gain_lo = f["ci"][1] * (target - peg_now)      # 보수적 하한
    gain_hi = f["ci"][0] * (target - peg_now)
    if gain_lo > gain_hi:
        gain_lo, gain_hi = gain_hi, gain_lo
    return {
        "actionable": True, "peg_now": peg_now, "peg_target": target,
        "expected_gain": float(gain),
        "gain_ci": (float(gain_lo), float(gain_hi)),
        "text": (f"현재 PEG {peg_now:.2f}% -> {target:.1f}% 로 낮추면 EE 가 "
                 f"약 {gain:+.1f} %p 변할 것으로 추정됩니다 "
                 f"(95% CI {gain_lo:+.1f} ~ {gain_hi:+.1f} %p). "
                 f"방향은 검증됐지만(83.5%) 크기는 MAE 33 %p 로 신뢰할 수 "
                 f"없으니 실험으로 확인하십시오."),
    }


CAVEAT = """**PEG 비율 변경 예측 — 네 성분 중 유일하게 검증을 통과한 기능입니다.**

553행 / 91편, 논문 단위 교차검증 실측 결과입니다.

| 항목 | 결과 |
|---|---|
| 논문 내 PEG 기울기 (전 구간) | −2.50 %p EE / PEG 1 %p, 95% CI [−3.23, −1.77], p=4.4e-11 |
| 기울기 안정성 | 논문 91회 LOPO 재추정 **100% 음수** |
| **PEG ≥ 2.5% 구간** | 기울기 −2.89 (p<0.001) · **방향 적중 83.5%** (n=85 / 6편) |
| **PEG < 2.5% 구간** | 기울기 **+3.74** (p=0.25) · 방향 적중 22% → **사용 불가** |
| 변화량 크기 | MAE 33 %p → 크기는 신뢰 불가, 방향만 |

**PEG 2.5% 가 경계입니다.** 그 이상에서 PEG 를 낮추면 EE 가 오르는 것이
데이터로 확인되지만, 2.5% 미만에서는 기울기 부호가 뒤집히고 유의하지도
않습니다. 이 기능은 2.5% 미만 처방에는 예측을 제공하지 않습니다.

**한계** — 방향 적중률을 측정한 94쌍 중 46쌍이 한 논문에서 나왔습니다.
그 논문을 빼면 적중률이 52~56% 로 떨어집니다. 기울기 추정 자체는 91편
전체에서 견고하지만, 83.5% 라는 수치는 소수 논문에 의존합니다.
"""
