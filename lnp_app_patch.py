# -*- coding: utf-8 -*-
"""app.py 의 정확도 손실 지점 3곳을 고치는 모듈.

app.py 에서 아래 세 줄만 바꿔 끼우면 됩니다.

  1) 탭3 CSV 업로드 — EE 문자열 정제
       (기존) d_clean[ee_col].astype(str).str.replace(r'[^0-9.]','',regex=True)
       (수정) d_clean[ee_col].map(robust_ee)

  2) 탭3 add_rows 직전 — 중복 제거
       (추가) valid_d = dedupe(valid_d)

  3) 앵커링 탭 — 학습 데이터 재예측 대신 홀드아웃 평가
       (추가) anchor_report(df) 를 화면에 띄웁니다.

각 항목의 실측 근거는 함수 docstring 에 적어 두었습니다.
"""
import io
import re
import contextlib

import numpy as np
import pandas as pd

EEC = "encapsulation_efficiency_percent_std_num"
DEDUPE_KEYS = ["reference_doi", "lipid_molar_ratio", EEC]


# --------------------------------------------------------------------------
# 1) EE 문자열 파서
# --------------------------------------------------------------------------
def robust_ee(s):
    """EE 문자열 -> 숫자. 범위·오차표기·분율을 모두 살립니다.

    왜 필요한가 — Atlas 원본 EE 문자열 628개에 실측한 결과:
        기존 app.py 정제 (숫자·점만 남기기)   유효 521개  (손실 107개)
        이 함수                                유효 624개  (손실   4개)
    기존 방식이 망가지는 이유는 구분자를 지우고 숫자를 이어붙이기 때문입니다.
        '92.08 ± 2.5' -> '92.082.5' -> NaN     (점이 둘이라 숫자 변환 실패)
        '85-90'       -> '8590'                (범위가 한 숫자로)
        '88 (n=3)'    -> '883'                 (표본 수가 값에 붙음)
        '0.94'        -> 0.94                  (분율인데 %로 안 바뀜)
    """
    t = str(s).strip()
    if not t or t.lower() in ("nan", "n.d.", "nd", "na", "none", "-", "--"):
        return np.nan

    # 표본 수·산포 괄호를 먼저 떼어냅니다: '88 (n=3)'
    t = re.sub(r"\((?:n\s*=\s*\d+|SD|SEM|s\.?d\.?)[^)]*\)", "", t, flags=re.I)
    # 오차 표기를 떼어냅니다: '92.08 ± 2.5'
    t = re.sub(r"\s*±\s*[\d.]+.*$", "", t)

    # 범위 표기는 중간값을 씁니다: '85-90' -> 87.5
    # (범위 판정을 오차 제거보다 뒤에 둬야 '69-100' 이 69 로 잘리지 않습니다)
    rng = re.match(r"^\s*[~≈>≥<≤]?\s*([\d.]+)\s*(?:-|–|—|to|~)\s*([\d.]+)",
                   t, re.I)
    if rng:
        a, b = float(rng.group(1)), float(rng.group(2))
        if b > a:
            return (a + b) / 2

    m = re.search(r"([\d.]+)", t)
    if not m:
        return np.nan
    try:
        v = float(m.group(1))
    except ValueError:
        return np.nan

    if 0 < v <= 1:          # 분율 표기 (0.94 -> 94)
        v *= 100
    return v if 0 < v <= 100 else np.nan


def clean_ee_column(series):
    """컬럼 단위 적용 + 회수 건수를 함께 돌려줍니다."""
    old = pd.to_numeric(
        series.astype(str).str.replace(r"[^0-9.]", "", regex=True),
        errors="coerce")
    new = series.map(robust_ee)
    recovered = int((new.between(0, 100) & ~old.between(0, 100)).sum())
    return new, recovered


# --------------------------------------------------------------------------
# 2) 중복 제거
# --------------------------------------------------------------------------
def dedupe(df, keys=None, verbose=True):
    """같은 논문 + 같은 조성 + 같은 EE 인 행을 하나만 남깁니다.

    왜 필요한가 — Atlas(628행)와 web 수확(680행)은 논문 49편을 공유합니다.
    그대로 합치면 1308행 중 754행이 중복입니다. 실측 결과:
        중복 포함   모델 MAE 17.25  baseline 17.11  개선 -0.8%
        중복 제거   모델 MAE 17.22  baseline 17.66  개선 +2.5%
    중복은 같은 값을 여러 번 학습시켜 baseline 을 인위적으로 강하게 만들고,
    논문 단위 CV 의 실질 표본 수를 실제보다 많아 보이게 합니다.
    """
    keys = keys or [k for k in (DEDUPE_KEYS) if k in df.columns]
    if not keys:
        return df
    n0 = len(df)
    out = df.drop_duplicates(subset=keys).reset_index(drop=True)
    if verbose and n0 != len(out):
        print(f"[중복 제거] {n0}행 -> {len(out)}행 (제거 {n0 - len(out)}행)")
    return out


# --------------------------------------------------------------------------
# 3) 앵커링 정직 평가
# --------------------------------------------------------------------------
def anchor_report(df, v3_module, anchor_module, k=3, min_rows=5):
    """앵커링 효과를 논문 홀드아웃으로 평가합니다.

    왜 필요한가 — app.py 앵커링 탭은 `df` 전체로 학습한 뒤 같은 `X` 를
    예측합니다. 앵커로 고른 두 행도 이미 학습에 들어가 있어, 화면 숫자가
    '새 논문에서의 성능'이 아닙니다. 이 함수는 논문을 하나씩 빼고
    학습해 실제로 쓰는 상황을 재현합니다.

    중복 제거된 554행/91편 데이터로 앵커 개수를 비교한 실측 (25편 홀드아웃):
        앵커 없이   MAE 15.84
        k=2         MAE 12.30  (+22%)  개선 16/25편  p = 0.090   ← 불안정
        k=3         MAE 10.90  (+31%)  개선 20/25편  p = 0.0006  ← 기본값
        k=5         MAE 11.21  (+29%)  개선 22/25편  p = 0.0016
    k=2 는 앵커 하나가 이상치일 때 오프셋이 크게 흔들려 유의성을 잃습니다.
    그래서 기본값을 3 으로 둡니다 (실험 1회 추가 비용으로 안정성을 얻음).
    """
    from sklearn.metrics import mean_absolute_error
    from scipy.stats import wilcoxon

    d = dedupe(df, verbose=False)
    y = pd.to_numeric(d[EEC], errors="coerce").values
    ok = np.isfinite(y) & (y > 0) & (y <= 100)
    d, y = d[ok].reset_index(drop=True), y[ok]
    g = d["reference_doi"].astype(str).str.strip().str.lower().values

    with contextlib.redirect_stdout(io.StringIO()):
        X, num_cols, cat_cols = v3_module.build_features(
            d, include_measured=False)

    rows = []
    for gid in pd.unique(g):
        te = (g == gid)
        if te.sum() < min_rows:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            m = anchor_module.AnchoredEEPredictor(
                v3_module, num_cols, cat_cols).fit(X[~te], y[~te])
        Xt = X[te].reset_index(drop=True)
        yt = y[te]
        a = m.suggest_anchors(Xt, k=k)
        rows.append({"paper": gid, "n": int(te.sum()),
                     "mae_no_anchor": mean_absolute_error(yt, m.predict(Xt)),
                     "mae_anchored": mean_absolute_error(
                         yt, m.predict(Xt, a, yt[a]))})

    R = pd.DataFrame(rows)
    if len(R) < 3:
        return R, {"note": f"논문당 {min_rows}행 이상인 논문이 "
                           f"{len(R)}편뿐이라 평가 불가"}
    p = wilcoxon(R.mae_no_anchor, R.mae_anchored).pvalue
    stats = {
        "papers": len(R),
        "mae_no_anchor": float(R.mae_no_anchor.mean()),
        "mae_anchored": float(R.mae_anchored.mean()),
        "gain_pct": float((R.mae_no_anchor.mean() - R.mae_anchored.mean())
                          / R.mae_no_anchor.mean() * 100),
        "improved": int((R.mae_anchored < R.mae_no_anchor).sum()),
        "p_value": float(p),
    }
    return R, stats
