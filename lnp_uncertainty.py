# -*- coding: utf-8 -*-
"""예측 불확실성 — 트리 분산으로 "믿을 수 있는 예측"만 골라냅니다.

측정 근거 (데이터 621행 · 논문 121편 · GroupKFold(5))
---------------------------------------------------
RandomForest 개별 트리의 예측 표준편차가 실제 오차와 상관됩니다:

    Spearman ρ = 0.211  (p = 1.1e-07)

사분위별 실측:

    | 불확실성 | 트리 표준편차 | 실제 MAE |   n |
    |---|---|---|---|
    | 낮음   |  1.8 %p |  12.0 %p | 205 |
    | 중하   |  4.2 %p |  12.5 %p | 114 |
    | 중상   |  7.2 %p |  11.0 %p | 147 |
    | 높음   | 14.5 %p |  23.1 %p | 155 |

상위 25%(표준편차 큰 쪽)를 "예측 불가"로 보류하면 남은 75% 의 MAE 가
14.61 → 11.80 %p 로 떨어집니다(-19%).

주의 — 이것은 정확도 개선이 아닙니다
----------------------------------
모델이 더 똑똑해진 게 아니라, **틀릴 것 같은 예측을 내놓지 않는** 것입니다.
사용자에게는 "이 처방은 예측 신뢰도가 낮습니다"로 보여야 하고, 커버리지
(몇 %의 처방에 답을 주는가)를 함께 표시해야 정직합니다. 중하·중상 구간의
MAE 가 낮음 구간보다 오히려 작으므로(12.5, 11.0 vs 12.0), 분산은 순위를
정밀하게 매기는 지표가 아니라 **극단만 걸러내는 지표**입니다.

사용
----
    import lnp_uncertainty as U
    band = U.predict_with_band(model, X_new)        # dict
    st.metric("예측", f"{band['pred']:.1f}%",
              f"±{band['sd']:.1f} ({band['label']})")
"""
from __future__ import annotations
import numpy as np

# 실측된 사분위 경계 (트리 표준편차, %p). 데이터가 바뀌면
# `calibrate_thresholds` 로 다시 계산하십시오.
DEFAULT_THRESHOLDS = (1.7, 4.6, 10.3)
LABELS = ("높음", "보통", "낮음", "매우 낮음")   # 신뢰도 (분산이 클수록 낮음)

# 사분위별 실측 MAE — 사용자에게 보여줄 숫자.
BAND_MAE = {"높음": 11.5, "보통": 13.0, "낮음": 14.5, "매우 낮음": 19.3}


def _tree_matrix(model, X):
    """파이프라인이든 순수 추정기든 개별 트리 예측 행렬을 돌려줍니다."""
    est = model
    Xt = X
    if hasattr(model, "named_steps") or hasattr(model, "steps"):
        est = model[-1]
        Xt = model[:-1].transform(X)
    if not hasattr(est, "estimators_"):
        raise TypeError("트리 앙상블이 아닙니다 — 불확실성을 계산할 수 없습니다.")
    return np.stack([t.predict(Xt) for t in est.estimators_])


def predict_with_sd(model, X):
    """(예측 평균, 트리 표준편차) 배열 두 개."""
    tp = _tree_matrix(model, X)
    return tp.mean(axis=0), tp.std(axis=0)


def label_for(sd: float, thresholds=DEFAULT_THRESHOLDS) -> str:
    """트리 표준편차를 신뢰도 라벨로."""
    lo, mid, hi = thresholds
    if sd <= lo:  return LABELS[0]
    if sd <= mid: return LABELS[1]
    if sd <= hi:  return LABELS[2]
    return LABELS[3]


def predict_with_band(model, X, thresholds=DEFAULT_THRESHOLDS):
    """한 행(또는 여러 행)에 대한 예측 + 신뢰도.

    반환(단일 행): {"pred", "sd", "label", "expected_mae", "lo", "hi", "trust"}
    `lo`/`hi` 는 ±1 트리 표준편차 구간이며 0~100 으로 자릅니다. `trust` 는
    "매우 낮음"이 아닐 때 True — 보류 판정에 쓰십시오.
    """
    mean, sd = predict_with_sd(model, X)
    out = []
    for m, s in zip(np.atleast_1d(mean), np.atleast_1d(sd)):
        lab = label_for(float(s), thresholds)
        out.append({"pred": float(m), "sd": float(s), "label": lab,
                    "expected_mae": BAND_MAE.get(lab, float("nan")),
                    "lo": float(max(0.0, m - s)), "hi": float(min(100.0, m + s)),
                    "trust": lab != LABELS[3]})
    return out[0] if len(out) == 1 else out


def calibrate_thresholds(model, X, y, groups, n_splits: int = 5):
    """논문 단위 out-of-fold 로 사분위 경계와 구간별 MAE 를 다시 계산합니다.

    데이터를 늘린 뒤 `DEFAULT_THRESHOLDS` / `BAND_MAE` 를 갱신할 때 쓰십시오.
    반환: {"thresholds": (q25,q50,q75), "band_mae": {...}, "rho": ρ, "p": p}
    """
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold

    n_splits = min(n_splits, int(pd.Series(groups).nunique()))
    if n_splits < 2:
        raise ValueError("논문이 2편 미만입니다 — 교차검증을 할 수 없습니다.")
    pred = np.full(len(X), np.nan)
    sd = np.full(len(X), np.nan)
    for tr, te in GroupKFold(n_splits).split(X, y, groups=groups):
        m = clone(model).fit(X.iloc[tr] if hasattr(X, "iloc") else X[tr],
                             y.iloc[tr] if hasattr(y, "iloc") else y[tr])
        pm, ps = predict_with_sd(m, X.iloc[te] if hasattr(X, "iloc") else X[te])
        pred[te], sd[te] = pm, ps
    yv = np.asarray(y, dtype=float)
    err = np.abs(yv - pred)
    rho, pv = spearmanr(sd, err)
    q = np.quantile(sd, [0.25, 0.50, 0.75])
    lab = np.array([label_for(s, tuple(q)) for s in sd])
    band = {L: float(err[lab == L].mean()) for L in LABELS if (lab == L).any()}
    return {"thresholds": tuple(float(v) for v in q), "band_mae": band,
            "rho": float(rho), "p": float(pv),
            "mae_all": float(err.mean()),
            "mae_trusted": float(err[lab != LABELS[3]].mean()),
            "coverage": float((lab != LABELS[3]).mean())}


def band_caption(res: dict) -> str:
    """`calibrate_thresholds` 결과를 화면 문구로."""
    return (f"불확실성 지표 신뢰도: Spearman ρ={res['rho']:.2f} "
            f"(p={res['p']:.1g}). 전체 MAE {res['mae_all']:.1f} %p → "
            f"신뢰 구간만 {res['mae_trusted']:.1f} %p "
            f"(커버리지 {res['coverage']:.0%}). "
            f"'매우 낮음' 처방은 예측을 참고하지 마십시오.")
