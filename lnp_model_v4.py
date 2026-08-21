# ==========================================================================
#  lnp_model_v4.py — logit 타깃 + RF/ExtraTrees 앙상블
#  ------------------------------------------------------------------------
#  왜 이 조합인가 (801행 / 214편, reference_doi 기준 GroupKFold(5) 실측)
#
#    현재 운영 모델 (RF, 현재 특징 24개)          14.97 %p
#    + SMILES 기술자 제거 (lnp_features_lean)     14.53 %p
#    + logit 타깃 변환                            13.75 %p   ← 가장 큰 이득
#    + ExtraTrees 와 평균                         13.56 %p
#    baseline (평균만 답하기)                     16.12 %p
#
#  logit 변환이 듣는 이유: EE 는 0~100% 로 경계가 있고 데이터가 85~95% 에
#  몰려 있습니다. 원래 스케일에서 트리는 100% 를 넘는 예측을 하고 경계
#  부근의 오차를 과소평가합니다. logit 공간에서는 경계가 무한대로 밀려나
#  포화 구간의 미세한 차이가 펴집니다. 214편 중 142편에서 개선되며
#  Wilcoxon p < 1e-5 입니다.
#
#  검증했으나 채택하지 않은 것 (같은 조건 실측):
#    XGBoost 기본             15.41   과적합 — 논문 214편에 트리 부스팅은 과함
#    XGBoost 강한 규제        14.57   RF 와 동급, p=0.28 (유의차 없음)
#    LightGBM                 14.80   RF 보다 나쁨
#    MLP (2층)                15.94   baseline 수준
#    MLP + Morgan 지문        17.23   지질 174종에 234비트는 차원의 저주
#    RF + Morgan 지문         14.72   지문이 오히려 방해
#    SVEM (Lasso 재표본 60회) 15.35   문헌 데이터에서는 RF 보다 나쁨 (p=0.001)
#
#  트리 모델들의 오차 상관이 0.93~0.99 입니다 — 서로 거의 같은 예측을
#  하므로 종류를 늘려도 앙상블 이득이 거의 없습니다. RF+ExtraTrees 두
#  개까지가 실익이고 (13.56), 다섯 개로 늘리면 오히려 13.65 입니다.
# ==========================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1.0          # logit 경계 여유 (%). EE 0/100 을 ±1% 로 클립합니다.


def to_logit(ee, eps: float = EPS) -> np.ndarray:
    """EE(%) → logit. 0/100 을 eps 로 클립해 무한대를 막습니다."""
    v = np.clip(np.asarray(ee, dtype=float), eps, 100.0 - eps)
    return np.log(v / (100.0 - v))


def from_logit(z) -> np.ndarray:
    """logit → EE(%). 항상 0~100 안에 들어옵니다."""
    return 100.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _pre(num_cols, cat_cols) -> ColumnTransformer:
    parts = [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler())]), list(num_cols))]
    if len(cat_cols):
        parts.append(("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                     ("o", OneHotEncoder(handle_unknown="ignore",
                                                         min_frequency=2))]), list(cat_cols)))
    return ColumnTransformer(parts)


class LogitEnsemble(BaseEstimator, RegressorMixin):
    """logit 공간에서 RF 와 ExtraTrees 를 학습해 평균하는 회귀기.

    fit 은 EE(%) 를 그대로 받습니다 — 내부에서 logit 으로 바꿉니다.
    predict 는 EE(%) 를 돌려주며 구조적으로 0~100 을 벗어나지 않습니다.
    predict_sd 는 두 앙상블의 개별 트리 예측을 모아 EE 단위 폭을 줍니다.
    """

    def __init__(self, num_cols=None, cat_cols=None, random_state: int = 42, n_jobs: int = -1):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _members(self):
        return [
            RandomForestRegressor(n_estimators=600, min_samples_leaf=5, max_features=0.5,
                                  random_state=self.random_state, n_jobs=self.n_jobs),
            ExtraTreesRegressor(n_estimators=600, min_samples_leaf=4, max_features=0.6,
                                random_state=self.random_state, n_jobs=self.n_jobs),
        ]

    def fit(self, X, y):
        nc = self.num_cols if self.num_cols is not None else list(X.columns)
        cc = self.cat_cols if self.cat_cols is not None else []
        z = to_logit(y)
        self.pipes_ = [Pipeline([("pre", _pre(nc, cc)), ("m", m)]).fit(X, z)
                       for m in self._members()]
        return self

    def predict(self, X) -> np.ndarray:
        z = np.mean([p.predict(X) for p in self.pipes_], axis=0)
        return from_logit(z)

    def predict_sd(self, X):
        """(EE 예측, EE 단위 ± 폭). 폭은 logit 공간 표준편차를 EE 로 환산한 것입니다."""
        zs = []
        for p in self.pipes_:
            Z = p.named_steps["pre"].transform(X)
            zs.append(np.vstack([t.predict(Z) for t in p.named_steps["m"].estimators_]))
        M = np.vstack(zs)
        mu, sd = M.mean(axis=0), M.std(axis=0)
        lo, hi = from_logit(mu - sd), from_logit(mu + sd)
        return from_logit(mu), (hi - lo) / 2.0


# --------------------------------------------------------------------------
# 앱 연결 — cached_model 을 그대로 대체합니다
# --------------------------------------------------------------------------
def make_cached_v4_model(st, v3, features_mod=None):
    """lnp_features_lean 의 축소 특징 + LogitEnsemble 을 캐시해 돌려줍니다.

        cached_model = M4.make_cached_v4_model(st, v3)
        base_model   = cached_model(work_df)     # 호출부 변경 없음
    """
    if features_mod is None:
        import lnp_features_lean as features_mod

    @st.cache_resource(show_spinner="모델 학습 중...")
    def _fit(fp: str, n: int):
        df = _fit.df
        X, y, _g, nc, cc = features_mod.build_lean_matrix(df, v3)
        est = LogitEnsemble(num_cols=nc, cat_cols=cc).fit(X, y)
        est.features = (list(nc), list(cc))
        return est

    def cached_v4_model(df: pd.DataFrame):
        _fit.df = df
        fp = features_mod.df_fingerprint(df) if hasattr(features_mod, "df_fingerprint") \
             else str(pd.util.hash_pandas_object(df.astype(str)).sum())
        return _fit(fp, len(df))

    return cached_v4_model


def band_of(sd_ee: float, thresholds=(0.25, 1.0, 4.5)) -> str:
    """EE 단위 폭 → 신뢰도 라벨. 경계는 801행/214편 사분위 실측값입니다."""
    lo, mid, hi = thresholds
    if sd_ee <= lo:  return "높음"
    if sd_ee <= mid: return "보통"
    if sd_ee <= hi:  return "낮음"
    return "매우 낮음"


# 801행/214편 out-of-fold 실측 — 밴드별 실제 MAE (%p)
BAND_MAE_V4 = {"높음": 10.67, "보통": 12.43, "낮음": 14.55, "매우 낮음": 16.53}
