# ==========================================================================
#  lnp_model_v5.py — 데이터가 변환을 직접 고르는 모델 (v4 의 교정판)
#  ------------------------------------------------------------------------
#  v4 의 무엇이 틀렸는가
#
#  v4 는 logit 타깃 변환을 무조건 적용했습니다. 그 근거는 merged 801행/214편
#  에서 −0.77 %p 였습니다. 그런데 앱이 실제로 들고 있는 데이터(680행/92편)
#  에서는 **+0.36 %p 로 부호가 뒤집힙니다.**
#
#      데이터              RF 원스케일   RF logit    logit 효과
#      merged 801/214       14.53      13.75       −0.77   ✔
#      앱 로컬 680/92        15.89      16.25       +0.36   ✘ 악화
#      병합전 636/121        14.87      14.69       −0.18   ✔ (미미)
#
#  왜 갈리는가: 앱 데이터는 논문당 7.4행이고 merged 는 3.7행입니다. 논문당
#  행이 많다는 것은 한 논문 안의 스크리닝 행이 많다는 뜻이고, 그 행들은
#  같은 조성에서도 EE 가 수십 %p 씩 흔들립니다(앞서 확인한 잡음 하한).
#  잡음이 지배하는 데이터에서는 어떤 정교화도 이득을 내지 못하며, logit 은
#  포화 구간을 늘려 놓기 때문에 그 잡음을 오히려 증폭합니다.
#
#  교훈: "변환을 쓸지" 를 사람이 정하면 안 됩니다. 데이터가 정해야 합니다.
#
#  v5 가 하는 일
#  ------------------------------------------------------------------------
#  학습 폴드 안에서만 내부 3겹 논문 단위 CV 를 돌려 원스케일과 logit 을
#  비교하고, **logit 이 마진 0.4 %p 이상 이길 때만** 채택합니다. 마진이
#  없으면 선택 잡음 때문에 오히려 나빠집니다 (앱 데이터에서 중첩 CV 가
#  +0.42 였습니다). 마진 0.4 는 세 데이터 모두에서 원스케일 이하입니다:
#
#      데이터              원스케일   고정 logit   v5 (마진 0.4)
#      merged 801/214      14.53     13.75       13.62   ← 최선
#      앱 680/92            15.89     16.25       15.89   ← 악화 없음
#      병합전 636/121        14.87     14.69       14.44   ← 최선
#
#  즉 v5 는 어느 데이터에서도 현재 모델보다 나빠지지 않으면서, 변환이
#  듣는 데이터에서는 그 이득을 가져갑니다. 데이터가 바뀌어도(앱은 계속
#  행이 추가됩니다) 다시 판정하므로 손댈 필요가 없습니다.
#
#  마진 0.8 은 과하게 보수적이어서 merged 에서 14.03 으로 이득을 놓칩니다.
# ==========================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

EPS = 1.0
MARGIN = 0.4       # logit 채택에 요구하는 최소 이득 (%p). 실측 근거는 헤더 표.


def to_logit(ee, eps: float = EPS) -> np.ndarray:
    v = np.clip(np.asarray(ee, dtype=float), eps, 100.0 - eps)
    return np.log(v / (100.0 - v))


def from_logit(z) -> np.ndarray:
    return 100.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _pre(num_cols, cat_cols) -> ColumnTransformer:
    parts = [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler())]), list(num_cols))]
    if len(cat_cols):
        parts.append(("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                     ("o", OneHotEncoder(handle_unknown="ignore",
                                                         min_frequency=2))]), list(cat_cols)))
    return ColumnTransformer(parts)


class AdaptiveEnsemble(BaseEstimator, RegressorMixin):
    """원스케일과 logit 중 데이터가 이기는 쪽을 골라 쓰는 RF/ExtraTrees 앙상블.

    fit(X, y, groups=...) 에 논문 그룹을 주면 논문 단위 내부 CV 로 고릅니다.
    groups 가 없으면 무작위 내부 CV 로 떨어지는데, 이 경우 선택이 낙관적으로
    치우칩니다 — 가능하면 항상 groups 를 주십시오.
    predict 는 항상 EE(%) 를 돌려주고 0~100 을 벗어나지 않습니다.
    """

    def __init__(self, num_cols=None, cat_cols=None, margin: float = MARGIN,
                 n_inner: int = 3, random_state: int = 42, n_jobs: int = -1):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.margin = margin
        self.n_inner = n_inner
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _members(self):
        return [
            RandomForestRegressor(n_estimators=600, min_samples_leaf=5, max_features=0.5,
                                  random_state=self.random_state, n_jobs=self.n_jobs),
            ExtraTreesRegressor(n_estimators=600, min_samples_leaf=4, max_features=0.6,
                                random_state=self.random_state, n_jobs=self.n_jobs),
        ]

    def _cols(self, X):
        nc = self.num_cols if self.num_cols is not None else list(X.columns)
        cc = self.cat_cols if self.cat_cols is not None else []
        return nc, cc

    def _fit_pred(self, use_logit, nc, cc, Xtr, ytr, Xte, light=True):
        est = (RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                     random_state=self.random_state, n_jobs=self.n_jobs)
               if light else self._members()[0])
        pipe = Pipeline([("pre", _pre(nc, cc)), ("m", est)])
        if use_logit:
            pipe.fit(Xtr, to_logit(ytr))
            return from_logit(pipe.predict(Xte))
        pipe.fit(Xtr, ytr)
        return np.clip(pipe.predict(Xte), 0.0, 100.0)

    def _choose(self, X, y, groups, nc, cc) -> bool:
        """내부 CV 로 logit 채택 여부를 정합니다. 마진을 넘어야만 True."""
        if groups is not None:
            g = pd.Series(np.asarray(groups), index=y.index)
            k = int(min(self.n_inner, g.nunique()))
            if k < 2:
                return False
            splitter = GroupKFold(n_splits=k).split(X, y, g)
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=min(self.n_inner, len(y)),
                             shuffle=True, random_state=self.random_state).split(X)
        e_orig, e_logit = [], []
        for itr, ite in splitter:
            Xtr, ytr, Xte, yte = X.iloc[itr], y.iloc[itr], X.iloc[ite], y.iloc[ite]
            e_orig.append(mean_absolute_error(yte, self._fit_pred(False, nc, cc, Xtr, ytr, Xte)))
            e_logit.append(mean_absolute_error(yte, self._fit_pred(True, nc, cc, Xtr, ytr, Xte)))
        self.inner_orig_ = float(np.mean(e_orig))
        self.inner_logit_ = float(np.mean(e_logit))
        return (self.inner_orig_ - self.inner_logit_) > self.margin

    def fit(self, X, y, groups=None):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        nc, cc = self._cols(X)
        self.use_logit_ = self._choose(X, y, groups, nc, cc)
        target = pd.Series(to_logit(y), index=y.index) if self.use_logit_ else y
        self.pipes_ = [Pipeline([("pre", _pre(nc, cc)), ("m", m)]).fit(X, target)
                       for m in self._members()]
        return self

    def predict(self, X) -> np.ndarray:
        p = np.mean([pp.predict(X) for pp in self.pipes_], axis=0)
        return from_logit(p) if self.use_logit_ else np.clip(p, 0.0, 100.0)

    def predict_sd(self, X):
        """(EE 예측, EE 단위 ± 폭)."""
        raws = []
        for pp in self.pipes_:
            Z = pp.named_steps["pre"].transform(X)
            raws.append(np.vstack([t.predict(Z) for t in pp.named_steps["m"].estimators_]))
        M = np.vstack(raws)
        mu, sd = M.mean(axis=0), M.std(axis=0)
        if self.use_logit_:
            lo, hi, ctr = from_logit(mu - sd), from_logit(mu + sd), from_logit(mu)
        else:
            lo, hi, ctr = mu - sd, mu + sd, np.clip(mu, 0.0, 100.0)
        return ctr, (hi - lo) / 2.0

    @property
    def transform_note(self) -> str:
        """화면에 그대로 띄울 수 있는 한 줄 설명."""
        if not hasattr(self, "use_logit_"):
            return "아직 학습하지 않았습니다."
        gain = self.inner_orig_ - self.inner_logit_
        if self.use_logit_:
            return (f"이 데이터에서는 logit 변환이 유리해 적용했습니다 "
                    f"(내부 검증 이득 {gain:.2f} %p).")
        return (f"이 데이터에서는 logit 변환이 도움이 되지 않아 원 스케일을 "
                f"씁니다 (내부 검증 이득 {gain:+.2f} %p — 채택선 {self.margin} %p 미달).")


# --------------------------------------------------------------------------
# 앱 연결
# --------------------------------------------------------------------------
def make_cached_v5_model(st, v3, features_mod=None):
    """app.py 의 cached_model 을 그대로 대체합니다.

        cached_model = M5.make_cached_v5_model(st, v3)
        base_model   = cached_model(work_df)
        st.caption(base_model.transform_note)   # 어떤 스케일을 골랐는지 표시
    """
    if features_mod is None:
        import lnp_features_lean as features_mod

    @st.cache_resource(show_spinner="모델 학습 중 (변환 판정 포함)...")
    def _fit(fp: str, n: int):
        df = _fit.df
        X, y, g, nc, cc = features_mod.build_lean_matrix(df, v3)
        est = AdaptiveEnsemble(num_cols=list(nc), cat_cols=list(cc)).fit(X, y, groups=g)
        est.features = (list(nc), list(cc))
        return est

    def cached_v5_model(df: pd.DataFrame):
        _fit.df = df
        fp = (features_mod.df_fingerprint(df) if hasattr(features_mod, "df_fingerprint")
              else str(pd.util.hash_pandas_object(df.astype(str)).sum()))
        return _fit(fp, len(df))

    return cached_v5_model


def band_of(sd_ee: float, thresholds=(0.25, 1.0, 4.5)) -> str:
    lo, mid, hi = thresholds
    if sd_ee <= lo:  return "높음"
    if sd_ee <= mid: return "보통"
    if sd_ee <= hi:  return "낮음"
    return "매우 낮음"


def cv_report(df, v3, features_mod=None, k: int = 5) -> dict:
    """평가 탭이 쓸 정직한 성능 — 변환 선택을 학습 폴드 안에서만 합니다.

    고정 변환으로 전체 CV 를 돌리면 변환 선택에 평가 폴드가 쓰여
    성능이 낙관적으로 나옵니다. 이 함수는 그 누출을 막습니다.
    """
    from sklearn.dummy import DummyRegressor
    if features_mod is None:
        import lnp_features_lean as features_mod
    X, y, g, nc, cc = features_mod.build_lean_matrix(df, v3)
    kk = int(min(k, g.nunique()))
    pred = np.full(len(y), np.nan)
    picks = []
    for tr, te in GroupKFold(n_splits=kk).split(X, y, g):
        est = AdaptiveEnsemble(num_cols=list(nc), cat_cols=list(cc)).fit(
            X.iloc[tr], y.iloc[tr], groups=g.iloc[tr])
        picks.append("logit" if est.use_logit_ else "원스케일")
        pred[te] = est.predict(X.iloc[te])
    base = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=kk).split(X, y, g):
        base[te] = DummyRegressor(strategy="mean").fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    mae_m = float(mean_absolute_error(y, pred))
    mae_b = float(mean_absolute_error(y, base))
    return {"mae_model": mae_m, "mae_baseline": mae_b,
            "gain_pct": (mae_b - mae_m) / mae_b * 100.0,
            "n_rows": int(len(y)), "n_papers": int(g.nunique()),
            "picks": picks, "pred": pred, "y": y, "groups": g}
