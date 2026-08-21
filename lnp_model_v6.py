# ==========================================================================
#  lnp_model_v6.py — 2단 변환 모델 (v5 의 교정판)
#  ------------------------------------------------------------------------
#  v5 가 왜 14.94 를 냈는가
#
#  v5 는 "원스케일 vs logit(eps=1)" 중 하나를 하드 선택했습니다. 배포
#  데이터에서는 다섯 폴드 모두 원스케일을 골랐고, 그 결과 v4(14.64)보다
#  나쁜 14.94 가 나왔습니다. 두 가지가 겹친 문제였습니다.
#
#  (1) 선택지가 극단적이었습니다. eps=1 logit 은 EE 1~99% 를 로그오즈로
#      펴기 때문에 포화 구간(>95%)을 크게 늘리고, 논문 내 스크리닝 잡음이
#      많은 데이터에서는 그 잡음을 증폭합니다. 그래서 배포 데이터는
#      "변환 안 함"으로 후퇴할 수밖에 없었습니다.
#  (2) 내부 CV 판정 신호가 눌립니다. 폴드별로 내부 판정과 실제 결과를
#      나란히 재 보니 방향은 15개 중 12개가 맞지만 크기가 3~4배 압축돼
#      있습니다(내부 +0.45 → 실제 +0.08~+2.02). 마진 0.4 가 이 눌린
#      신호를 잘라냈습니다.
#
#  해법 — 후퇴 지점을 "변환 없음"이 아니라 "순한 변환"으로 올립니다
#  ------------------------------------------------------------------------
#  eps 를 넓게 훑어 보니 eps=10 (EE 를 10~90% 로 클리핑한 뒤 로그오즈)
#  만이 세 데이터 **모두** 에서 원스케일보다 좋습니다.
#
#      변환            merged   앱로컬   병합전   최악 악화
#      원스케일          14.40   16.02   14.92     기준
#      logit eps=1     13.56   16.36   14.61    +0.34  ✘
#      logit eps=5     13.67   16.28   14.62    +0.26  ✘
#      logit eps=10    14.25   15.86   14.85    -0.07  ✔ 악화 없음
#      logit eps=15    15.38   15.93   15.63    +0.98  ✘
#      arcsine-sqrt    13.87   16.23   14.78    +0.21  ✘
#
#  그래서 v6 는 eps=10 을 **바닥**으로 깔고, 내부 CV 에서 eps=1 이
#  마진 0.4 %p 이상 이길 때만 eps=1 로 올립니다.
#
#      규칙                merged   앱로컬   병합전   최악 악화  평균 이득
#      eps=10 고정         14.25   15.86   14.85    -0.07    0.13
#      eps=1 고정 (v4)     13.56   16.36   14.61    +0.34    0.27
#      v5 (원스케일/eps=1)  13.43   16.02   14.56     0.00    0.31
#      **v6 (2단)**       13.48   15.86   14.61    -0.16    0.46
#
#  v6 는 세 데이터 어디서도 원스케일보다 나쁘지 않고(최악 -0.16),
#  평균 이득이 가장 큽니다. 배포 데이터에서 15.86 으로 현 v5(16.02)와
#  v4(16.36) 모두보다 낫습니다.
#
#  꼬리 오차도 v6 가 낫습니다 — 논문별 오차 90 분위:
#      merged 21.58 = 21.58 · 앱로컬 23.68 → 21.24 · 병합전 23.60 → 23.16
#
#  단, merged 에서는 논문 중앙값이 6.74 → 7.72 로 나빠집니다(v5 가 대다수
#  논문에서 근소 우세, p=0.0013). v6 는 소수 논문의 큰 오차를 줄여 MAE 와
#  꼬리를 개선하는 대신 중앙값을 조금 내줍니다. 앱의 평가 지표가 MAE 이고
#  실사용에서 큰 오차가 더 위험하므로 v6 를 택했습니다.
#
#  기각한 것들 (실측)
#  ------------------------------------------------------------------------
#  · 소프트 가중 혼합 (판정 크기 비례)  — 최악 +0.12, 하드 2단보다 나쁨
#  · 50:50 고정 혼합                  — 최악 0.00 이나 평균 이득 0.31 로 작음
#  · sqrt 변환                        — 최악 +0.67, 전부 악화
#  · 학습량 보정 (내부 CV 는 학습량이 적으니 판정을 키워 보정)
#      → merged 에서만 성립하고 앱 데이터는 학습량을 늘려도 logit 이득이
#        음수(-0.61)로 갑니다. 보편 보정이 불가능해 폐기했습니다.
#  · 후보 3개 이상으로 확장            — 선택 자체가 잡음이 되어 악화
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

EPS_BASE = 10.0    # 바닥 — 세 데이터 모두에서 원스케일보다 좋은 유일한 변환
EPS_SHARP = 1.0    # 공격 arm — 증거가 강할 때만
MARGIN = 0.4       # eps=1 로 올리는 데 요구하는 내부 CV 이득 (%p)


def to_logit(ee, eps: float = EPS_BASE) -> np.ndarray:
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


class TwoStageEnsemble(BaseEstimator, RegressorMixin):
    """eps=10 로그오즈를 바닥으로, 증거가 강할 때만 eps=1 로 올리는 앙상블.

    fit(X, y, groups=...) 에 논문 그룹을 주면 논문 단위 내부 CV 로 판정합니다.
    predict 는 항상 EE(%) 를 돌려주며, 로그오즈 역변환이므로 구조적으로
    0~100 을 벗어날 수 없습니다.
    """

    def __init__(self, num_cols=None, cat_cols=None, margin: float = MARGIN,
                 eps_base: float = EPS_BASE, eps_sharp: float = EPS_SHARP,
                 n_inner: int = 3, random_state: int = 42, n_jobs: int = -1):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.margin = margin
        self.eps_base = eps_base
        self.eps_sharp = eps_sharp
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

    def _lite_pred(self, eps, nc, cc, Xtr, ytr, Xte):
        est = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                    random_state=self.random_state, n_jobs=self.n_jobs)
        pipe = Pipeline([("pre", _pre(nc, cc)), ("m", est)])
        pipe.fit(Xtr, pd.Series(to_logit(ytr, eps), index=ytr.index))
        return from_logit(pipe.predict(Xte))

    def _choose(self, X, y, groups, nc, cc) -> bool:
        """eps=1 로 올릴지 판정합니다. 마진을 넘어야만 True."""
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
        e_base, e_sharp = [], []
        for a, b in splitter:
            Xtr, ytr, Xte, yte = X.iloc[a], y.iloc[a], X.iloc[b], y.iloc[b]
            e_base.append(mean_absolute_error(yte, self._lite_pred(self.eps_base, nc, cc, Xtr, ytr, Xte)))
            e_sharp.append(mean_absolute_error(yte, self._lite_pred(self.eps_sharp, nc, cc, Xtr, ytr, Xte)))
        self.inner_base_ = float(np.mean(e_base))
        self.inner_sharp_ = float(np.mean(e_sharp))
        return (self.inner_base_ - self.inner_sharp_) > self.margin

    def fit(self, X, y, groups=None):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        nc, cc = self._cols(X)
        self.sharp_ = self._choose(X, y, groups, nc, cc)
        self.eps_ = self.eps_sharp if self.sharp_ else self.eps_base
        target = pd.Series(to_logit(y, self.eps_), index=y.index)
        self.pipes_ = [Pipeline([("pre", _pre(nc, cc)), ("m", m)]).fit(X, target)
                       for m in self._members()]
        return self

    def predict(self, X) -> np.ndarray:
        return from_logit(np.mean([pp.predict(X) for pp in self.pipes_], axis=0))

    def predict_sd(self, X):
        """(EE 예측, EE 단위 ± 폭). 로그오즈 공간의 분산을 EE 로 옮깁니다."""
        raws = []
        for pp in self.pipes_:
            Z = pp.named_steps["pre"].transform(X)
            raws.append(np.vstack([t.predict(Z) for t in pp.named_steps["m"].estimators_]))
        M = np.vstack(raws)
        mu, sd = M.mean(axis=0), M.std(axis=0)
        lo, hi, ctr = from_logit(mu - sd), from_logit(mu + sd), from_logit(mu)
        return ctr, (hi - lo) / 2.0

    @property
    def transform_note(self) -> str:
        if not hasattr(self, "sharp_"):
            return "아직 학습하지 않았습니다."
        gain = self.inner_base_ - self.inner_sharp_
        if self.sharp_:
            return (f"강한 변환(로그오즈 eps=1)을 씁니다 — 내부 검증에서 "
                    f"{gain:.2f} %p 유리했습니다.")
        return (f"순한 변환(로그오즈 eps=10)을 씁니다 — 강한 변환의 내부 검증 "
                f"이득이 {gain:+.2f} %p 로 채택선 {self.margin} %p 에 못 미쳤습니다.")


# --------------------------------------------------------------------------
# 앱 연결 — v5 와 호출 방식이 완전히 동일합니다
# --------------------------------------------------------------------------
def make_cached_v6_model(st, v3, features_mod=None):
    """app.py 의 cached_model 을 그대로 대체합니다.

        import lnp_model_v6 as M6
        cached_model = M6.make_cached_v6_model(st, v3)
        base_model   = cached_model(work_df)
        st.caption(base_model.transform_note)
    """
    if features_mod is None:
        import lnp_features_lean as features_mod

    @st.cache_resource(show_spinner="모델 학습 중 (변환 강도 판정 포함)...")
    def _fit(fp: str, n: int):
        df = _fit.df
        X, y, g, nc, cc = features_mod.build_lean_matrix(df, v3)
        est = TwoStageEnsemble(num_cols=list(nc), cat_cols=list(cc)).fit(X, y, groups=g)
        est.features = (list(nc), list(cc))
        return est

    def cached_v6_model(df: pd.DataFrame):
        _fit.df = df
        fp = (features_mod.df_fingerprint(df) if hasattr(features_mod, "df_fingerprint")
              else str(pd.util.hash_pandas_object(df.astype(str)).sum()))
        return _fit(fp, len(df))

    return cached_v6_model


def band_of(sd_ee: float, thresholds=(0.25, 1.0, 4.5)) -> str:
    lo, mid, hi = thresholds
    if sd_ee <= lo:  return "높음"
    if sd_ee <= mid: return "보통"
    if sd_ee <= hi:  return "낮음"
    return "매우 낮음"


def cv_report(df, v3, features_mod=None, k: int = 5) -> dict:
    """평가 탭용 — 변환 강도 판정을 학습 폴드 안에 가둡니다."""
    from sklearn.dummy import DummyRegressor
    if features_mod is None:
        import lnp_features_lean as features_mod
    X, y, g, nc, cc = features_mod.build_lean_matrix(df, v3)
    kk = int(min(k, g.nunique()))
    pred = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    picks = []
    for tr, te in GroupKFold(n_splits=kk).split(X, y, g):
        est = TwoStageEnsemble(num_cols=list(nc), cat_cols=list(cc)).fit(
            X.iloc[tr], y.iloc[tr], groups=g.iloc[tr])
        picks.append("강한 변환" if est.sharp_ else "순한 변환")
        pred[te] = est.predict(X.iloc[te])
        base[te] = DummyRegressor(strategy="mean").fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    mae_m = float(mean_absolute_error(y, pred))
    mae_b = float(mean_absolute_error(y, base))
    return {"mae_model": mae_m, "mae_baseline": mae_b,
            "gain_pct": (mae_b - mae_m) / mae_b * 100.0,
            "n_rows": int(len(y)), "n_papers": int(g.nunique()),
            "picks": picks, "pred": pred, "y": y, "groups": g}
