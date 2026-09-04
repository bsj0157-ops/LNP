# -*- coding: utf-8 -*-
"""lnp_model_v7 — v6의 두 가지 결함을 고친 EE 예측 모델.

v6 대비 변경점 (998행 · 324논문, 논문 단위 GroupKFold(5) 실측):
  1) 손실함수를 평가지표에 맞춤: HistGradientBoostingRegressor(loss="absolute_error")
       MAE 13.869 %p (v6 = RF+ET logit 평균)  ->  13.138 %p (logit 공간)
                                              ->  13.058 %p (원 스케일, 본 모듈 그대로 재현)
       논문 단위 Wilcoxon p < 0.001, 324논문 중 188논문에서 개선.
  2) 타깃 clipping 제거: v6은 logit 변환 전 [eps, 100-eps], eps_base=10 으로 잘라
     EE >= 90 인 행(데이터의 44.1 %)이 전부 90으로 붕괴했다. 원 스케일 학습은
     변환이 없으므로 상한 구분이 살아 있다: v7의 out-of-fold 예측은 최대 96.8 %,
     예측의 41.6 %가 90 % 이상으로 나온다(v6은 구조적으로 90을 넘을 수 없다).

측정 안 된 것 / 하지 말 것:
  - 멤버 추가·스태킹·다중 시드 평균: 멤버 오차 상관 0.91~0.99 로 실익 없음(측정됨).
  - 같은 논문 안에서의 처방 순위: Spearman 중앙값 0.08 (>=5행 논문 21편) — 사실상 없음.
    스크리닝 순위용으로 쓰지 말 것.

불확실성: RF(600)+ET(600) 의 개별 트리 예측 산포(logit 공간)를 EE 스케일 반폭으로 환산.
  v6 중심값 기준 실측 곡선 — 100 % 커버리지 MAE 13.87, 하위 25 % 보류 시 12.24 %p.

호출 규약은 v6과 동일하게 유지한다:
    m = fit_v7(df, v3_module)          # 또는 cached_v7_model(df, v3_module)
    out = predict_v7(m, df_query)      # DataFrame: pred, sd, band
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

EE_COL = "encapsulation_efficiency_percent_std_num"
DOI_COL = "reference_doi"
SEED = 42


def _pre(num_cols, cat_cols):
    parts = []
    if len(num_cols):
        parts.append(("num", Pipeline([("i", SimpleImputer(strategy="median"))]), list(num_cols)))
    if len(cat_cols):
        parts.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                       ("o", OneHotEncoder(handle_unknown="ignore",
                                                           min_frequency=2,
                                                           sparse_output=False))]), list(cat_cols)))
    return ColumnTransformer(parts)


def df_fingerprint_full(df: pd.DataFrame) -> str:
    """전체 컬럼·전체 값을 반영하는 지문. v6의 3컬럼 지문은 캐시 미갱신 원인이었다."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def fit_v7(df: pd.DataFrame, v3_module=None, with_uncertainty: bool = True) -> dict:
    """EE가 있는 행으로 학습. 반환 dict는 predict_v7 에 그대로 넘긴다."""
    import lnp_features_lean as FL

    X, y, g, nc, cc = FL.build_lean_matrix(df, v3_module)
    if len(y) < 30:
        raise ValueError(f"학습 가능한 행이 {len(y)}개뿐입니다 (최소 30).")

    point = Pipeline([("pre", _pre(nc, cc)),
                      ("m", HistGradientBoostingRegressor(
                          loss="absolute_error", max_depth=3, learning_rate=0.05,
                          max_iter=400, min_samples_leaf=20, l2_regularization=5.0,
                          random_state=SEED))]).fit(X, y)

    spread = []
    if with_uncertainty:
        for est in (RandomForestRegressor(n_estimators=600, min_samples_leaf=5, max_features=0.5,
                                          random_state=SEED, n_jobs=-1),
                    ExtraTreesRegressor(n_estimators=600, min_samples_leaf=4, max_features=0.6,
                                        random_state=SEED, n_jobs=-1)):
            spread.append(Pipeline([("pre", _pre(nc, cc)), ("m", est)]).fit(X, y))

    return dict(point=point, spread=spread, num_cols=list(nc), cat_cols=list(cc),
                v3=v3_module, n_rows=int(len(y)), n_papers=int(pd.Series(g).nunique()),
                fingerprint=df_fingerprint_full(df), version="v7")


def _align(model: dict, df_q: pd.DataFrame) -> pd.DataFrame:
    """학습과 **같은** 특징 엔지니어링(v3.build_features)을 질의 행에 적용한다.

    원본 컬럼을 그대로 넘기면 조성 파싱 특징(ionizable/helper/chol/peg …)이
    전부 NaN 이 되어 모든 행이 같은 값으로 예측된다 — v7 초기 구현의 버그였다.
    """
    cols = model["num_cols"] + model["cat_cols"]
    v3 = model.get("v3")
    if v3 is not None and hasattr(v3, "build_features"):
        Xq, _, _ = v3.build_features(df_q.reset_index(drop=True), include_measured=False)
        Xq.index = df_q.index
    else:
        Xq = df_q
    q = pd.DataFrame(index=df_q.index)
    for c in cols:
        q[c] = Xq[c] if c in Xq.columns else np.nan
    return q[cols]


def predict_v7(model: dict, df_q: pd.DataFrame) -> pd.DataFrame:
    """점추정 + 트리 산포 기반 sd + 신뢰 구간대 라벨."""
    Xq = _align(model, df_q)
    pred = np.clip(model["point"].predict(Xq), 0.0, 100.0)

    sd = np.full(len(Xq), np.nan)
    if model["spread"]:
        per_tree = []
        for pp in model["spread"]:
            Z = pp.named_steps["pre"].transform(Xq)
            per_tree.append(np.vstack([t.predict(Z) for t in pp.named_steps["m"].estimators_]))
        sd = np.vstack(per_tree).std(axis=0)

    band = pd.cut(pd.Series(sd, index=df_q.index), bins=[-np.inf, 3.14, 7.14, 11.97, np.inf],
                  labels=["높음", "중상", "중하", "낮음"])
    return pd.DataFrame({"pred": pred, "sd": sd, "band": band.astype(object),
                         "show_value": pd.Series(sd, index=df_q.index) <= 7.14},
                        index=df_q.index)


def cached_v7_model(df: pd.DataFrame, v3_module=None):
    """Streamlit 캐시 래퍼. 데이터프레임을 해시 가능한 인자로 넘겨
    v5/v6의 함수-속성 전달(세션 간 교차 오염)을 피한다."""
    import streamlit as st

    @st.cache_resource(show_spinner="모델 학습 중…")
    def _fit(csv_bytes: bytes, mod_name: str):
        import io
        import importlib
        d = pd.read_csv(io.BytesIO(csv_bytes))
        mod = importlib.import_module(mod_name) if mod_name else None
        return fit_v7(d, mod)

    name = getattr(v3_module, "__name__", "") if v3_module is not None else ""
    return _fit(df.to_csv(index=False).encode(), name)


# ==========================================================================
# v6 호환 계층 — app.py 의 M6 자리에 그대로 꽂을 수 있게 한다
#   import lnp_model_v7 as M7
#   cached_model = M7.make_cached_v7_model(st, v3)     # M6.make_cached_v6_model 대체
#   rep = M7.cv_report(work_df, v3)                    # M6.cv_report 대체
# ==========================================================================
UNCERTAINTY_SCALE_V7 = 1.95   # 트리 SD -> CV MAE 환산. 1107행/374논문 실측(cv_report scale_hat)


class V7Model:
    """TwoStageEnsemble(v6) 과 같은 자리에서 쓰이는 래퍼.

    노출하는 것과 이유
      predict(X)      : 최적화·What-If 탭이 lean 특징행렬을 그대로 넘긴다.
      predict_sd(X)   : RF+ET 트리 산포(EE 스케일). lnp_optimize 의 상수 fallback 대신 쓸 것.
      transform_note  : app.py 가 st.caption 으로 띄운다.
      named_steps     : lnp_optimize.predict_with_uncertainty 가 마지막 스텝의
                        estimators_ 로 트리 산포를 뽑는다. 점추정은 HistGB(트리 산포 없음)
                        이므로 **불확실성 전용** RF 파이프라인을 여기에 노출한다.
                        점추정은 predict() 가 담당하므로 값이 섞이지는 않는다.
    """

    def __init__(self, model: dict):
        self.m = model
        self.features = (model["num_cols"], model["cat_cols"])
        self.transform_note = (
            f"v7 — 원 스케일 HistGB(absolute_error) · 학습 {model['n_rows']}행 / "
            f"{model['n_papers']}논문 · 타깃 변환·clipping 없음(EE 90% 이상 예측 가능)")

    @property
    def named_steps(self):
        return self.m["spread"][0].named_steps if self.m["spread"] else self.m["point"].named_steps

    def _matrix(self, X):
        cols = self.m["num_cols"] + self.m["cat_cols"]
        if isinstance(X, pd.DataFrame) and all(c in X.columns for c in cols):
            return X[cols]            # 이미 lean 행렬 — 다시 엔지니어링하면 안 된다
        return _align(self.m, X)      # 원본 컬럼 — v3.build_features 를 거친다

    def predict(self, X):
        return np.clip(self.m["point"].predict(self._matrix(X)), 0.0, 100.0)

    def predict_sd(self, X):
        Xq = self._matrix(X)
        if not self.m["spread"]:
            return np.full(len(Xq), np.nan)
        per_tree = []
        for pp in self.m["spread"]:
            Z = pp.named_steps["pre"].transform(Xq)
            per_tree.append(np.vstack([t.predict(Z) for t in pp.named_steps["m"].estimators_]))
        return np.vstack(per_tree).std(axis=0)


def make_cached_v7_model(st, v3, features_mod=None):
    """M6.make_cached_v6_model 과 호출 규약이 같다 (반환값은 V7Model).

    v6와 다른 점: 데이터프레임을 해시 가능한 인자로 캐시에 넘긴다.
    v5/v6은 `_fit.df = df` 로 함수 속성에 실어 보내서, 지문이 같으면
    다른 세션의 데이터로 학습된 모델이 그대로 서빙될 수 있었다.
    """
    @st.cache_resource(show_spinner="모델 학습 중 (v7)…")
    def _fit(csv_bytes: bytes, mod_name: str):
        import importlib
        import io
        d = pd.read_csv(io.BytesIO(csv_bytes))
        mod = importlib.import_module(mod_name) if mod_name else None
        return V7Model(fit_v7(d, mod))

    mod_name = getattr(v3, "__name__", "") if v3 is not None else ""

    def cached(df: pd.DataFrame):
        return _fit(df.to_csv(index=False).encode(), mod_name)

    return cached


def cv_report(df, v3, features_mod=None, k: int = 5) -> dict:
    """M6.cv_report 와 같은 키를 돌려준다 (탭 4 를 그대로 쓸 수 있게).

    추가 키: sd(트리 산포), scale_hat(트리 SD -> MAE 환산 계수 실측값).
    v6과 달리 폴드별 '변환 강도' 선택이 없으므로 picks 는 고정 문구다.
    """
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import GroupKFold
    if features_mod is None:
        import lnp_features_lean as features_mod

    X, y, g, nc, cc = features_mod.build_lean_matrix(df, v3)
    kk = int(min(k, pd.Series(g).nunique()))
    pred = np.full(len(y), np.nan)
    base = np.full(len(y), np.nan)
    sd = np.full(len(y), np.nan)

    for tr, te in GroupKFold(n_splits=kk).split(X, y, g):
        point = Pipeline([("pre", _pre(nc, cc)),
                          ("m", HistGradientBoostingRegressor(
                              loss="absolute_error", max_depth=3, learning_rate=0.05,
                              max_iter=400, min_samples_leaf=20, l2_regularization=5.0,
                              random_state=SEED))]).fit(X.iloc[tr], y.iloc[tr])
        pred[te] = np.clip(point.predict(X.iloc[te]), 0.0, 100.0)
        base[te] = DummyRegressor(strategy="median").fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        rf = Pipeline([("pre", _pre(nc, cc)),
                       ("m", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                   max_features=0.5, random_state=SEED,
                                                   n_jobs=-1))]).fit(X.iloc[tr], y.iloc[tr])
        Z = rf.named_steps["pre"].transform(X.iloc[te])
        sd[te] = np.vstack([t.predict(Z) for t in rf.named_steps["m"].estimators_]).std(axis=0)

    mae_m = float(np.mean(np.abs(pred - y.values)))
    mae_b = float(np.mean(np.abs(base - y.values)))
    return {"mae_model": mae_m, "mae_baseline": mae_b,
            "gain_pct": (mae_b - mae_m) / mae_b * 100.0,
            "n_rows": int(len(y)), "n_papers": int(pd.Series(g).nunique()),
            "picks": ["원 스케일 (변환 없음)"] * kk,
            "pred": pred, "y": y, "groups": g, "sd": sd,
            "scale_hat": float(mae_m / np.nanmean(sd))}
