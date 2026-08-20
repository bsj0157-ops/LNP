# -*- coding: utf-8 -*-
"""특징 축소 — SMILES 기술자를 빼면 정확도가 올라갑니다.

측정 근거 (데이터 621행 · 논문 121편 · GroupKFold(5) by reference_doi)
-------------------------------------------------------------------
    현재 전체 특징 (24개)          MAE 15.52 %p
    SMILES 기술자 14개 제거        MAE 14.71 %p   ← -0.81
    + min_samples_leaf=5           MAE 14.61 %p   ← -0.91 (-5.9%)

왜 빼는 게 낫는가
----------------
SMILES 기술자(MolWt·MolLogP·TPSA·NumHDonors·… 14개)는 이온화지질 한 종에
대해 **완전히 상관된 상수 벡터**입니다. 지질이 50종뿐이고 상위 8종이 전체의
80%를 차지하므로, 이 14개 컬럼은 사실상 "지질 종류"를 14차원으로 중복
표현한 것입니다. RandomForest 는 분할 후보가 많아질수록 이런 중복 축을
고르기 쉬워지고, 논문 단위 분할에서는 학습에 없던 지질을 만나면 그 축이
전부 무의미해집니다 — 잡음만 늘어납니다.

조성 비율(7개)과 카테고리(3개)는 제거하면 나빠집니다:
    조성 비율 제거   MAE 15.89 (+0.37)
    카테고리 제거    MAE 16.14 (+0.62)
→ 이 둘은 유지합니다.

사용
----
    import lnp_features_lean as FL
    X, y, g, nc, cc = FL.build_lean_matrix(work_df, v3)
    model = FL.make_lean_model().fit(X, y)

앱에 적용하려면 `make_cached_lean_model(st, v3)` 로 캐시된 것을 쓰십시오.
"""
from __future__ import annotations

# build_eval_matrix 가 만드는 조성 특징 — 유지합니다.
COMP_COLS = ["ionizable", "helper", "chol", "peg",
             "ion_to_helper", "ion_plus_chol", "log_peg"]

# SMILES 기술자 — 제거 대상.
SMILES_COLS = ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
               "NumRotatableBonds", "RingCount", "FractionCSP3",
               "HeavyAtomCount", "n_tert_amine", "n_ester", "n_amide",
               "n_N_total", "n_c8_chain"]


def build_lean_matrix(work_df, v3_module, drop_smiles: bool = True):
    """`build_eval_matrix` 결과에서 SMILES 기술자를 뺀 행렬을 돌려줍니다.

    반환은 원본과 같은 (X, y, groups, num_cols, cat_cols) 5-튜플이므로
    기존 호출부를 그대로 대체할 수 있습니다.
    """
    import lnp_app_fix2 as F2
    X, y, groups, num_cols, cat_cols = F2.build_eval_matrix(work_df, v3_module)
    if not drop_smiles:
        return X, y, groups, list(num_cols), list(cat_cols)
    keep = [c for c in num_cols if c not in SMILES_COLS]
    if not keep:                      # 방어: 조성 특징이 없으면 원본 유지
        return X, y, groups, list(num_cols), list(cat_cols)
    return X, y, groups, keep, list(cat_cols)


def make_lean_model(min_samples_leaf: int = 5, n_estimators: int = 600,
                    num_cols=None, cat_cols=None, random_state: int = 42):
    """축소 특징에 맞춘 파이프라인. `min_samples_leaf=5` 가 3보다 나았습니다
    (14.69 → 14.61 %p) — 논문 수가 121편뿐이라 잎이 클수록 안정적입니다."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    steps = [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler())]), num_cols or COMP_COLS)]
    if cat_cols:
        steps.append(("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                     ("o", OneHotEncoder(handle_unknown="ignore",
                                                         min_frequency=3))]), cat_cols))
    return Pipeline([("pre", ColumnTransformer(steps)),
                     ("m", RandomForestRegressor(
                         n_estimators=n_estimators,
                         min_samples_leaf=min_samples_leaf,
                         max_features=0.5, random_state=random_state, n_jobs=-1))])


def make_cached_lean_model(st, v3_module):
    """앱용. `F2.make_cached_base_model` 과 같은 인터페이스입니다.

        cached_model = FL.make_cached_lean_model(st, v3)
        base_model   = cached_model(work_df)      # 그대로 대체 가능
    """
    import lnp_app_fix2 as F2

    @st.cache_data(show_spinner="특징 계산 중...")
    def _feats(fingerprint, _df):
        X, y, g, nc, cc = build_lean_matrix(_df, v3_module)
        return X, y, g, tuple(nc), tuple(cc)

    @st.cache_resource(show_spinner="모델 학습 중... (데이터가 바뀔 때만 실행됩니다)")
    def _fit(fingerprint, num_cols, cat_cols, _X, _y):
        return make_lean_model(num_cols=list(num_cols),
                               cat_cols=list(cat_cols)).fit(_X, _y)

    def cached_lean_model(work_df):
        fp = F2.df_fingerprint(work_df)
        X, y, _, nc, cc = _feats(fp, work_df)
        return _fit(fp, nc, cc, X, y)

    def cached_features(work_df):
        return _feats(F2.df_fingerprint(work_df), work_df)

    cached_lean_model.features = cached_features
    return cached_lean_model
