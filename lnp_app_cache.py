# -*- coding: utf-8 -*-
"""앱 캐시 계층 — 위젯을 조작할 때마다 무거운 연산을 다시 돌지 않게 합니다.

현재 앱에는 `st.cache_data` 가 한 개도 없습니다. Streamlit 은 위젯을 하나
건드릴 때마다 스크립트를 처음부터 다시 실행하므로, 앵커를 바꿀 때마다
아래 연산이 전부 다시 돕니다(682행에서 실측).

  get_working_df      0.02초
  build_eval_matrix   1.16초
  cross_val_predict   5.71초   ← 앵커 셀렉트박스를 한 번 만질 때마다
  ------------------------------
  합계                약 7초

앵커링은 셀렉트박스와 숫자 입력을 여러 번 조작하는 기능이라 이 7초가
매번 붙으면 쓰기 어렵습니다. 캐시를 붙이면 데이터가 바뀌지 않는 한
첫 실행만 7초이고 이후 조작은 즉시 반응합니다.

캐시 키는 DataFrame 의 내용 해시로 잡습니다 — 행이 추가·수정되면 자동으로
무효화되고, 그렇지 않으면 재사용됩니다.
"""
import hashlib

import pandas as pd


def df_signature(df: pd.DataFrame) -> str:
    """DataFrame 내용의 해시. 캐시 키로 씁니다.

    st.cache_data 는 DataFrame 을 인자로 받으면 직렬화해 해싱하는데 큰 표에서
    느립니다. 명시적 해시를 키로 넘기고 DataFrame 자체는 해싱에서 제외합니다.
    """
    if df is None or not len(df):
        return "empty"
    h = hashlib.sha1()
    h.update(str(df.shape).encode())
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def install(st, F2, v3, P=None):
    """캐시된 함수 3개를 돌려줍니다.

    사용법 (app.py 에서 무거운 호출을 이것으로 바꿉니다)
    ---------------------------------------------------
    >>> import lnp_app_cache as C
    >>> cached = C.install(st, F2, v3, P)
    >>> work_df, work_info = cached["working_df"](st.session_state.df)
    >>> oof = cached["oof"](work_df)
    """

    @st.cache_data(show_spinner=False)
    def _working(sig, _df):
        return F2.get_working_df(_df, patch_mod=P)

    @st.cache_data(show_spinner=False)
    def _matrix(sig, _df, include_measured):
        return F2.build_eval_matrix(_df, v3, include_measured=include_measured)

    @st.cache_data(show_spinner="논문 단위 교차검증 예측 계산 중… (첫 실행만)")
    def _oof(sig, _df, include_measured, n_splits):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import GroupKFold, cross_val_predict
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        X, y, groups, num_cols, cat_cols = F2.build_eval_matrix(
            _df, v3, include_measured=include_measured)
        if groups.nunique() < 2:
            return None
        steps = [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                                 ("s", StandardScaler())]), num_cols)]
        if cat_cols:
            steps.append(("c", Pipeline([
                ("i", SimpleImputer(strategy="most_frequent")),
                ("o", OneHotEncoder(handle_unknown="ignore",
                                    min_frequency=2))]), cat_cols))
        model = Pipeline([("pre", ColumnTransformer(steps)),
                          ("m", RandomForestRegressor(
                              n_estimators=400, min_samples_leaf=3,
                              max_features=0.5, random_state=42, n_jobs=-1))])
        k = int(min(n_splits, groups.nunique()))
        p = cross_val_predict(model, X, y, cv=GroupKFold(n_splits=k),
                              groups=groups)
        return pd.Series(p, index=y.index)

    return {
        "working_df": lambda df: _working(df_signature(df), df),
        "matrix": lambda df, include_measured=True: _matrix(
            df_signature(df), df, include_measured),
        "oof": lambda df, include_measured=True, n_splits=5: _oof(
            df_signature(df), df, include_measured, n_splits),
        "signature": df_signature,
    }
