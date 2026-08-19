# -*- coding: utf-8 -*-
"""개선된 앵커링 UI.

기존 앵커링 UI의 문제와 실측 근거:

1) 앵커링을 쓸 수 없는 논문도 선택 목록에 나옵니다. 166편 중 103편(62%)이
   행이 1개뿐이라 앵커를 잡으면 예측할 행이 남지 않습니다. 이제 그 논문은
   목록에서 빠지고, 왜 빠졌는지 표시됩니다.

2) 앵커를 사용자가 임의로 고릅니다. 예측값이 중앙에 가까운 처방을 고르면
   무작위보다 낫습니다(앵커 2개: 12.3% → 17.6%). 추천 버튼을 넣었습니다.

3) 보정량을 축소 없이 그대로 적용합니다. 앵커 1개에서 이것이 개선이 아니라
   악화(-4.4%)를 냈습니다. 이제 앵커 수에 맞춰 축소합니다.

4) 앵커 잔차가 서로 엇갈릴 때 경고가 없습니다. 영점 하나로 논문을 대표할 수
   없는 경우인데 결과는 정상처럼 보였습니다. 신뢰도 등급을 붙였습니다.
"""
import numpy as np
import pandas as pd

import lnp_anchor2 as A2

DOI = "reference_doi"
EE = "encapsulation_efficiency_percent_std_num"
RATIO = "lipid_molar_ratio"


def _label(df, i) -> str:
    r = df.loc[i]
    ee = pd.to_numeric(pd.Series([r.get(EE)]), errors="coerce").iloc[0]
    ion = str(r.get("ionizable_lipid_name") or "?")[:16]
    return f"[{i}] {ion} · {r.get(RATIO)} · EE {ee:.1f}%" if pd.notna(ee) \
        else f"[{i}] {ion} · {r.get(RATIO)} · EE 미기재"


def render(st, work_df, v3_module, fix2_module, oof_series=None):
    """앵커링 탭을 그립니다.

    oof_series: 미리 계산한 논문 단위 out-of-fold 예측(Series, work_df 인덱스).
                없으면 이 함수가 계산합니다(느립니다).
    """
    st.subheader("⚓ 앵커링 — 이 논문의 영점을 잡아 나머지를 예측")
    st.markdown(
        "새 논문·새 랩의 처방 중 **1~3개만 실제로 측정**하면, 그 오차로 영점을 "
        "잡아 같은 논문의 나머지 처방 예측을 보정합니다. EE 변동의 약 40%가 "
        "조성이 아니라 '어느 랩인지'에서 오기 때문입니다.")

    if DOI not in work_df or not len(work_df):
        st.info("데이터가 없습니다.")
        return

    # ---- 1. 앵커링이 가능한 논문만 목록에 올립니다 -----------------------
    key = work_df[DOI].astype(str).str.strip().str.lower()
    sz = key.value_counts()
    ok_papers = sorted(sz[sz >= 2].index)
    n_single = int((sz == 1).sum())

    if not ok_papers:
        st.warning(
            f"앵커링을 쓸 수 있는 논문이 없습니다. 논문 {len(sz)}편 모두 행이 "
            "1개뿐입니다 — 앵커로 쓰면 예측할 행이 남지 않습니다. "
            "같은 논문에서 처방 2개 이상을 입력하십시오.")
        return

    c1, c2 = st.columns([2, 1])
    c1.caption(
        f"앵커링 가능 논문 **{len(ok_papers)}편** (행 2개 이상). "
        f"행이 1개뿐인 {n_single}편은 앵커를 잡으면 예측할 대상이 남지 않아 "
        "목록에서 제외했습니다.")
    k_want = c2.selectbox("앵커 개수", [1, 2, 3], index=0,
                          help="실측 개선율: 1개 +15.4% / 2개 +17.6% / 3개 +20.0%. "
                               "1개만으로도 유의합니다(p=0.002).")

    sel = st.selectbox("기준 논문", ["(선택하세요)"] + ok_papers,
                       format_func=lambda s: s if s == "(선택하세요)"
                       else f"{s}  ({sz[s]}행)")
    if sel == "(선택하세요)":
        st.info("논문을 선택하면 앵커 추천이 나옵니다.")
        return

    sub = work_df[key == sel]
    if len(sub) <= k_want:
        st.warning(
            f"이 논문은 {len(sub)}행입니다. 앵커를 {k_want}개 쓰면 예측할 행이 "
            f"남지 않습니다 — 앵커 개수를 {len(sub) - 1}개 이하로 줄이십시오.")
        return

    # ---- 2. out-of-fold 예측 --------------------------------------------
    if oof_series is None:
        with st.spinner("논문 단위 out-of-fold 예측 계산 중…"):
            oof_series = _compute_oof(work_df, v3_module, fix2_module)
    if oof_series is None:
        st.error("out-of-fold 예측을 만들 수 없습니다(논문 2편 미만).")
        return

    pred_sub = oof_series.reindex(sub.index).dropna()
    if len(pred_sub) <= k_want:
        st.warning("이 논문에서 예측 가능한 행이 앵커 개수보다 적습니다.")
        return

    # ---- 3. 앵커 추천 ----------------------------------------------------
    rec = A2.suggest_anchors(pred_sub, k=k_want)
    st.caption(
        "**추천 기준** — 예측값이 이 논문의 중앙에 가까운 처방입니다. "
        "양 극단을 고르면 모델이 원래 크게 틀리는 지점이라 영점이 그 오차에 "
        "끌려갑니다(실측: 극단 선택 6.7% vs 중앙 선택 17.6%).")

    labels = {i: _label(sub, i) for i in pred_sub.index}
    picks, ys = [], []
    for j in range(k_want):
        cc1, cc2 = st.columns([3, 1])
        default = rec[j] if j < len(rec) else pred_sub.index[j]
        opts = list(pred_sub.index)
        i = cc1.selectbox(
            f"앵커 {j+1}", opts, index=opts.index(default),
            format_func=lambda x: labels[x] + ("  ⭐추천" if x in rec else ""),
            key=f"a2_pick_{j}")
        cur = pd.to_numeric(pd.Series([sub.loc[i].get(EE)]), errors="coerce").iloc[0]
        v = cc2.number_input(f"실측 EE {j+1} (%)", 0.0, 100.0,
                             float(cur) if pd.notna(cur) else 90.0, 0.1,
                             key=f"a2_ee_{j}",
                             help="이 랩에서 실제로 측정한 값으로 바꾸십시오.")
        picks.append(i)
        ys.append(float(v))

    if len(set(picks)) < len(picks):
        st.error("같은 처방을 두 번 골랐습니다. 서로 다른 행을 고르십시오.")
        return

    # ---- 4. 축소 보정 ----------------------------------------------------
    resid = [ys[j] - float(oof_series[picks[j]]) for j in range(len(picks))]
    info = A2.offset_reliability(resid, n_predict=len(pred_sub) - len(picks))
    off = info["offset"]

    # 영점을 세션에 게시해 설계 탭(최적화·What-If·PEG)이 쓸 수 있게 합니다.
    # off 는 A2.shrunk_offset 에서 이미 축소된 값이므로 already_shrunk=True 로
    # 넘겨야 합니다 — 그렇지 않으면 publish 가 축소를 한 번 더 걸어 영점이
    # 의도의 67~75% 로 줄어듭니다.
    # ref_pred 는 그 논문 전체 행의 예측 중앙값입니다. 여유 비례 보정의
    # 기준점이므로 반드시 넘기십시오(기본값 80.0 은 논문마다 크게 다릅니다).
    try:
        import lnp_offset_bus as OB
        OB.publish(st, raw_offset=off, k=info["n_anchor"], paper=str(sel),
                   n_rows_paper=len(sub), already_shrunk=True,
                   ref_pred=float(pred_sub.median()))
    except Exception as _e:
        st.caption(f"영점 전파 실패(설계 탭은 보정 없이 동작합니다): {_e}")

    m1, m2, m3 = st.columns(3)
    m1.metric("영점 보정량", f"{off:+.1f} %p",
              help="앵커 잔차의 중앙값에 축소를 적용한 값입니다.")
    m2.metric("축소 가중", f"{info['weight']:.0%}",
              help="앵커가 적을수록 영점 추정이 부정확하므로 0 쪽으로 당깁니다. "
                   "이 축소를 빼면 앵커 1개의 개선율이 15.4%→8.9%로 떨어집니다.")
    m3.metric("신뢰도", info["grade"])

    raw_med = float(np.median(resid))
    st.caption(f"앵커 잔차 중앙값 {raw_med:+.1f} %p → 축소 후 {off:+.1f} %p. "
               + info["message"])
    if info["grade"] == "낮음":
        st.warning(info["message"])

    # ---- 5. 결과표 -------------------------------------------------------
    rest = [i for i in pred_sub.index if i not in picks]
    tab = pd.DataFrame({
        "앵커": ["⚓" if i in picks else "" for i in pred_sub.index],
        "지질 몰비": sub.loc[pred_sub.index, RATIO].values,
        "이온화지질": sub.loc[pred_sub.index].get(
            "ionizable_lipid_name", pd.Series(index=pred_sub.index)).values,
        "보정 전 예측 (%)": pred_sub.values.round(1),
        "보정 후 예측 (%)": np.clip(pred_sub.values + off, 0, 100).round(1),
    }, index=pred_sub.index)

    meas = pd.to_numeric(sub.loc[pred_sub.index, EE], errors="coerce")
    if meas.notna().any():
        tab["문헌 EE (%)"] = meas.values.round(1)
        tab["보정 후 오차 (%p)"] = (np.clip(pred_sub.values + off, 0, 100)
                                 - meas.values).round(1)

    st.dataframe(tab, use_container_width=True)

    if rest and meas.notna().any():
        b = float(np.abs(pred_sub[rest].values - meas[rest].values).mean())
        a = float(np.abs(np.clip(pred_sub[rest].values + off, 0, 100)
                         - meas[rest].values).mean())
        d1, d2 = st.columns(2)
        d1.metric("앵커 외 행 MAE — 보정 전", f"{b:.1f} %p")
        d2.metric("앵커 외 행 MAE — 보정 후", f"{a:.1f} %p",
                  delta=f"{b - a:+.1f} %p", delta_color="normal")
        st.caption(
            "이 논문 하나의 결과입니다. 실측으로 63편을 돌렸을 때 앵커 1개는 "
            "40편에서 개선, 23편에서 악화됐습니다 — 논문마다 갈립니다. "
            "전체 평균 개선율이 +15.4%(p=0.002)라는 뜻이고, 개별 논문의 개선을 "
            "보장하지는 않습니다.")

    st.download_button(
        "결과 CSV 내려받기",
        tab.to_csv(index=False).encode("utf-8-sig"),
        f"anchored_{str(sel)[:24].replace('/', '_')}.csv", "text/csv")


def _compute_oof(work_df, v3_module, fix2_module, n_splits=5):
    """논문 단위 out-of-fold 예측. 학습=예측 값을 쓰면 2.4배 정확해 보입니다."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    X, y, groups, num_cols, cat_cols = fix2_module.build_eval_matrix(
        work_df, v3_module, include_measured=True)
    if groups.nunique() < 2:
        return None
    steps = [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler())]), num_cols)]
    if cat_cols:
        steps.append(("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                     ("o", OneHotEncoder(handle_unknown="ignore",
                                                         min_frequency=2))]), cat_cols))
    model = Pipeline([("pre", ColumnTransformer(steps)),
                      ("m", RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                                  max_features=0.5, random_state=42,
                                                  n_jobs=-1))])
    k = int(min(n_splits, groups.nunique()))
    p = cross_val_predict(model, X, y, cv=GroupKFold(n_splits=k), groups=groups)
    return pd.Series(p, index=y.index)
