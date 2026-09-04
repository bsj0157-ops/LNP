# -*- coding: utf-8 -*-
"""지질 비율 최적화 + 비율 변경 시 EE 예측 (what-if).

app.py 에 두 기능을 붙이기 위한 모듈입니다.

  optimize_ratio(...)  — 데이터가 지지하는 범위 안에서 EE 최대 조성을 찾습니다
  what_if(...)         — 기존 처방의 비율만 바꿨을 때 예측 EE 를 냅니다
  ratio_response(...)  — 한 성분을 훑으며 예측 EE 곡선을 냅니다 (그래프용)

--------------------------------------------------------------------------
이 모듈을 쓰기 전에 반드시 읽어야 하는 실측 결과
--------------------------------------------------------------------------
554행 / 91편(중복 제거) 데이터, 논문 단위 5-fold CV 로 측정했습니다.

1) 절대값 예측은 신뢰할 수 없습니다.
       CV MAE = 16.4 %p  (실측 EE 표준편차 21.6 %p)
   따라서 이 모듈의 출력은 "이 조성의 EE 는 87%" 가 아니라
   "이 방향으로 바꾸면 오를 가능성이 있다" 수준으로만 읽어야 합니다.

2) **비율만 바꾼 what-if 는 방향조차 맞히지 못합니다.** 이것이 가장 중요합니다.
   두 가지 검증을 구분해야 합니다.

   (a) 논문 내 처방 쌍을 그대로 비교 (지질 종류·cargo 도 함께 다름)
           7171쌍 중 증감 방향 일치 55.0%  (p=2e-17)
           EE 차이 10 %p 이상 4720쌍만 보면 57.6%  (p=2e-25)
       → 유의하지만 약합니다. 그리고 이 55% 는 조성이 아니라 지질 종류
         차이에서 나온 것일 수 있습니다.

   (b) 지질 종류·cargo·공정을 고정하고 **비율만** 교체 (= what_if() 가 하는 일)
           231쌍 (EE 차이 5 %p 이상) 중 방향 일치 44.2%
           무작위(50%) 대비 p = 0.087 — 무작위보다 오히려 낮습니다.

   (a) 와 (b) 의 차이가 결론입니다: 이 데이터에서 EE 를 설명하는 것은
   지질 '종류'와 논문 조건이고, 같은 지질에서의 '비율'은 거의 설명하지
   못합니다. what_if() 의 개별 예측은 신뢰할 수 없습니다.

   단 하나의 예외 — 예측 변화량이 불확실성을 넘는 경우(significant=True):
           231쌍 중 9쌍만 해당, 그 9쌍은 방향 9/9 적중 (p=0.004)
           예측 -60.3 / 실제 -70,  예측 +43.2 / 실제 +38  처럼 크기도 근접
   즉 significant=True 일 때만 읽을 가치가 있고, 그것은 조성을 극단적으로
   (PEG 2.7% -> 0.1% 처럼) 바꿨을 때만 나타납니다. n=9 이므로 이 규칙 자체도
   더 많은 데이터로 재확인해야 합니다.

3) 네 성분 중 PEG 만 실제 화학 신호를 보입니다.
   조성을 +10 %p 흔들었을 때 예측이 움직이는 폭 (%p):
       성분        실제 y   y를 섞은 대조군   비(신호/잡음)
       PEG          14.4        5.4           2.7   ← 신호
       헬퍼          3.6        2.6           1.4
       이온화        3.5        2.5           1.4
       콜레스테롤    3.3        1.8           1.8
   y 를 논문 내에서 무작위로 섞어도 이온화·헬퍼는 반응 크기가 비슷합니다.
   즉 PEG 를 제외한 성분의 '최적값'은 데이터가 뒷받침하지 않습니다.

4) 불확실성을 반드시 함께 보여야 합니다.
   RF 트리 분산은 실제 오차의 48% 만 반영합니다 (SD 7.9 vs MAE 16.4).
   그래서 UNCERTAINTY_SCALE 로 보정합니다.

결론 — 이 기능은 **가설 생성기**입니다. 실험 순서를 정하는 데 쓰고,
실험을 대체하는 데 쓰지 마십시오. 앵커링(논문당 3개 실측)과 함께 쓰면
절대값 오차가 31% 줄어듭니다.
"""
import io
import contextlib
import itertools

import numpy as np
import pandas as pd

COMP = ["ionizable", "helper", "chol", "peg"]
COMP_KR = {"ionizable": "이온화지질", "helper": "헬퍼지질",
           "chol": "콜레스테롤", "peg": "PEG지질"}

# 트리 분산이 실제 오차의 48% 만 반영 → 보정 계수 (실측 16.44/7.89)
UNCERTAINTY_SCALE = 1.95

# 신호/잡음비가 2.0 을 넘은 성분만 '최적화 근거 있음'으로 표시합니다 (위 3번)
TRUSTED_COMPONENTS = ["peg"]


# --------------------------------------------------------------------------
# 조성 특징 재계산
# --------------------------------------------------------------------------
def ratio_features(comp4):
    """4성분 값 -> v3.parse_lipid_ratio 와 동일한 파생특징.

    합계 100 정규화, ion_to_helper, ion_plus_chol, log_peg 까지 맞춥니다.
    v3 와 같은 식을 쓰지 않으면 what-if 예측이 학습 시점과 다른 좌표계에
    놓여 조용히 틀립니다.
    """
    a = np.asarray(comp4, dtype=float)
    t = np.nansum(a)
    if not np.isfinite(t) or t <= 0:
        return None
    a = a / t * 100.0
    d = dict(zip(COMP, a))
    d["ion_to_helper"] = (d["ionizable"] / d["helper"]
                          if d["helper"] > 0 else np.nan)
    d["ion_plus_chol"] = d["ionizable"] + d["chol"]
    d["log_peg"] = np.log1p(d["peg"])
    return d


def substitute_ratio(x_row, comp4):
    """특징 벡터 한 행에서 조성 관련 값만 교체합니다 (지질 종류·공정은 유지)."""
    f = ratio_features(comp4)
    if f is None:
        return None
    r = x_row.copy()
    for k, v in f.items():
        if k in r.index:
            r[k] = v
    return r


# --------------------------------------------------------------------------
# 데이터가 지지하는 범위
# --------------------------------------------------------------------------
def data_support(df, v3_module, lo=5, hi=95):
    """각 성분의 관측 분포에서 신뢰 구간을 뽑습니다.

    학습 데이터에 없는 조성(예: PEG 20%)을 추천하면 그건 외삽이고,
    RF 는 외삽 구간에서 학습 범위의 경계값을 그대로 되돌려 줍니다 —
    그럴듯한 숫자가 나오지만 근거는 없습니다. 그래서 범위를 제한합니다.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        X, _, _ = v3_module.build_features(df, include_measured=False)
    out = {}
    for c in COMP:
        if c in X.columns:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            if len(s):
                out[c] = (float(np.percentile(s, lo)),
                          float(np.percentile(s, hi)),
                          float(s.median()))
    return out


# --------------------------------------------------------------------------
# 1) 최적 비율 탐색
# --------------------------------------------------------------------------
def optimize_ratio(df, model, v3_module, template_idx=0,
                   n_grid=7, top_n=10, respect_support=True):
    """데이터가 지지하는 범위 안에서 예측 EE 가 높은 조성을 찾습니다.

    template_idx 행의 지질 종류·cargo·공정 조건은 그대로 두고 비율만 바꿉니다
    (지질 종류까지 같이 바꾸면 무엇이 원인인지 알 수 없습니다).

    돌려주는 표의 컬럼:
        ionizable/helper/chol/peg  조성 (합 100)
        pred_ee                    예측 EE (%)
        pred_sd                    보정된 불확실성 (±%p, 1 sigma)
        in_support                 네 성분 모두 관측 범위 안인가
        delta_vs_template          템플릿 대비 예측 변화량
    """
    with contextlib.redirect_stdout(io.StringIO()):
        X, num_cols, cat_cols = v3_module.build_features(
            df, include_measured=False)
    if template_idx >= len(X):
        template_idx = 0
    base_row = X.iloc[template_idx]
    sup = data_support(df, v3_module)

    grids = []
    for c in COMP:
        if c in sup and respect_support:
            lo, hi, _ = sup[c]
        else:
            lo, hi = (0.0, 60.0) if c != "peg" else (0.5, 5.0)
        grids.append(np.linspace(lo, hi, n_grid))

    rows, combos = [], []
    for combo in itertools.product(*grids):
        if sum(combo) <= 0:
            continue
        r = substitute_ratio(base_row, combo)
        if r is None:
            continue
        rows.append(r)
        combos.append(combo)
    if not rows:
        return pd.DataFrame()

    G = pd.DataFrame(rows).reset_index(drop=True)
    pred, sd = predict_with_uncertainty(model, G)

    C = pd.DataFrame(np.array(combos), columns=COMP)
    tot = C.sum(axis=1)
    C = C.div(tot, axis=0) * 100.0
    C["pred_ee"] = pred
    C["pred_sd"] = sd

    if respect_support:
        ok = np.ones(len(C), bool)
        for c in COMP:
            if c in sup:
                lo, hi, _ = sup[c]
                ok &= C[c].between(lo - 1e-9, hi + 1e-9).values
        C["in_support"] = ok
    else:
        C["in_support"] = True

    base_pred, _ = predict_with_uncertainty(model, base_row.to_frame().T)
    C["delta_vs_template"] = C.pred_ee - float(base_pred[0])

    C = C.sort_values("pred_ee", ascending=False).reset_index(drop=True)

    # 상위권은 서로 구별되지 않습니다 — 실측으로 확인한 사실입니다.
    # 격자 2401점 전체는 예측 64.7~89.7 (폭 25 %p) 로 넓게 반응하지만,
    # 상위 200점만 보면 폭이 1.05 %p 인데 불확실성이 ±11.1 %p 입니다
    # (예측 폭이 불확실성의 0.09배). 1등을 2등보다 낫다고 말할 근거가 없습니다.
    # 그래서 "최적 1개"가 아니라 "동등 후보군"으로 돌려줍니다.
    best = float(C.pred_ee.iloc[0])
    tol = float(C.pred_sd.iloc[0])
    C["tied_with_best"] = C.pred_ee >= best - tol
    C["rank_note"] = np.where(
        C["tied_with_best"],
        "1등과 구별 불가 (불확실성 이내)", "1등보다 유의하게 낮음")
    out = C.head(top_n).round(2)
    out.attrs["n_tied"] = int(C.tied_with_best.sum())
    out.attrs["n_grid_total"] = int(len(C))
    out.attrs["grid_span"] = round(float(C.pred_ee.max() - C.pred_ee.min()), 2)
    return out


# --------------------------------------------------------------------------
# 2) what-if — 비율만 바꿨을 때
# --------------------------------------------------------------------------
def what_if(df, model, v3_module, row_idx, new_ratio):
    """row_idx 처방의 비율을 new_ratio 로 바꿨을 때 예측 EE.

    new_ratio: '50:10:38.5:1.5' 문자열 또는 [50, 10, 38.5, 1.5] 리스트
               (이온화 : 헬퍼 : 콜레스테롤 : PEG 순서)

    돌려주는 dict 에는 원래 조성의 예측, 문헌 실측값, 변경 후 예측,
    그리고 변화량이 신뢰구간 안인지 여부가 들어갑니다.
    """
    if isinstance(new_ratio, str):
        import re
        vals = [float(v) for v in re.split(r"[:\/,;|\-]+", new_ratio.strip())
                if v.strip()]
    else:
        vals = list(map(float, new_ratio))
    if len(vals) == 3:              # 헬퍼 없는 3성분 처방
        vals = [vals[0], 0.0, vals[1], vals[2]]
    if len(vals) != 4:
        raise ValueError("비율은 4개(또는 3개) 값이어야 합니다: "
                         "이온화:헬퍼:콜레스테롤:PEG")

    with contextlib.redirect_stdout(io.StringIO()):
        X, _, _ = v3_module.build_features(df, include_measured=False)
    base_row = X.iloc[row_idx]
    new_row = substitute_ratio(base_row, vals)
    if new_row is None:
        raise ValueError("비율 합이 0 입니다.")

    p_old, sd_old = predict_with_uncertainty(model, base_row.to_frame().T)
    p_new, sd_new = predict_with_uncertainty(model, new_row.to_frame().T)

    ee_col = "encapsulation_efficiency_percent_std_num"
    measured = (pd.to_numeric(df[ee_col], errors="coerce").iloc[row_idx]
                if ee_col in df.columns else np.nan)

    delta = float(p_new[0] - p_old[0])
    # 변화량이 두 예측의 불확실성 합보다 작으면 의미 있는 차이가 아닙니다
    noise = float(np.hypot(sd_old[0], sd_new[0]))

    return {
        "row_idx": int(row_idx),
        "ratio_before": _fmt(base_row),
        "ratio_after": ":".join(f"{v:g}" for v in
                                np.array(vals) / sum(vals) * 100),
        "measured_ee": None if pd.isna(measured) else float(measured),
        "pred_before": float(p_old[0]),
        "pred_after": float(p_new[0]),
        "delta": delta,
        "delta_sd": noise,
        "significant": bool(abs(delta) > noise),
        "verdict": ("의미 있는 변화로 보기 어렵습니다 (불확실성 이내)"
                    if abs(delta) <= noise else
                    ("상승 방향" if delta > 0 else "하락 방향")),
    }


def ratio_response(df, model, v3_module, row_idx, component,
                   n_points=21, respect_support=True):
    """한 성분을 훑으며 예측 EE 곡선을 냅니다 (app.py 그래프용).

    나머지 세 성분은 원래 비율을 유지한 채 합이 100 이 되도록 재정규화합니다.
    """
    if component not in COMP:
        raise ValueError(f"component 는 {COMP} 중 하나여야 합니다.")
    with contextlib.redirect_stdout(io.StringIO()):
        X, _, _ = v3_module.build_features(df, include_measured=False)
    base_row = X.iloc[row_idx]
    cur = np.array([base_row.get(c, np.nan) for c in COMP], float)
    if not np.isfinite(cur).all():
        raise ValueError("이 행의 조성을 읽을 수 없습니다.")

    sup = data_support(df, v3_module)
    lo, hi = (sup[component][0], sup[component][1]) if (
        respect_support and component in sup) else (0.0, 60.0)
    ci = COMP.index(component)
    others = [i for i in range(4) if i != ci]
    other_sum = cur[others].sum()

    rows, xs = [], []
    for v in np.linspace(lo, hi, n_points):
        a = cur.copy()
        a[ci] = v
        if other_sum > 0:                      # 나머지는 비율 유지
            a[others] = cur[others] / other_sum * (100.0 - v)
        r = substitute_ratio(base_row, a)
        if r is not None:
            rows.append(r)
            xs.append(v)
    G = pd.DataFrame(rows).reset_index(drop=True)
    pred, sd = predict_with_uncertainty(model, G)
    return pd.DataFrame({component: xs, "pred_ee": pred, "pred_sd": sd,
                         "lo": pred - sd, "hi": pred + sd}).round(3)


# --------------------------------------------------------------------------
# 불확실성 포함 예측
# --------------------------------------------------------------------------
def predict_with_uncertainty(model, X):
    """예측값과 보정된 표준편차를 함께 돌려줍니다.

    RandomForest 의 트리 간 표준편차는 실제 CV 오차의 48% 밖에 되지 않습니다
    (실측: 트리 SD 7.89 %p vs CV MAE 16.44 %p). 그대로 쓰면 과신하므로
    UNCERTAINTY_SCALE 을 곱합니다. 트리를 못 꺼내는 모델이면
    NaN 대신 CV MAE 를 상수로 씁니다.
    """
    pred = np.asarray(model.predict(X), float)
    sd = None
    # v7 래퍼처럼 산포를 직접 주는 모델이면 그것을 씁니다. 점추정이 HistGB 인
    # 모델은 named_steps 로 트리를 꺼낼 수 없어 아래 경로가 상수로 떨어집니다.
    if hasattr(model, "predict_sd"):
        try:
            s = np.asarray(model.predict_sd(X), float)
            if np.isfinite(s).any():
                return pred, s * UNCERTAINTY_SCALE
        except Exception:
            pass
    try:
        steps = getattr(model, "named_steps", None)
        if steps:
            keys = list(steps.keys())
            est = steps[keys[-1]]
            pre = steps[keys[0]] if len(keys) > 1 else None
            Z = pre.transform(X) if pre is not None else X
            if hasattr(est, "estimators_"):
                tp = np.stack([t.predict(Z) for t in est.estimators_])
                sd = tp.std(axis=0) * UNCERTAINTY_SCALE
    except Exception:
        sd = None
    if sd is None:
        sd = np.full(len(pred), 13.1)     # 실측 CV MAE (v7, 1107행/374논문)
    return pred, sd


def _fmt(x_row):
    try:
        return ":".join(f"{float(x_row[c]):g}" for c in COMP)
    except Exception:
        return "?"


# --------------------------------------------------------------------------
# 신뢰도 배너 — app.py 에서 기능 위에 반드시 띄우십시오
# --------------------------------------------------------------------------
CAVEAT = """**현재 데이터로는 이 두 기능이 신뢰할 만한 답을 주지 못합니다.**
554행/91편, 논문 단위 교차검증으로 직접 측정한 결과입니다.

| 검증 항목 | 결과 | 판정 |
|---|---|---|
| 절대값 예측 오차 | MAE 16.4 %p (EE 표준편차 21.6 %p) | 신뢰 불가 |
| 비율만 바꿀 때 증감 방향 | 231쌍 중 44.2% 적중 (무작위 50%) | 무작위 이하 |
| 최적 조성 상위권 구별 | 상위 200개 예측 폭 1.05 %p vs 불확실성 ±11.1 %p | 구별 불가 |
| 성분별 신호 (섞은 y 대비) | PEG 2.7배 / 나머지 1.4~1.8배 | PEG만 신호 |

**읽어야 할 것은 개별 숫자가 아니라 큰 방향입니다.** 격자 2401점 전체로 보면
예측이 64.7~89.7 %p 로 반응하고, PEG 가 낮은 조성(평균 88.0)이 높은
조성(평균 80.3)보다 좋게 나옵니다 — 이건 화학적으로도 타당하고 무작위
대조군에서 사라지므로 실제 신호입니다.

**쓰는 방법**
1. 최적화 탭의 결과는 "1등 조성"이 아니라 **PEG 가 낮은 후보군**으로 읽습니다.
   상위권끼리는 통계적으로 구별되지 않습니다 (`tied_with_best` 컬럼 확인).
2. what-if 는 `significant=True` 인 경우만 읽습니다. 231쌍 중 9쌍만
   해당했고 그 9쌍은 방향을 모두 맞혔지만(p=0.004), 조성을 극단적으로
   바꿨을 때만 나타납니다.
3. **실제로 정확도를 올리는 방법은 앵커링입니다** — 새 조성 3개를 실측해
   영점을 맞추면 오차가 31% 줄어듭니다 (p=0.0006). 예측만으로는 안 됩니다.
"""
