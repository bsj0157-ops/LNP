# -*- coding: utf-8 -*-
"""앵커 영점을 설계 탭(최적화·What-If·PEG)까지 전파합니다.

## 현재 앱의 문제

`app.py` 450행에서 `summ["offset"]` 을 계산해 463행에서 화면에 표시한 뒤
**그대로 버립니다.** `st.session_state` 에 저장하지 않으므로 탭 5·6·7은 영점을
모르는 상태로 예측합니다. `tab_optimize(st, work_df, base_model, v3_module=v3)`
와 `tab_whatif(...)` 의 시그니처에는 영점을 받을 인자가 아예 없습니다.

특히 What-If 탭은 "이 예측을 실제로 쓸 수 있게 만드는 방법: 앵커링입니다 …
앵커링 탭을 함께 쓰십시오"라고 안내하는데, 실제로 함께 쓸 방법이 없습니다.

## 왜 중요한가 — 실측

논문 단위 out-of-fold 예측에서 논문별 영점(잔차 중앙값)을 재면, 앵커 3개가
가능한 44편에서 **|영점| 중앙값 6.2 %p**, 최대 40.3 %p 입니다. 26편이 5 %p를
넘습니다.

같은 논문에서 최적화 격자(7^4 = 2401개 조성)의 **예측 폭은 0.1~1.5 %p** 에
불과합니다. 즉 **영점이 설계 신호의 중앙값 35배**입니다. 영점을 적용하지 않으면
최적화 탭이 내놓는 절대값은 그 랩에서 의미가 없습니다.

앵커 3개로 영점을 잡아 같은 논문 나머지 행을 예측하면(33편·990회 시행):

| 방식 | MAE | 개선 | p | 개선 논문 |
|---|---|---|---|---|
| 영점 미적용 (현재 앱) | 12.97 %p | — | — | — |
| 원영점 그대로 | 10.81 %p | +16.7% | 0.30 | 16/33편 |
| **축소 영점** | **10.46 %p** | **+19.4%** | **0.051** | **21/33편** |

## 무엇에 영향을 주고 무엇에 주지 않는가

영점은 예측을 평행이동합니다. 따라서

* **최적화 탭의 절대 예측값** — 영향받습니다. 반드시 적용해야 합니다.
* **최적화 순위** — 불변입니다. 상위 후보 목록은 그대로입니다.
* **What-If 의 변화량 Δ** — 불변입니다(전·후 모두 같은 값이 더해지므로).
* **What-If 의 전·후 절대값** — 영향받습니다.
* **PEG 곡선의 높이** — 영향받습니다. 곡선 모양은 불변입니다.
* **유의성 판정** — 불변입니다(Δ와 불확실성이 모두 그대로).

이 구분을 UI에 명시해야 사용자가 영점을 오해하지 않습니다.

## 상한 처리 — 반드시 필요합니다

영점 +11.9 %p 를 최적화 상위 후보(예측 91.7%)에 더하면 **103.6%** 가 나옵니다.
EE는 100%를 넘을 수 없습니다. `apply_offset` 은 [0, 100] 으로 자르고, 잘렸다는
사실을 함께 돌려줍니다 — 조용히 자르면 사용자가 영점이 과대하다는 신호를
놓칩니다.
"""
import numpy as np
import pandas as pd

KEY = "lnp_anchor_offset"

# 앵커 수별 축소 계수 — lnp_anchor2 와 같은 값을 씁니다
LAMBDA_BY_K = {1: 0.5, 2: 0.75}
DEFAULT_LAMBDA = 1.0


def lambda_for(k: int) -> float:
    return LAMBDA_BY_K.get(int(k), DEFAULT_LAMBDA)


def shrink(raw_offset: float, k: int) -> float:
    """원영점을 앵커 수 k 에 맞춰 축소합니다."""
    k = max(1, int(k))
    return float(raw_offset) * k / (k + lambda_for(k))


# ---------------------------------------------------------------- 저장·조회
def publish(st, *, raw_offset: float, k: int, paper: str,
            n_rows_paper: int = None, mae_before: float = None,
            mae_after: float = None, ref_pred: float = None,
            already_shrunk: bool = False) -> dict:
    """앵커링 탭에서 영점을 계산한 직후 호출합니다.

    ref_pred
      그 논문(랩) 전체 행의 **모델 예측 중앙값**입니다. 여유 비례 보정의
      기준점으로 쓰이므로 함께 넘기십시오. 없으면 설계 탭이 80.0 을 씁니다.

    >>> import lnp_offset_bus as OB
    >>> OB.publish(st, raw_offset=summ["offset"], k=len(anchor_idx),
    ...            paper=sel_paper, n_rows_paper=len(sub_df),
    ...            ref_pred=float(sub_pred.median()))
    """
    # already_shrunk=True 는 raw_offset 이 이미 축소된 값(예: A2.shrunk_offset
    # 또는 A2.offset_reliability()["offset"])일 때 씁니다. 이 플래그 없이
    # 축소된 값을 넘기면 축소가 두 번 걸려 영점이 의도의 67~75% 로 줄어듭니다.
    _shrunk = float(raw_offset) if already_shrunk else shrink(raw_offset, k)
    rec = {"raw": float(raw_offset), "k": int(k),
           "shrunk": _shrunk, "already_shrunk": bool(already_shrunk),
           "paper": str(paper),
           "n_rows_paper": n_rows_paper,
           "mae_before": mae_before, "mae_after": mae_after,
           "ref_pred": None if ref_pred is None else float(ref_pred),
           "active": True}
    st.session_state[KEY] = rec
    return rec


def current(st) -> dict | None:
    """활성 영점을 돌려줍니다. 없거나 껐으면 None."""
    rec = st.session_state.get(KEY)
    if not rec or not rec.get("active"):
        return None
    return rec


def offset_value(st) -> float:
    """설계 탭에서 예측에 더할 값. 영점이 없으면 0.0."""
    rec = current(st)
    return float(rec["shrunk"]) if rec else 0.0


def reference_prediction(st, default: float = 80.0) -> float:
    """여유 비례 보정의 기준 예측값. 앵커 논문 전체 행의 예측 중앙값입니다."""
    rec = current(st)
    if rec and rec.get("ref_pred") is not None:
        return float(rec["ref_pred"])
    return float(default)


def clear(st) -> None:
    st.session_state.pop(KEY, None)


# ---------------------------------------------------------------- 적용
def apply_offset_headroom(pred, offset: float, hi: float = 100.0,
                          ref: float = None):
    """상한 여유에 비례해 영점을 적용합니다 — 포화해도 순위가 보존됩니다.

    `apply_offset` 은 평행이동 후 자르므로, 상한을 넘는 후보들이 모두 100 이
    되어 **서로 구별할 수 없게 됩니다.** 최적화 탭은 격자 2401개 중 예측
    최댓값을 고르므로 이 일이 실제로 일어납니다(측정: 영점 +11.9 %p 인
    논문에서 상위 15개가 전부 100 으로 잘렸습니다).

    여기서는 남은 여유 (hi - pred) 에 비례해 올립니다:

        보정값 = pred + offset × (hi - pred) / (hi - pred_ref)

    `pred_ref` 는 그 논문의 대표 예측값(중앙값)입니다. 예측이 낮은 후보는
    영점을 거의 그대로 받고, 이미 100 에 가까운 후보는 덜 받습니다. 결과가
    상한을 넘지 않으므로 자르기가 불필요하고 순위가 유지됩니다.

    영점이 음수면 하한(0) 쪽 여유를 씁니다.
    """
    if offset == 0:
        return pred, 0.0
    s = pd.Series(pred, dtype="float64") if not np.isscalar(pred) else pd.Series([float(pred)])
    # ref 는 그 랩의 **대표** 예측값이어야 합니다. 최적화 상위 후보 표의
    # 중앙값을 쓰면 안 됩니다 — 그 표는 격자 2401개 중 최댓값만 고른
    # 선택 편향 표본이라 이미 상한에 붙어 있고, 여유가 0에 가까워 배율이
    # 폭발합니다(측정: ref=93.8 로 잡으면 상위 15개가 모두 100 이 됩니다).
    # 앵커 논문 전체 행의 예측 중앙값을 넘기십시오.
    ref = float(s.median()) if ref is None else float(ref)
    if offset > 0:
        room_ref = max(hi - ref, 1e-9)
        adj = s + float(offset) * (hi - s).clip(lower=0) / room_ref
        adj = adj.clip(upper=hi)
    else:
        room_ref = max(ref, 1e-9)
        adj = s + float(offset) * s.clip(lower=0) / room_ref
        adj = adj.clip(lower=0.0)
    eff = float((adj - s).median())
    if np.isscalar(pred):
        return float(adj.iloc[0]), eff
    return adj, eff


def apply_offset(pred, offset: float, lo: float = 0.0, hi: float = 100.0):
    """예측에 영점을 더하고 [lo, hi] 로 자릅니다.

    반환: (보정값, n_clipped)  — 스칼라 입력이면 n_clipped 는 0/1.
    잘린 개수를 함께 돌려주는 것이 중요합니다. 영점 +12 %p 를 예측 92% 에
    더하면 104% 가 되는데, 조용히 100 으로 자르면 사용자는 영점이 과대하다는
    신호를 놓칩니다.
    """
    if offset == 0:
        return pred, 0
    if np.isscalar(pred):
        v = float(pred) + float(offset)
        return float(min(max(v, lo), hi)), int(v < lo or v > hi)
    s = pd.Series(pred, dtype="float64") + float(offset)
    n_clip = int(((s < lo) | (s > hi)).sum())
    return s.clip(lo, hi), n_clip


def banner(st, *, context: str = "absolute") -> float:
    """설계 탭 상단에 영점 상태를 표시하고 적용할 값을 돌려줍니다.

    context
      "absolute" : 절대값이 중요한 탭(최적화·PEG) — 영점이 없으면 경고합니다.
      "delta"    : 변화량이 중요한 탭(What-If) — 영점이 Δ 에는 영향이 없다고
                   명시합니다.
    """
    rec = current(st)
    if rec is None:
        if context == "absolute":
            st.warning(
                "**앵커 영점이 적용되지 않았습니다.** 이 탭의 절대 예측값은 "
                "문헌 평균 기준이며, 여러분 랩의 값과 중앙값 6.2 %p (최대 40 %p) "
                "차이가 납니다. 최적화 격자 전체의 예측 폭이 1 %p 안팎이므로 "
                "**영점 차이가 조성 차이의 35배**입니다. 앵커링 탭에서 여러분 "
                "데이터로 영점을 먼저 잡으십시오.")
        else:
            st.info(
                "앵커 영점이 없습니다. 이 탭의 **변화량 Δ 는 영점과 무관하므로 "
                "그대로 읽을 수 있습니다.** 전·후 절대값을 여러분 랩 기준으로 "
                "보려면 앵커링 탭에서 영점을 잡으십시오.")
        return 0.0

    on = st.checkbox(
        f"앵커 영점 {rec['shrunk']:+.1f} %p 적용 "
        f"(앵커 {rec['k']}개 · {str(rec['paper'])[:38]})",
        value=True, key=f"ob_on_{context}")
    if not on:
        return 0.0

    # already_shrunk 인 경우 raw 와 shrunk 가 같으므로 축소 문구가 무의미합니다.
    w = rec["k"] / (rec["k"] + lambda_for(rec["k"]))
    head = (f"앵커 {rec['k']}개에 맞춰 관측 잔차의 {w:.0%} 만 적용한 값입니다 "
            f"(λ={lambda_for(rec['k'])})."
            if rec.get("already_shrunk") else
            f"원영점 {rec['raw']:+.1f} %p 를 앵커 {rec['k']}개에 맞춰 "
            f"{rec['shrunk']:+.1f} %p 로 축소했습니다 (λ={lambda_for(rec['k'])}).")
    st.caption(
        head + " "
        + ("영점은 절대값을 평행이동하며 **순위와 곡선 모양은 바뀌지 "
           "않습니다.**" if context == "absolute" else
           "영점은 **Δ 와 유의성 판정에는 영향이 없고** 전·후 절대값만 "
           "옮깁니다."))
    return float(rec["shrunk"])
