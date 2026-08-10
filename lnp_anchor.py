# -*- coding: utf-8 -*-
"""앵커링 EE 예측기.

새 실험 조건(새 논문 / 새 랩 / 새 장비)에서 처방 2개만 실측하면,
그 오프셋으로 나머지 처방의 예측을 보정합니다.

왜 이렇게 하는가 — 실측 근거:
  조성만으로 예측    MAE 16.9 %p  (baseline 17.2 와 사실상 동일)
  앵커 2개 실측 후   MAE 12.6 %p  (26% 개선)
  논문조건 완전확보  MAE  9.7 %p  (이론 상한)
EE 변동의 약 45%가 '어느 논문/랩인지'에서 오기 때문에(ICC 0.45),
그 오프셋 하나를 없애는 것이 모델을 바꾸는 것보다 훨씬 큽니다.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class AnchoredEEPredictor:
    """조성 -> EE 예측 + 앵커 기반 오프셋 보정.

    사용법
    ------
    >>> import lnp_predictor_v3 as v3, lnp_anchor
    >>> X, num_cols, cat_cols = v3.build_features(train_df)
    >>> m = lnp_anchor.AnchoredEEPredictor(v3, num_cols, cat_cols)
    >>> m.fit(X, train_df["encapsulation_efficiency_percent_std_num"])
    >>> # 새 조건에서 2개 실측 -> 나머지 예측
    >>> m.predict(X_new, anchor_idx=[0, 1], anchor_y=[88.0, 71.5])
    """

    def __init__(self, v3_module, num_cols, cat_cols, n_estimators=400,
                 random_state=0):
        self.v3 = v3_module
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.model = v3_module.make_pipeline(
            num_cols, cat_cols,
            RandomForestRegressor(n_estimators=n_estimators,
                                  random_state=random_state, n_jobs=-1))
        self.offset_ = 0.0

    def fit(self, X, y):
        self.model.fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X, anchor_idx=None, anchor_y=None, clip=(0, 100)):
        """anchor_idx / anchor_y 를 주면 그 잔차의 중앙값만큼 전체를 이동.

        중앙값을 쓰는 이유: 앵커가 2-3개뿐이라 하나가 이상치면 평균은
        통째로 끌려갑니다. 중앙값은 그 영향을 덜 받습니다.
        """
        p = self.model.predict(X)
        if anchor_idx is not None and anchor_y is not None:
            ai = np.asarray(anchor_idx)
            ay = np.asarray(anchor_y, dtype=float)
            if len(ai) != len(ay):
                raise ValueError("anchor_idx 와 anchor_y 의 길이가 다릅니다")
            self.offset_ = float(np.median(ay - p[ai]))
            p = p + self.offset_
        return np.clip(p, *clip)

    def suggest_anchors(self, X, k=2):
        """어떤 처방을 실측할지 고릅니다.

        예측값의 분위수를 고르게 훑도록 선택합니다 — 한쪽 끝만 재면
        오프셋 추정이 그 끝으로 치우칩니다.
        """
        p = self.model.predict(X)
        qs = np.linspace(0.25, 0.75, k) if k > 1 else [0.5]
        picks = []
        for q in qs:
            target = np.quantile(p, q)
            order = np.argsort(np.abs(p - target))
            for i in order:
                if i not in picks:
                    picks.append(int(i))
                    break
        return picks
