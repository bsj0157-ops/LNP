# -*- coding: utf-8 -*-
"""app.py 에 붙일 PEG 전용 탭.

    import lnp_peg as PG
    from app_tab_peg import tab_peg
    with tab6: tab_peg(st, df)

df 는 P.dedupe() / P.clean_ee_column() 을 거친 데이터여야 합니다.
모델을 받지 않습니다 — 이 기능은 RF 예측이 아니라 논문 고정효과 회귀
기울기에 근거합니다(회귀가 크기 상관 rho=0.64 vs RF 0.43 으로 낫고,
계수와 신뢰구간을 사용자에게 그대로 보여줄 수 있습니다).
"""
import numpy as np
import pandas as pd

import lnp_peg as PG


def tab_peg(st, df):
    st.subheader("PEG 비율을 바꾸면 EE 가 어떻게 될까")

    with st.expander("이 기능의 검증 결과 — 먼저 읽어주세요", expanded=True):
        st.markdown(PG.CAVEAT)

    try:
        fit = PG.fit_peg_slope(df)
    except Exception as e:
        st.error(f"기울기를 추정할 수 없습니다: {e}")
        return

    meta = fit["_data"]
    c1, c2, c3 = st.columns(3)
    c1.metric("학습 데이터", f"{meta['n']}행 / {meta['n_papers']}편")
    if fit["high"]:
        f = fit["high"]
        c2.metric("PEG ≥ 2.5% 기울기", f"{f['slope']:+.2f} %p/1%p",
                  help=f"95% CI [{f['ci'][0]:+.2f}, {f['ci'][1]:+.2f}], "
                       f"p={f['p']:.2g}, n={f['n']}/{f['n_papers']}편")
    if fit["low"]:
        f = fit["low"]
        c3.metric("PEG < 2.5% 기울기", f"{f['slope']:+.2f} %p/1%p",
                  help=f"p={f['p']:.2g} — 유의하지 않아 사용하지 않습니다")

    peg_all = PG._peg_of(df)
    ee_all = pd.to_numeric(df.get(PG.EE_COL, pd.Series(np.nan, index=df.index)),
                           errors="coerce")
    ok = peg_all.notna() & ee_all.notna()
    if not ok.any():
        st.error("PEG 비율과 EE 를 함께 읽을 수 있는 행이 없습니다.")
        return

    idx_opts = list(df.index[ok])
    row = st.selectbox(
        "처방 선택",
        options=idx_opts,
        format_func=lambda i: (
            f"[{i}] PEG {peg_all.loc[i]:.2f}% · 실측 EE {ee_all.loc[i]:.1f}% · "
            f"{str(df.get('ionizable_lipid_name', pd.Series(['?']*len(df))).loc[i])[:24]}"),
        help="같은 처방에서 PEG 만 바꿉니다 (나머지 세 성분은 비율 유지)")

    rec = PG.recommend_peg(df, row, fit=fit)
    (st.success if rec["actionable"] else st.warning)(rec["text"])

    cur_peg = float(peg_all.loc[row])
    new_peg = st.slider("변경할 PEG 몰비 (%)", 0.0, 10.0, float(round(cur_peg, 2)),
                        step=0.1,
                        help="2.5% 미만은 검증되지 않은 구간이라 예측을 제공하지 않습니다")

    try:
        r = PG.predict_peg_change(df, row, new_peg, fit=fit)
    except ValueError as e:
        st.error(str(e))
        return

    if not r["usable"]:
        st.error(f"**예측을 제공하지 않습니다.** {r['note']}")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("문헌 실측 EE", f"{r['measured_ee']:.1f} %")
        m2.metric("예측 EE", f"{r['pred_ee']:.1f} %",
                  delta=f"{r['delta_ee']:+.1f} %p")
        m3.metric("95% 신뢰구간",
                  f"{r['delta_ci'][0]:+.1f} ~ {r['delta_ci'][1]:+.1f} %p")

        st.markdown(
            f"- PEG **{r['peg_before']:.2f}% → {r['peg_after']:.2f}%** "
            f"(Δ{r['d_peg']:+.2f} %p), 기울기 {r['slope']:+.2f} "
            f"%p EE / PEG 1 %p (p={r['slope_p']:.2g})\n"
            f"- 방향: **{r['direction']}** — 검증에서 이 구간 방향 적중률 "
            f"83.5% (n=85 / 6편)\n"
            f"- 개별 예측 불확실성 ±{r['pred_sd']:.1f} %p "
            f"(회귀 잔차 SD)")

        if r["clipped_at_bound"]:
            st.warning(
                "예측값이 0~100% 범위를 벗어나 잘렸습니다. 실측 EE 가 이미 "
                "높아(데이터의 10.5% 가 95% 초과) 선형 외삽이 상한에 부딪히는 "
                "경우입니다. 이 처방에서는 개선 여지를 신뢰할 수 없습니다.")

        st.info(
            f"**크기는 믿지 마십시오.** 변화량 예측의 MAE 는 33 %p 였습니다 "
            f"(실제 변화량 평균 크기 36 %p). 방향과 대략적 규모까지만 읽고, "
            f"값은 실험으로 확인하십시오.")

    st.markdown("---")
    st.markdown("**PEG 응답 곡선**")
    curve = PG.peg_curve(df, row, fit=fit, peg_min=0.5, peg_max=8.0, n=31)
    if len(curve):
        show = curve.copy()
        show["검증 구간"] = np.where(show.usable, show.pred_ee, np.nan)
        show["외삽 (근거 없음)"] = np.where(~show.usable, show.pred_ee, np.nan)
        st.line_chart(show.set_index("peg")[["검증 구간", "외삽 (근거 없음)"]])
        st.caption(
            f"검증 구간(PEG ≥ 2.5%)은 실측 기울기 "
            f"{fit['high']['slope']:+.2f} 를 씁니다. 2.5% 미만은 그 기울기를 "
            f"연장한 것일 뿐이며, 이 구간의 실제 기울기는 부호가 반대"
            f"({fit['low']['slope']:+.2f}, p={fit['low']['p']:.2g})였습니다 — "
            f"그래서 근거 없음으로 표시합니다.")

    with st.expander("전체 데이터에서 PEG 와 EE 의 관계"):
        q = pd.DataFrame({"peg": peg_all[ok], "ee": ee_all[ok]})
        qb = pd.qcut(q.peg, 4, duplicates="drop")
        g = q.groupby(qb, observed=True).ee.agg(["size", "mean"]).round(1)
        g.index = [f"{iv.left:.1f}–{iv.right:.1f}%" for iv in g.index]
        g.columns = ["행 수", "평균 EE (%)"]
        st.dataframe(g)
        st.caption(
            "전체 상관은 rho=-0.345 (p=7e-17) 이지만, 이 중 일부는 논문 간 "
            "차이입니다(논문 평균끼리 rho=-0.265). 논문 평균을 제거해도 "
            "rho=-0.227 (p<0.001) 이 남아 논문 내 효과가 실재하며, 위 기능은 "
            "논문더미를 넣은 회귀로 그 부분만 씁니다.")
