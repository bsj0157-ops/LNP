# -*- coding: utf-8 -*-
"""app.py 에 붙일 두 탭 — 최적 비율 탐색 / 비율 변경 what-if.

app.py 안에서 이렇게 씁니다.

    import lnp_optimize as O
    from app_tabs_optimize import tab_optimize, tab_whatif

    t1, t2, t3, t4, t5 = st.tabs([... , "최적 비율", "비율 what-if"])
    with t4: tab_optimize(st, df, model, v3_module=V3)
    with t5: tab_whatif(st, df, model, v3_module=V3)

df 는 이미 P.dedupe() / P.clean_ee_column() 을 거친 데이터,
model 은 학습이 끝난 파이프라인이어야 합니다.
"""
import numpy as np
import pandas as pd

import lnp_optimize as O
import lnp_uncertainty as U


def _banner(st):
    """실측 한계를 기능 위에 항상 띄웁니다.

    이 배너를 지우지 마십시오. 이 두 기능은 개별 예측이 신뢰 불가라는
    것을 실측으로 확인했고(what-if 방향 적중 44.2%), 배너 없이 숫자만
    보여주면 사용자가 실험 결정을 그 숫자에 걸게 됩니다.
    """
    with st.expander("⚠ 이 기능의 정확도 — 먼저 읽어주세요", expanded=True):
        st.markdown(O.CAVEAT)


# --------------------------------------------------------------------------
def tab_optimize(st, df, model, v3_module):
    st.subheader("데이터가 지지하는 범위에서 최적 조성 찾기")
    _banner(st)

    sup = O.data_support(df, v3_module)
    st.markdown("**탐색 범위** (관측 분포 5~95 백분위 — 이 밖은 외삽이라 제외)")
    st.dataframe(pd.DataFrame(
        [{"성분": O.COMP_KR[c], "최소": round(v[0], 1),
          "최대": round(v[1], 1), "중앙값": round(v[2], 1)}
         for c, v in sup.items()]), hide_index=True)

    c1, c2 = st.columns(2)
    n_grid = c1.slider("성분별 격자 점 수", 5, 11, 7,
                       help="7이면 7^4 = 2401개 조성을 평가합니다")
    top_n = c2.slider("표시할 후보 수", 5, 50, 15)

    tmpl = st.selectbox(
        "기준 처방 (지질 종류·cargo·공정은 이 행을 그대로 씁니다)",
        options=list(range(len(df))),
        format_func=lambda i: (
            f"[{i}] {df.get('ionizable_lipid_name', pd.Series(['?']*len(df))).iloc[i]}"
            f" / {df.get('cargo_type', pd.Series(['?']*len(df))).iloc[i]}"
            f" / 현재 {df.get('lipid_molar_ratio', pd.Series(['?']*len(df))).iloc[i]}"),
        help="비율만 바꿔 비교하려면 기준을 고정해야 합니다")

    if st.button("최적 조성 탐색", type="primary"):
        with st.spinner(f"{n_grid**4}개 조성 평가 중..."):
            T = O.optimize_ratio(df, model, v3_module,
                                 template_idx=int(tmpl),
                                 n_grid=n_grid, top_n=top_n)
        if T.empty:
            st.error("조성을 읽을 수 없습니다.")
            return

        # 💡 [핵심 패치] 신뢰도가 '매우 낮음'인 엉터리 처방 필터링
        trust_mask = T["pred_sd"].apply(lambda s: U.label_for(s) != "매우 낮음")
        filtered_T = T[trust_mask]
        
        n_tied = filtered_T.attrs.get("n_tied", 0) if hasattr(filtered_T, "attrs") else T.attrs.get("n_tied", 0)
        n_tot = T.attrs.get("n_grid_total", 0)
        span = T.attrs.get("grid_span", 0)

        if len(filtered_T) < len(T):
            st.warning(f"🚨 예측 신뢰도가 '매우 낮음'인 불안정한 레시피 {len(T) - len(filtered_T)}개를 상위 목록에서 제거했습니다.")
            if len(filtered_T) == 0:
                st.error("신뢰할 수 있는 예측 결과가 없습니다. 기준 처방이나 탐색 범위를 조정해 보세요.")
                return

        st.warning(
            f"**격자 {n_tot}개 중 상위 후보들이 통계적으로 구별되지 "
            f"않습니다.** 아래 순위는 참고용이며, 1등을 2등보다 낫다고 "
            f"말할 근거가 없습니다. (전체 격자 예측 폭 {span} %p, "
            f"개별 불확실성 ±{filtered_T.pred_sd.mean():.1f} %p)")

        show = filtered_T.rename(columns={
            "ionizable": "이온화(%)", "helper": "헬퍼(%)",
            "chol": "콜레스테롤(%)", "peg": "PEG(%)",
            "pred_ee": "예측 EE(%)", "pred_sd": "±불확실성",
            "delta_vs_template": "기준 대비", "rank_note": "순위 해석"})
            
        show["신뢰도"] = filtered_T["pred_sd"].apply(U.label_for)
        
        st.dataframe(show[["이온화(%)", "헬퍼(%)", "콜레스테롤(%)", "PEG(%)",
                           "예측 EE(%)", "±불확실성", "신뢰도", "기준 대비", "순위 해석"]],
                     hide_index=True)

        # 신뢰할 수 있는 유일한 결론을 명시적으로 보여줍니다
        st.info(
            f"**실제로 읽어야 할 결론: PEG 비율입니다.** 상위 후보의 PEG "
            f"중앙값은 {filtered_T.peg.median():.2f}% 입니다. 네 성분 중 PEG 만 "
            f"음성대조군 대비 뚜렷한 신호(2.7배)를 보였고, 실측 데이터에서도 "
            f"PEG–EE 상관이 rho=-0.35 (p=7e-17) 로 확인됩니다. "
            f"나머지 세 성분의 '최적값'은 데이터가 뒷받침하지 않습니다.")

        st.download_button("후보 조성 CSV 내려받기",
                           filtered_T.to_csv(index=False).encode("utf-8-sig"),
                           "lnp_optimize_candidates.csv", "text/csv")

        st.markdown("---")
        st.markdown("**성분별 응답 곡선** (나머지 세 성분은 비율 유지하며 재정규화)")
        comp = st.selectbox("성분 선택", O.COMP,
                            format_func=lambda c: O.COMP_KR[c], index=3)
        try:
            R = O.ratio_response(df, model, v3_module,
                                 row_idx=int(tmpl), component=comp)
            st.line_chart(R.set_index(comp)[["pred_ee", "lo", "hi"]])
            st.caption(
                f"{O.COMP_KR[comp]}: 예측 EE {R.pred_ee.min():.1f}~"
                f"{R.pred_ee.max():.1f} % (폭 {R.pred_ee.max()-R.pred_ee.min():.1f} %p), "
                f"불확실성 ±{R.pred_sd.mean():.1f} %p. "
                f"{'폭이 불확실성보다 작아 이 성분의 최적값은 판단할 수 없습니다.' if (R.pred_ee.max()-R.pred_ee.min()) < R.pred_sd.mean() else '폭이 불확실성에 근접합니다 — 방향만 참고하십시오.'}")
        except Exception as e:
            st.error(f"곡선을 그릴 수 없습니다: {e}")


# --------------------------------------------------------------------------
def tab_whatif(st, df, model, v3_module):
    st.subheader("기존 처방의 비율을 바꾸면 EE 가 어떻게 될까")
    _banner(st)

    st.error(
        "**이 탭의 개별 예측은 신뢰할 수 없습니다.** 지질 종류·cargo·공정을 "
        "고정하고 비율만 바꾼 231쌍을 검증했더니 증감 방향 적중률이 "
        "44.2% 였습니다 — 무작위(50%)보다 낮습니다. `유의` 판정이 나온 "
        "경우만 읽으십시오(231쌍 중 9쌍, 그 9쌍은 방향 9/9 적중, p=0.004).")

    row = st.selectbox(
        "바꿀 처방",
        options=list(range(len(df))),
        format_func=lambda i: (
            f"[{i}] {df.get('lipid_molar_ratio', pd.Series(['?']*len(df))).iloc[i]}"
            f"  실측 EE "
            f"{pd.to_numeric(df.get('encapsulation_efficiency_percent_std_num', pd.Series([np.nan]*len(df))), errors='coerce').iloc[i]}"))

    cur = str(df.get("lipid_molar_ratio", pd.Series(["50:10:38.5:1.5"]*len(df))).iloc[int(row)])
    new = st.text_input("새 비율 (이온화 : 헬퍼 : 콜레스테롤 : PEG)", value=cur)

    if st.button("예측", type="primary"):
        try:
            r = O.what_if(df, model, v3_module, int(row), new)
        except Exception as e:
            st.error(f"비율을 읽을 수 없습니다: {e}")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("문헌 실측 EE",
                  "없음" if r["measured_ee"] is None else f"{r['measured_ee']:.1f} %")
        
        c2.metric("변경 전 예측", f"{r['pred_before']:.1f} %")
        c3.metric("변경 후 예측", f"{r['pred_after']:.1f} %",
                  delta=f"{r['delta']:+.1f} %p")

        st.markdown(f"- 조성: `{r['ratio_before']}` → `{r['ratio_after']}`")
        st.markdown(f"- 예측 변화량 **{r['delta']:+.1f} %p**, "
                    f"불확실성 ±{r['delta_sd']:.1f} %p")
                    
        # 💡 [핵심 패치] 신뢰도 평가 및 경고 출력
        trust_label = U.label_for(r['delta_sd'])
        if trust_label == "매우 낮음":
             st.error("🚨 낯선 처방입니다! 트리 간 예측 편차가 너무 커서 신뢰도가 매우 낮습니다. 실험 결과가 크게 다를 수 있으니 참고하지 마십시오.")

        if r["significant"]:
            st.success(
                f"**유의 — 읽을 가치가 있습니다 ({r['verdict']}).** 예측 "
                f"변화가 불확실성을 넘었습니다. 검증에서 이 조건에 해당한 "
                f"9쌍은 방향을 모두 맞혔고(p=0.004) 크기도 근접했습니다"
                f"(예: 예측 -60 / 실제 -70). 단 표본이 9쌍이므로 확정된 "
                f"규칙은 아닙니다.")
        else:
            st.warning(
                f"**유의하지 않습니다 — 이 예측을 근거로 쓰지 마십시오.** "
                f"변화량 {abs(r['delta']):.1f} %p 가 불확실성 "
                f"±{r['delta_sd']:.1f} %p 이내입니다. 이 구간에서는 방향 "
                f"적중률이 41.9% 로 무작위보다 낮았습니다.")

        st.info(
            "**이 예측을 실제로 쓸 수 있게 만드는 방법: 앵커링입니다.** "
            "새 조건에서 조성 3개를 실측해 영점을 맞추면 절대값 오차가 31% "
            "줄어듭니다(p=0.0006, 25편 중 20편 개선). 앵커링 탭을 함께 "
            "쓰십시오.")
