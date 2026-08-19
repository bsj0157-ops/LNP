# -*- coding: utf-8 -*-
"""설계 탭(최적화·What-If·PEG)에 앵커 영점을 적용하는 감싸개.

기존 `app_tabs_optimize.tab_optimize` / `tab_whatif` / `app_tab_peg.tab_peg` 를
그대로 두고, 예측 절대값에만 영점을 더해 표시합니다. 원 모듈을 고치지 않으므로
영점 없이 쓰던 동작은 그대로 유지됩니다.

app.py 교체 방법
----------------
    # 기존
    #   tab_optimize(st, work_df, base_model, v3_module=v3)
    #   tab_whatif(st, work_df, base_model, v3_module=v3)
    #   tab_peg(st, work_df)
    # 교체
    import app_tabs_offset as TO
    TO.tab_optimize_anchored(st, work_df, base_model, v3, O)
    TO.tab_whatif_anchored(st, work_df, base_model, v3, O)
    TO.tab_peg_anchored(st, work_df, peg_module)

그리고 앵커링 탭에서 영점을 계산한 직후 한 줄을 넣습니다:

    import lnp_offset_bus as OB
    OB.publish(st, raw_offset=summ["offset"], k=len(anchor_idx),
               paper=sel_paper, n_rows_paper=len(sub_df))
"""
import numpy as np
import pandas as pd

import lnp_offset_bus as OB


def tab_optimize_anchored(st, df, model, v3_module, O):
    """최적화 탭 — 절대 예측값에 영점을 적용합니다. 순위는 불변입니다."""
    st.subheader("데이터가 지지하는 범위에서 최적 조성 찾기")
    off = OB.banner(st, context="absolute")

    sup = O.data_support(df, v3_module)
    st.markdown("**탐색 범위** (관측 분포 5~95 백분위 — 이 밖은 외삽이라 제외)")
    st.dataframe(pd.DataFrame(
        [{"성분": O.COMP_KR[c], "최소": round(v[0], 1),
          "최대": round(v[1], 1), "중앙값": round(v[2], 1)}
         for c, v in sup.items()]), hide_index=True)

    c1, c2 = st.columns(2)
    n_grid = c1.slider("성분별 격자 점 수", 5, 11, 7, key="oa_grid")
    top_n = c2.slider("표시할 후보 수", 5, 50, 15, key="oa_top")

    tmpl = st.selectbox(
        "기준 처방 (지질 종류·cargo·공정은 이 행을 그대로 씁니다)",
        options=list(range(len(df))), key="oa_tmpl",
        format_func=lambda i: (
            f"[{i}] {df.get('ionizable_lipid_name', pd.Series(['?']*len(df))).iloc[i]}"
            f" / {df.get('cargo_type', pd.Series(['?']*len(df))).iloc[i]}"
            f" / 현재 {df.get('lipid_molar_ratio', pd.Series(['?']*len(df))).iloc[i]}"))

    if not st.button("최적 조성 탐색", type="primary", key="oa_run"):
        return
    with st.spinner(f"{n_grid**4}개 조성 평가 중..."):
        T = O.optimize_ratio(df, model, v3_module, template_idx=int(tmpl),
                             n_grid=n_grid, top_n=top_n)
    if T.empty:
        st.error("조성을 읽을 수 없습니다.")
        return

    n_tied = T.attrs.get("n_tied", 0)
    n_tot = T.attrs.get("n_grid_total", 0)
    span = T.attrs.get("grid_span", 0)
    st.warning(
        f"**격자 {n_tot}개 중 {n_tied}개가 1등과 통계적으로 구별되지 "
        f"않습니다.** 아래 순위는 참고용입니다. (전체 격자 예측 폭 {span} %p, "
        f"개별 불확실성 ±{T.pred_sd.mean():.1f} %p)")

    if off:
        # ref 는 앵커 논문 전체 행의 예측 중앙값입니다. 상위 후보 표의
        # 중앙값을 쓰면 안 됩니다 — 선택 편향으로 이미 상한에 붙어 있습니다.
        ref = OB.reference_prediction(st)
        adj, eff = OB.apply_offset_headroom(T.pred_ee, off, ref=ref)
        T = T.assign(pred_ee_lab=np.round(T.pred_ee, 1), pred_ee=adj)
        st.caption(
            f"영점 {off:+.1f} %p 를 상한 여유에 비례해 적용했습니다 "
            f"(실효 {eff:+.1f} %p, 기준 예측 {ref:.1f}%). 100%를 넘지 않으므로 "
            f"**순위가 그대로 보존됩니다** — 격자 {n_tot}개 중 최댓값을 고르는 "
            f"최적화에서 단순 덧셈은 상위 후보를 모두 100%로 만들어 구별할 수 "
            f"없게 만듭니다.")

    show = T.rename(columns={
        "ionizable": "이온화(%)", "helper": "헬퍼(%)", "chol": "콜레스테롤(%)",
        "peg": "PEG(%)", "pred_ee": "예측 EE(%)", "pred_sd": "±불확실성",
        "delta_vs_template": "기준 대비", "rank_note": "순위 해석",
        "pred_ee_lab": "보정 전"})
    cols = ["이온화(%)", "헬퍼(%)", "콜레스테롤(%)", "PEG(%)", "예측 EE(%)"]
    if off:
        cols.append("보정 전")
    cols += ["±불확실성", "기준 대비", "순위 해석"]
    st.dataframe(show[cols], hide_index=True)

    st.info(
        f"**실제로 읽어야 할 결론: PEG 비율입니다.** 상위 후보의 PEG "
        f"중앙값은 {T.peg.median():.2f}% 입니다. 네 성분 중 PEG 만 "
        f"음성대조군 대비 뚜렷한 신호(2.7배)를 보였고, 실측에서도 PEG–EE "
        f"상관이 rho=-0.35 (p=7e-17) 로 확인됩니다.")
    st.download_button("후보 조성 CSV 내려받기",
                       T.to_csv(index=False).encode("utf-8-sig"),
                       "lnp_optimize_candidates.csv", "text/csv",
                       key="oa_dl")


def tab_whatif_anchored(st, df, model, v3_module, O):
    """What-If 탭 — Δ 는 영점과 무관하고 전·후 절대값만 옮깁니다."""
    st.subheader("기존 처방의 비율을 바꾸면 EE 가 어떻게 될까")
    off = OB.banner(st, context="delta")

    st.error(
        "**이 탭의 개별 예측은 신뢰할 수 없습니다.** 지질 종류·cargo·공정을 "
        "고정하고 비율만 바꾼 231쌍에서 증감 방향 적중률이 44.2% 였습니다 — "
        "무작위(50%)보다 낮습니다. `유의` 판정이 나온 경우만 읽으십시오"
        "(231쌍 중 9쌍, 그 9쌍은 방향 9/9 적중, p=0.004).")

    row = st.selectbox(
        "바꿀 처방", options=list(range(len(df))), key="wa_row",
        format_func=lambda i: (
            f"[{i}] {df.get('lipid_molar_ratio', pd.Series(['?']*len(df))).iloc[i]}"
            f"  실측 EE "
            f"{pd.to_numeric(df.get('encapsulation_efficiency_percent_std_num', pd.Series([np.nan]*len(df))), errors='coerce').iloc[i]}"))
    cur = str(df.get("lipid_molar_ratio",
                     pd.Series(["50:10:38.5:1.5"]*len(df))).iloc[int(row)])
    new = st.text_input("새 비율 (이온화 : 헬퍼 : 콜레스테롤 : PEG)",
                        value=cur, key="wa_new")

    if not st.button("예측", type="primary", key="wa_run"):
        return
    try:
        r = O.what_if(df, model, v3_module, int(row), new)
    except Exception as e:
        st.error(f"비율을 읽을 수 없습니다: {e}")
        return

    # 전·후를 따로 자르면 둘 다 100 이 되어 Δ 가 사라집니다(측정: 영점
    # +11.9 %p 에서 93.1→84.8 이 100→100 이 됐습니다). 두 값을 **함께**
    # 여유 비례로 보정해 Δ 를 보존합니다.
    ref = OB.reference_prediction(st)
    both, eff = OB.apply_offset_headroom(
        [r["pred_before"], r["pred_after"]], off, ref=ref)
    pb, pa = float(both.iloc[0]), float(both.iloc[1])
    delta_adj = pa - pb
    cb = ca = 0
    c1, c2, c3 = st.columns(3)
    c1.metric("문헌 실측 EE",
              "없음" if r["measured_ee"] is None else f"{r['measured_ee']:.1f} %")
    c2.metric("변경 전 예측" + (" (영점 적용)" if off else ""), f"{pb:.1f} %")
    c3.metric("변경 후 예측" + (" (영점 적용)" if off else ""), f"{pa:.1f} %",
              delta=f"{(delta_adj if off else r['delta']):+.1f} %p")

    st.markdown(f"- 조성: `{r['ratio_before']}` → `{r['ratio_after']}`")
    st.markdown(f"- 영점 없는 예측 변화량 **{r['delta']:+.1f} %p**, "
                f"불확실성 ±{r['delta_sd']:.1f} %p")
    if off:
        st.caption(
            f"영점 {off:+.1f} %p 를 상한 여유에 비례해 적용했습니다 "
            f"(실효 {eff:+.1f} %p, 기준 예측 {ref:.1f}%). 이 방식은 예측이 "
            f"100%에 가까울수록 보정을 줄이므로 **변화량이 "
            f"{r['delta']:+.1f} → {delta_adj:+.1f} %p 로 압축됩니다.** "
            f"단순 덧셈은 Δ 를 정확히 보존하지만 전·후가 모두 100%에서 잘려 "
            f"Δ 가 0 이 되는 경우가 있어 쓰지 않았습니다. "
            f"**유의성 판정은 영점 없는 Δ 로 하십시오** — 아래 판정이 "
            f"그 값입니다.")

    if r["significant"]:
        st.success(
            f"**유의 — 읽을 가치가 있습니다 ({r['verdict']}).** 예측 변화가 "
            f"불확실성을 넘었습니다. 검증에서 이 조건에 해당한 9쌍은 방향을 "
            f"모두 맞혔습니다(p=0.004). 단 표본이 9쌍이므로 확정된 규칙은 "
            f"아닙니다.")
    else:
        st.warning(
            f"**유의하지 않습니다 — 근거로 쓰지 마십시오.** 변화량 "
            f"{abs(r['delta']):.1f} %p 가 불확실성 ±{r['delta_sd']:.1f} %p "
            f"이내입니다. 이 구간의 방향 적중률은 41.9% 로 무작위보다 "
            f"낮았습니다.")


def tab_peg_anchored(st, df, peg_module):
    """PEG 탭 — 곡선 높이에 영점을 적용합니다. 모양은 불변입니다."""
    off = OB.banner(st, context="absolute")
    if off:
        st.caption(
            f"아래 예측 절대값에 영점 {off:+.1f} %p 가 적용됩니다. "
            f"**PEG 변화에 따른 곡선 모양과 기울기는 바뀌지 않습니다.**")
    # 원 모듈은 내부에서 직접 그리므로, 영점을 세션에 남겨 두고 그대로 호출합니다.
    # peg_module 이 offset 인자를 받도록 확장되면 그 인자로 넘기십시오.
    try:
        return peg_module.tab_peg(st, df, offset=off)
    except TypeError:
        if off:
            st.warning(
                "PEG 모듈이 아직 영점 인자를 받지 않습니다 — 아래 곡선은 "
                f"문헌 평균 기준입니다. 여러분 랩 기준으로 읽으려면 표시된 "
                f"값에 {off:+.1f} %p 를 더하십시오 (100% 상한 유의).")
        return peg_module.tab_peg(st, df)
