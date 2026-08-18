"""app_tab_harvest — 자동 데이터 수집 탭 (Streamlit)

app.py 에 다음을 넣으면 탭이 생깁니다.

    import app_tab_harvest as TH
    with tabs[-1]:
        TH.render(work_df, on_add=add_rows)

on_add 는 지금 쓰고 있는 저장 함수를 그대로 넘기십시오. 이 모듈은
데이터를 직접 저장하지 않습니다 — 사람이 표를 보고 승인한 뒤에만
on_add 가 호출됩니다.

설계 원칙은 하나입니다. **자동 수집은 사람의 승인 없이 저장하지 않습니다.**
7차 실측에서 LLM 1차 추출의 몰비 14%가 원문에 없는 값이었고, 원문 대조
게이트를 통과한 뒤에도 cargo 오분류 25건, 제타/pKa 혼동 11건이 남았습니다.
게이트는 명백한 환각을 막지만 사람의 눈을 대신하지는 못합니다.

LLM 접근이 없는 배포(Streamlit Cloud 등)에서는 이 탭이 실행 버튼 대신
안내를 표시합니다 — 수집은 로컬에서 돌리고 결과 CSV 를 업로드하는 경로를
권합니다.
"""
from __future__ import annotations

import io
import pandas as pd
import streamlit as st

import lnp_autoharvest as AH

# 검색어 프리셋 — 6차에서 실제로 수확이 있었던 조합입니다
PRESETS = {
    "mRNA LNP (기본)":
        "lipid nanoparticle mRNA encapsulation efficiency molar ratio",
    "이온화지질 신규 합성":
        "ionizable lipid synthesis library screening lipid nanoparticle encapsulation",
    "siRNA / ASO":
        "lipid nanoparticle siRNA antisense encapsulation efficiency composition",
    "백신 제형":
        "mRNA vaccine lipid nanoparticle formulation DSPC cholesterol PEG molar",
    "공정 변수(N/P·pH)":
        "lipid nanoparticle N/P ratio buffer pH encapsulation efficiency microfluidic",
}


def _llm_available() -> bool:
    try:
        host  # noqa: F821 — 앱 런타임에 주입되는 객체
        return True
    except NameError:
        return False


def render(work_df: pd.DataFrame, on_add=None, llm=None) -> None:
    st.subheader("논문 자동 수집")
    st.caption(
        "PMC 공개 논문에서 처방·EE를 추출하고 **원문과 대조해 검증한 행만** "
        "후보로 올립니다. 저장은 아래 표를 확인하신 뒤 버튼을 누를 때만 됩니다."
    )

    if llm is None and not _llm_available():
        st.info(
            "이 배포에는 추출 모델 접근이 없습니다. 수집은 로컬에서 "
            "`lnp_autoharvest` 로 돌리고, 결과 CSV를 '데이터 추가' 탭에 "
            "업로드하십시오. 검증 게이트는 그 CSV 에도 이미 적용돼 있습니다."
        )
        with st.expander("로컬 실행 방법"):
            st.code(
                "import lnp_autoharvest as AH\n"
                "job = AH.HarvestJob(existing_df=df, llm=host.llm, target_rows=40)\n"
                "for ev in job.run(max_papers=150):\n"
                "    print(f'{ev.frac:.0%} {ev.message}')\n"
                "job.accepted.to_csv('new_rows.csv', index=False)",
                language="python")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        preset = st.selectbox("검색 주제", list(PRESETS), index=0)
    with c2:
        max_papers = st.number_input("검토 논문 수", 20, 400, 120, step=20,
                                     help="6차 실측 채택률로는 100행에 약 300편이 필요합니다")
    with c3:
        target = st.number_input("목표 행 수", 10, 300, 40, step=10)

    with st.expander("배제 기준", expanded=True):
        d1, d2 = st.columns(2)
        with d1:
            req_named = st.checkbox(
                "이름이 불분명한 이온화지질 배제", value=True,
                help="논문이 이름을 밝히지 않은 처방(Custom lipid)을 제외합니다")
            req_exact = st.checkbox(
                "EE 근사표현 배제", value=False,
                help="'90% 이상' 같은 표현을 제외합니다. 6차 실측에서 이 조건을 "
                     "걸면 채택률이 61%→32%로 떨어지지만 MAE는 15.79→15.23으로 좋아집니다")
        with d2:
            st.markdown(
                "**항상 적용되는 검사** (끌 수 없음)\n"
                "- EE 숫자가 원문에 있어야 함\n"
                "- 몰비 값이 원문의 몰비와 집합 일치 (순서 변경·5성분 부분집합 허용, "
                "값 변경은 기각)\n"
                "- 지질명이 원문에 있어야 함 (표기 변형 허용)\n"
                "- 헬퍼·PEG·화물·약물이 이온화지질 칸에 오면 기각\n"
                "- 값 범위: EE 0-100, 크기 20-500nm, PDI 0-1, ζ ±60mV, N/P 0.5-50, pH 3-9\n"
                "- 기존 데이터·수집분 내부 중복 제거")

    if st.button("수집 시작", type="primary"):
        job = AH.HarvestJob(
            existing_df=work_df, llm=llm or host.llm,  # noqa: F821
            target_rows=int(target), require_named=req_named,
            require_exact_ee=req_exact)
        bar = st.progress(0.0, "시작")
        try:
            for ev in job.run(queries=[PRESETS[preset]], max_papers=int(max_papers)):
                bar.progress(min(ev.frac, 1.0), ev.message)
        except Exception as e:
            st.error(f"수집 중 오류: {type(e).__name__}: {e}")
            return
        st.session_state["harvest_result"] = {
            "accepted": job.accepted, "rejected": job.rejected, "stats": job.stats}

    res = st.session_state.get("harvest_result")
    if not res:
        return

    acc, rej, stats = res["accepted"], res["rejected"], res["stats"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("검토 논문", stats.get("papers_seen", 0))
    m2.metric("검증 통과", len(acc))
    m3.metric("기각", len(rej))
    if len(acc) + len(rej) > 0:
        m4.metric("채택률", f"{len(acc)/(len(acc)+len(rej))*100:.0f}%")

    if len(acc) == 0:
        st.warning("검증을 통과한 행이 없습니다. 검색 주제를 바꾸거나 "
                   "검토 논문 수를 늘려 보십시오.")
        if len(rej):
            with st.expander(f"기각 사유 {len(rej)}건"):
                st.dataframe(rej["why"].value_counts().rename("건수"),
                             use_container_width=True)
        return

    st.markdown("#### 검증을 통과한 후보 — 확인 후 저장하십시오")
    st.caption("`evidence` 열에 EE 를 담은 원문 인용문이 있습니다. "
               "`pmcid` 로 원문을 바로 찾을 수 있습니다.")
    show = ["ionizable_lipid_name", "lipid_molar_ratio",
            "encapsulation_efficiency_percent_std_num", "cargo_type",
            "confidence", "ee_is_approximate", "pmcid", "evidence"]
    edited = st.data_editor(
        acc[[c for c in show if c in acc.columns]].assign(저장=True),
        use_container_width=True, hide_index=True, height=340,
        column_config={"저장": st.column_config.CheckboxColumn(
            "저장", help="체크를 해제하면 이 행은 저장하지 않습니다")})

    keep_mask = edited["저장"].fillna(False).values
    n_keep = int(keep_mask.sum())
    cA, cB = st.columns([1, 3])
    with cA:
        if st.button(f"선택한 {n_keep}행 저장", disabled=(n_keep == 0)):
            if on_add is None:
                st.error("on_add 가 연결되지 않았습니다.")
            else:
                added = acc.loc[keep_mask].reset_index(drop=True)
                on_add(added)
                st.success(f"{len(added)}행 저장했습니다. "
                           f"모델은 다음 예측에서 이 데이터를 사용합니다.")
                st.session_state.pop("harvest_result", None)
                st.rerun()
    with cB:
        buf = io.StringIO()
        acc.to_csv(buf, index=False)
        st.download_button("후보 전체 CSV 내려받기", buf.getvalue(),
                           file_name="harvest_candidates.csv", mime="text/csv")

    if len(rej):
        with st.expander(f"기각된 {len(rej)}행 — 사유별"):
            st.dataframe(rej["why"].value_counts().rename("건수"),
                         use_container_width=True)
            st.caption(
                "'몰비가 원문에 없음'은 모델이 값을 만들어낸 경우입니다. "
                "6차 실측에서 추출 227행 중 32행이 이 사유로 기각됐습니다 — "
                "이 검사가 없으면 그 값들이 데이터베이스에 들어갑니다.")
