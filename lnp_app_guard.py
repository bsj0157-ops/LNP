# -*- coding: utf-8 -*-
"""입력 시점 방어 — 중복·불량 행이 데이터베이스에 들어가는 것을 막습니다.

현재 앱의 문제와 실측:

1) **중복 검사가 없습니다.** 앱은 추가 성공 시 "중복 제거 후 안전하게
   추가되었습니다"라고 표시하지만 실제 중복 검사 코드는 없습니다.
   결합 데이터 819행 중 **116행이 중복**입니다(같은 논문·같은 이온화지질·
   같은 몰비·같은 EE). 한 논문이 17행으로 부풀어 있습니다.

   모델 정확도에는 거의 영향이 없습니다(work_df 정제 단계가 대부분 흡수해
   682행 vs 681행). 그러나 데이터베이스 신뢰도, 논문 편수 집계, 앵커링의
   "이 논문의 행 개수" 판정은 모두 틀어집니다.

2) **guard 가 행을 조용히 버립니다.** 5성분 축약이 불가능한 행을 반환에서
   제외하는데, 사용자는 몇 행이 왜 빠졌는지 알 수 없습니다.

3) 중복 판정 키는 몰비의 **값 집합**으로 잡습니다. `50:10:38.5:1.5` 와
   `50:38.5:10:1.5` 는 성분 순서만 다른 같은 처방이므로 같은 키가 됩니다.
   자동 수집 모듈에서 같은 이유로 순서 중복이 걸렸던 사례가 있습니다.
"""
import numpy as np
import pandas as pd

DOI = "reference_doi"
EE = "encapsulation_efficiency_percent_std_num"
RATIO = "lipid_molar_ratio"
ION = "ionizable_lipid_name"


def row_key(row) -> tuple:
    """중복 판정 키 — 몰비는 값 집합으로 정규화해 성분 순서에 무관하게 합니다."""
    ee = pd.to_numeric(pd.Series([row.get(EE)]), errors="coerce").iloc[0]
    parts = sorted(p.strip() for p in str(row.get(RATIO, "")).split(":") if p.strip())
    return (str(row.get(DOI, "")).strip().lower(),
            str(row.get(ION, "")).strip().lower(),
            ":".join(parts),
            f"{ee:.1f}" if pd.notna(ee) else "")


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """기존 데이터 안의 중복을 찾습니다. 반환: 중복 그룹 요약표."""
    if df is None or not len(df):
        return pd.DataFrame()
    keys = pd.Series([row_key(r) for _, r in df.iterrows()], index=df.index)
    vc = keys.value_counts()
    dup = vc[vc > 1]
    rows = []
    for k, c in dup.items():
        idx = list(keys[keys == k].index)
        rows.append({"논문": k[0][:44], "이온화지질": k[1][:20],
                     "몰비(정렬)": k[2], "EE": k[3], "행 수": int(c),
                     "행 인덱스": idx[:8]})
    return pd.DataFrame(rows).sort_values("행 수", ascending=False)


def screen_new_rows(new: pd.DataFrame, existing: pd.DataFrame,
                    reduce_fn=None) -> dict:
    """추가하려는 행을 심사합니다. 무엇이 왜 빠졌는지 전부 보고합니다.

    reduce_fn: 5성분 축약 함수 (lnp_autoharvest.reduce_to_four). None 이면
               축약을 건너뜁니다.

    반환 dict
      accepted     : 추가할 행
      rejected     : 사유가 붙은 기각 행 (why 컬럼)
      n_dup_exist  : 기존 데이터와 중복
      n_dup_within : 추가 묶음 안에서의 중복
      n_reduced    : 5성분에서 4성분으로 축약된 행
      messages     : 사용자에게 보여줄 문장 리스트
    """
    if new is None or not len(new):
        return {"accepted": new, "rejected": pd.DataFrame(),
                "n_dup_exist": 0, "n_dup_within": 0, "n_reduced": 0,
                "messages": ["추가할 행이 없습니다."]}

    work = new.copy()
    rej, notes = [], []
    n_reduced = 0

    # 1) 5성분 축약 — 버리지 않고 사유를 남깁니다
    if reduce_fn is not None and RATIO in work.columns:
        fixed, keep = [], []
        for i, s in zip(work.index, work[RATIO].astype(str)):
            g, note = reduce_fn(s)
            if not g:
                r = work.loc[i].to_dict()
                r["why"] = f"몰비 축약 불가: {note or s}"
                rej.append(r)
                keep.append(False)
                fixed.append("")
            else:
                if note:
                    n_reduced += 1
                    notes.append((i, note))
                keep.append(True)
                fixed.append(g)
        work = work.assign(**{RATIO: fixed})
        if notes:
            base = (work.get("repair_note", pd.Series("", index=work.index))
                    .fillna("").astype(str))
            for i, note in notes:
                base.loc[i] = (base.loc[i] + " | " + note) if base.loc[i] else note
            work = work.assign(repair_note=base)
        work = work[pd.Series(keep, index=work.index)]

    # 2) 기존 데이터와의 중복
    ex_keys = set()
    if existing is not None and len(existing):
        ex_keys = {row_key(r) for _, r in existing.iterrows()}
    n_dup_exist = n_dup_within = 0
    seen, keep2 = set(), []
    for i, r in work.iterrows():
        k = row_key(r)
        if k in ex_keys:
            d = r.to_dict(); d["why"] = "기존 데이터와 중복"
            rej.append(d); keep2.append(False); n_dup_exist += 1
        elif k in seen:
            d = r.to_dict(); d["why"] = "추가 묶음 안에서 중복"
            rej.append(d); keep2.append(False); n_dup_within += 1
        else:
            seen.add(k); keep2.append(True)
    accepted = work[pd.Series(keep2, index=work.index)] if len(work) else work

    msgs = [f"추가 요청 {len(new)}행 → 채택 {len(accepted)}행"]
    if n_dup_exist:
        msgs.append(f"기존 데이터와 중복 {n_dup_exist}행을 제외했습니다.")
    if n_dup_within:
        msgs.append(f"추가 묶음 안의 중복 {n_dup_within}행을 제외했습니다.")
    if n_reduced:
        msgs.append(f"5성분 이상 몰비 {n_reduced}행을 4성분으로 축약했습니다 "
                    "(repair_note 에 기록).")
    n_cut = len(new) - len(accepted) - n_dup_exist - n_dup_within
    if n_cut > 0:
        msgs.append(f"몰비 축약 불가 {n_cut}행을 제외했습니다.")

    return {"accepted": accepted,
            "rejected": pd.DataFrame(rej) if rej else pd.DataFrame(),
            "n_dup_exist": n_dup_exist, "n_dup_within": n_dup_within,
            "n_reduced": n_reduced, "messages": msgs}
