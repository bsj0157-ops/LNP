# -*- coding: utf-8 -*-
"""배포된 LNP Data Studio 의 결함 수정 — 전부 실데이터로 측정한 것만 담았습니다.

Atlas 628행 + web 680행 = 1308행으로 배포 앱과 같은 조건에서 측정했습니다.

--------------------------------------------------------------------------
[치명] 1. 탭마다 다른 프레임을 쓰기 때문에 '행 번호'가 서로 다른 처방을 가리킵니다
--------------------------------------------------------------------------
    데이터 관리 탭·앵커링 탭  : st.session_state.df        1308행 (중복 포함)
    탭6 PEG                  : P.dedupe(df)                553행

    같은 위치를 비교하면 553행 중 **548행(99%)** 이 다른 처방입니다.
    5번 행부터 이미 어긋납니다 (df[5] EE=69.0 vs clean_df[5] EE=72.0).

    사용자는 데이터 관리 탭 표에서 본 번호를 앵커링 탭에 입력합니다.
    그러면 다른 처방의 EE 를 앵커로 넣게 됩니다. 실측 결과:

        올바른 앵커 → MAE  6.15 %p
        어긋난 앵커 → MAE 10.97 %p      (78% 악화, 예측 전체가 9.9 %p 이동)

    앵커링은 오차를 31% 줄이는 기능인데, 이 버그가 그 이득을 전부 삼키고
    반대 방향으로 넘깁니다. 게다가 오류 메시지가 없어 사용자는 모릅니다.

    -> 해결: 앱 전체가 **하나의 정규화된 프레임**을 쓰게 합니다.
       get_working_df() 를 한 번만 만들어 모든 탭에 넘기십시오.
       행 번호 대신 안정적인 라벨(DOI + 몰비)로 선택하게 하는 것이 더 낫습니다.

--------------------------------------------------------------------------
[치명] 2. 모듈 임포트가 데모 학습을 실행합니다
--------------------------------------------------------------------------
    lnp_predictor_v3_patched.py 파일 끝에 __main__ 가드 없이
    `results = run_all()` 이 있습니다. `import` 만으로 합성 데이터로
    모델 3개를 교차검증하고 lnp_v3_report.png 를 씁니다.

    측정: 임포트에 약 140초. 실제 특징 생성은 3.0초입니다.

    Streamlit Community Cloud 는 슬립 후 첫 방문·재배포 때 앱을
    재시작합니다. 그때마다 방문자가 140초를 기다립니다.

    -> 해결: 그 파일 마지막 줄을 다음으로 바꾸십시오.
           if __name__ == "__main__":
               results = run_all()

--------------------------------------------------------------------------
[중대] 3. 저장한 데이터가 사라지고, 방문자끼리 섞입니다
--------------------------------------------------------------------------
    save_disk() 는 앱 컨테이너에 lnp_web_data.csv 를 씁니다.
    Community Cloud 컨테이너는 재시작 때 초기화되므로 입력한 데이터가
    사라집니다. 더 중요한 것은 **방문자 전원이 같은 파일을 공유**한다는
    점입니다 — A 가 넣은 데이터가 B 에게 보이고, B 의 '교체' 버튼이
    A 의 데이터를 지웁니다.

    -> 해결: 아래 세 가지 중 하나.
       (a) 읽기 전용 데모로 만들고, 데이터는 저장소에 커밋한 CSV 를 씁니다.
           사용자는 CSV 를 내려받아 각자 보관합니다. (가장 간단)
       (b) 세션별 격리: st.session_state 에만 두고 save_disk() 를 없앱니다.
           브라우저를 닫으면 사라지지만 남의 데이터를 망치지 않습니다.
       (c) 외부 저장소(st.secrets + Google Sheets / S3)를 씁니다.
    show_persistence_warning() 이 사용자에게 이 사실을 알립니다.

--------------------------------------------------------------------------
[중대] 4. 탭4가 보고하는 성능은 탭5·6이 쓰는 모델의 성능이 아닙니다
--------------------------------------------------------------------------
    탭4는 v3.build_features 를 쓰지 않고 자체 파싱을 합니다(특징 9개).
    탭5·6은 v3.build_features(특징 25개, SMILES 화학특징 포함)를 씁니다.

    같은 논문 단위 CV 로 측정하면 (553행 / dedupe 후):
        탭4 자체 특징 (9개)   MAE 17.07 %p   baseline 대비 +4.0%
        v3.build_features(25) MAE 16.43 %p   baseline 대비 +7.6%
        baseline (평균)       MAE 17.78 %p

    사용자가 탭4에서 본 숫자로 탭5·6의 결과를 해석하게 됩니다.
    게다가 열등한 특징집합의 성능을 보고 있습니다.

    -> 해결: 탭4도 v3.build_features 를 쓰십시오 (build_eval_matrix 참고).

--------------------------------------------------------------------------
[중대] 5. 중복 제거가 탭마다 다릅니다
--------------------------------------------------------------------------
    1308행 중 754행(58%)이 중복입니다. dedupe 적용 현황:
        적용:  탭3 CSV 업로드, 탭6 PEG
        누락:  탭4 모델 실행, 탭5 최적화, 앵커링, PDF 추가, 직접 입력
    중복이 남으면 논문 단위 CV 에서도 같은 처방이 학습·평가에 함께 들어가
    성능이 부풀려집니다.

--------------------------------------------------------------------------
[보통] 6. 앵커링 탭에 EE 범위 검사가 없습니다
--------------------------------------------------------------------------
    탭4·탭5는 0<EE<=100 을 확인하고 분수 표기(0~1)를 100배 합니다.
    앵커링 탭은 df[EEC] 를 그대로 fit 에 넣습니다.
    현 데이터에도 EE<=0 인 행이 4개 있어 학습에 섞입니다.

--------------------------------------------------------------------------
[보통] 7. 앵커 실측값 0% 를 입력할 수 없습니다
--------------------------------------------------------------------------
    `if anchor_1_ee > 0 and ...` 이므로 캡슐화가 실패해 EE=0% 인 실험을
    앵커로 쓸 수 없습니다. 기본값도 0.0 이라 '미입력'과 '0% 측정'이
    구분되지 않습니다. -> value=None 과 체크박스로 분리하십시오.

--------------------------------------------------------------------------
[보통] 8. 탭5 최적화가 렌더링마다 모델을 재학습합니다
--------------------------------------------------------------------------
    `with tab_opt:` 안에서 get_base_pipeline(df) 를 호출합니다.
    Streamlit 은 위젯을 건드릴 때마다 스크립트 전체를 다시 실행하므로,
    슬라이더를 한 칸 옮길 때마다 특징 생성 3.0초 + RF 학습 1.6초 = 4.5초
    를 다시 냅니다. 탭을 보고 있지 않아도 실행됩니다.
    -> @st.cache_resource 로 감싸십시오 (cached_base_model 참고).
"""
import hashlib

import numpy as np
import pandas as pd

EE_COL = "encapsulation_efficiency_percent_std_num"
DOI_COL = "reference_doi"
RATIO_COL = "lipid_molar_ratio"


# ==========================================================================
# 1. 모든 탭이 공유하는 단일 프레임
# ==========================================================================
def normalize_ee(s):
    """EE 를 0~100 스케일 숫자로 정규화합니다.

    분수 표기(0~1)를 100배 하고, 범위를 벗어난 값은 NaN 으로 둡니다.
    탭4·탭5에는 이 처리가 있었지만 앵커링 탭에는 없어 EE<=0 인 4행이
    학습에 섞였습니다.
    """
    y = pd.to_numeric(s, errors="coerce")
    frac = (y > 0) & (y <= 1)
    y = y.where(~frac, y * 100)
    return y.where((y > 0) & (y <= 100))


def get_working_df(df, patch_mod=None, verbose=False):
    """앱 전체가 쓸 정규화된 프레임 하나를 만듭니다.

    중복 제거 + EE 정규화 + 인덱스 리셋을 한 곳에서 처리합니다.
    **모든 탭이 이 함수의 결과만 쓰게 하십시오.** 탭마다 다른 프레임을
    쓰면 행 번호가 서로 다른 처방을 가리킵니다(측정: 99% 불일치,
    앵커링 MAE 6.15 -> 10.97 %p).

    반환: (working_df, info) — info 에 몇 행이 왜 빠졌는지 담깁니다.
    """
    info = {"입력": len(df)}
    d = df.copy()

    if patch_mod is not None and hasattr(patch_mod, "dedupe"):
        d = patch_mod.dedupe(d, verbose=verbose)
    info["중복 제거 후"] = len(d)

    if EE_COL in d.columns:
        d[EE_COL] = normalize_ee(d[EE_COL])
        before = len(d)
        d = d[d[EE_COL].notna()]
        info["EE 범위 밖 제외"] = before - len(d)

    d = d.reset_index(drop=True)
    info["최종"] = len(d)
    if verbose:
        print(" · ".join(f"{k} {v}" for k, v in info.items()))
    return d, info


def row_label(df, i, maxlen=64):
    """행 번호 대신 쓸 안정적인 라벨. DOI + 몰비 + EE.

    행 번호는 프레임이 바뀌면 다른 처방을 가리킵니다. 라벨은 내용에
    묶여 있으므로 어느 프레임에서든 같은 처방을 뜻합니다.
    """
    r = df.iloc[i]
    doi = str(r.get(DOI_COL, "?"))
    doi = doi[-28:] if len(doi) > 28 else doi
    ee = pd.to_numeric(pd.Series([r.get(EE_COL)]), errors="coerce").iloc[0]
    ee_s = f"{ee:.1f}%" if pd.notna(ee) else "EE?"
    return f"[{i}] {r.get(RATIO_COL, '?')} · {ee_s} · {doi}"[:maxlen]


def df_fingerprint(df):
    """프레임 내용 해시. 두 탭이 같은 데이터를 보고 있는지 확인용."""
    cols = [c for c in (DOI_COL, RATIO_COL, EE_COL) if c in df.columns]
    if not cols:
        return "?"
    s = df[cols].astype(str).agg("|".join, axis=1).str.cat(sep="\n")
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


# ==========================================================================
# 2. 탭4를 v3 특징으로 통일
# ==========================================================================
def build_eval_matrix(work_df, v3_module):
    """탭4의 자체 파싱을 v3.build_features 로 교체합니다.

    측정 결과 v3 특징이 더 정확합니다(MAE 16.43 vs 17.07 %p).
    무엇보다 탭5·6이 쓰는 것과 같은 특징이라, 탭4가 보고하는 성능이
    실제로 다른 탭의 성능을 뜻하게 됩니다.

    반환: (X, y, groups, num_cols, cat_cols)
    """
    y = normalize_ee(work_df[EE_COL])
    keep = y.notna()
    d = work_df[keep].reset_index(drop=True)
    y = y[keep].reset_index(drop=True)
    X, num_cols, cat_cols = v3_module.build_features(d, include_measured=False)
    g = d[DOI_COL].astype(str).str.strip().str.lower()
    return X, y, g, num_cols, cat_cols


# ==========================================================================
# 3. 캐싱 — 렌더링마다 재학습 방지
# ==========================================================================
def make_cached_base_model(st, v3_module):
    """@st.cache_resource 로 감싼 모델 학습 함수를 돌려줍니다.

    측정: 특징 생성 3.0초 + RF 학습 1.6초 = 렌더링마다 4.5초.
    Streamlit 은 위젯 조작마다 스크립트 전체를 재실행하므로 캐시가
    없으면 슬라이더를 옮길 때마다 이 비용을 냅니다.

    사용:
        cached_model = make_cached_base_model(st, v3)
        base_model = cached_model(work_df)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # 프레임을 JSON 으로 왕복시키지 않습니다. 결측 표기(nan/None)가 바뀌고,
    # 직렬화 자체가 불필요한 실패 지점입니다. 캐시 키는 지문 문자열만 쓰고
    # 프레임은 _X/_y 인자로 직접 넘깁니다(밑줄 이름은 Streamlit 이 해시하지
    # 않으므로 해시 불가 객체를 안전하게 전달할 수 있습니다).
    @st.cache_resource(show_spinner="모델 학습 중... (데이터가 바뀔 때만 실행됩니다)")
    def _fit(fingerprint, num_cols, cat_cols, _X, _y):
        X, y = _X, _y
        num_cols, cat_cols = list(num_cols), list(cat_cols)
        pre = ColumnTransformer(
            [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler())]), num_cols),
             ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                             ("o", OneHotEncoder(handle_unknown="ignore",
                                                 min_frequency=2))]), cat_cols)]
            if cat_cols else
            [("n", Pipeline([("i", SimpleImputer(strategy="median")),
                             ("s", StandardScaler())]), num_cols)])
        m = Pipeline([("pre", pre),
                      ("m", RandomForestRegressor(
                          n_estimators=400, min_samples_leaf=3,
                          max_features=0.5, random_state=42, n_jobs=-1))])
        m.fit(X, y)
        return m

    # build_eval_matrix 자체도 캐시합니다. 측정: 특징 생성 3.0초.
    # 캐시하지 않으면 학습만 캐시돼도 렌더링마다 3초를 계속 냅니다.
    @st.cache_data(show_spinner="특징 계산 중...")
    def _feats(fingerprint, _df):
        X, y, g, nc, cc = build_eval_matrix(_df, v3_module)
        return X, y, g, tuple(nc), tuple(cc)

    def cached_features(work_df):
        """(X, y, groups, num_cols, cat_cols) — 캐시됩니다."""
        return _feats(df_fingerprint(work_df), work_df)

    def cached_base_model(work_df):
        fp = df_fingerprint(work_df)
        X, y, _, nc, cc = _feats(fp, work_df)
        return _fit(fp, nc, cc, X, y)

    cached_base_model.features = cached_features
    return cached_base_model


# ==========================================================================
# 4. 사용자에게 알려야 하는 것들
# ==========================================================================
def show_persistence_warning(st):
    """배포 환경의 데이터 저장 한계를 사용자에게 알립니다."""
    st.warning(
        "**입력한 데이터는 영구 저장되지 않습니다.** 이 앱은 Streamlit "
        "Community Cloud 컨테이너의 파일에 저장하는데, 컨테이너는 "
        "재시작·슬립 복귀·재배포 때 초기화됩니다. 또한 방문자 전원이 같은 "
        "파일을 공유하므로 다른 사람의 편집이 서로에게 영향을 줍니다. "
        "**작업한 데이터는 사이드바의 '전체 CSV 내려받기'로 반드시 "
        "직접 보관하십시오.**")


def show_data_consistency(st, work_df, raw_df):
    """모든 탭이 같은 프레임을 쓰고 있음을 표시합니다."""
    c1, c2, c3 = st.columns(3)
    c1.metric("사용 중인 데이터", f"{len(work_df)} 행",
              delta=f"{len(work_df) - len(raw_df)} (중복·범위밖 제외)")
    c2.metric("논문 수 (실질 표본)",
              f"{work_df[DOI_COL].astype(str).str.strip().str.lower().nunique()} 편"
              if DOI_COL in work_df else "?")
    c3.metric("데이터 지문", df_fingerprint(work_df),
              help="모든 탭이 이 값을 씁니다. 탭마다 다르면 행 번호가 "
                   "서로 다른 처방을 가리킵니다.")


def anchor_selector(st, work_df, n=3, key_prefix="anc"):
    """앵커를 **행 번호 입력이 아니라 선택 목록**으로 고르게 합니다.

    행 번호 입력이 이 앱의 가장 심각한 결함입니다. 사용자는 데이터 관리
    탭에서 본 번호를 앵커링 탭에 적는데, 두 탭이 다른 프레임을 쓰면
    다른 처방을 가리킵니다(측정: 553행 중 548행 불일치).

    선택 목록은 처방 내용(몰비·EE·DOI)을 보여주므로 사용자가 무엇을
    고르는지 확인할 수 있고, 인덱스는 코드가 직접 얻습니다.

    반환: (idx_list, y_list) — 유효한 앵커만 담깁니다. 비어 있으면 앵커링
    을 건너뛰십시오.
    """
    labels = [row_label(work_df, i) for i in range(len(work_df))]
    idx_list, y_list = [], []
    st.caption(f"데이터 지문 `{df_fingerprint(work_df)}` — 모든 탭이 같은 "
               f"{len(work_df)}행을 씁니다.")
    for k in range(n):
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox(f"앵커 {k+1} 처방", ["(사용 안 함)"] + labels,
                            key=f"{key_prefix}_sel_{k}")
        if pick == "(사용 안 함)":
            continue
        i = labels.index(pick)
        # 이 랩에서 실제로 측정한 EE. 데이터의 EE 를 기본값으로 제시하되
        # 사용자가 자기 실측치로 덮어쓸 수 있게 합니다.
        default = float(pd.to_numeric(
            pd.Series([work_df.iloc[i].get(EE_COL)]), errors="coerce").iloc[0] or 0.0)
        # value=None 로 두어 '미입력'과 '0% 측정'을 구분합니다.
        # 원래 코드의 `if ee > 0` 은 캡슐화 실패(EE=0%)를 입력 불가로
        # 만들었습니다.
        ee = c2.number_input(f"실측 EE {k+1} (%)", min_value=0.0, max_value=100.0,
                             value=default, step=0.1, key=f"{key_prefix}_ee_{k}")
        use = st.checkbox(f"앵커 {k+1} 사용", value=True, key=f"{key_prefix}_use_{k}")
        if use:
            idx_list.append(i)
            y_list.append(float(ee))
    return idx_list, y_list


ACCURACY_NOTE = """**이 앱의 예측 정확도 — 측정값입니다**

| 항목 | 값 |
|---|---|
| 논문 단위 CV MAE | 16.4 %p (baseline 17.8 %p, 개선 7.6%) |
| 논문 간 분산 비중 (ICC) | 0.40 — EE 변동의 40%가 조성이 아니라 '어느 논문인지'에서 옵니다 |
| 앵커링 3개 적용 시 | 오차 21.6% 감소 (논문 22편 홀드아웃, 17.19 → 13.49 %p) |
| PEG 방향 (≥2.5%, 실측 데이터) | 65.7% 적중 (n=722 쌍) — PEG↑ 면 EE↓ |
| PEG 방향 (모델 예측, 한 논문 제외 시) | 52~56% — 무작위 수준 |
| 비율만 바꾼 what-if | 무작위 수준. 방향 판단에 쓰지 마십시오 |

**절대값 예측은 원리적으로 어렵습니다.** ICC 0.40 은 같은 조성이라도
논문(랩·프로토콜·측정법)에 따라 EE 가 크게 다르다는 뜻입니다. 그래서
새 논문의 EE 절대값을 맞히는 것보다, **앵커링으로 그 논문의 영점을
잡은 뒤 상대 비교**하는 쓰임이 맞습니다.
"""
