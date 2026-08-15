# -*- coding: utf-8 -*-
"""데이터를 컨테이너 밖에 저장합니다 — 코드를 수정해도 데이터가 남습니다.

왜 지금은 사라지는가
--------------------
    1. 사이트에서 입력 -> save_disk() -> 컨테이너의 lnp_web_data.csv
    2. GitHub 에서 코드 수정 -> Streamlit Cloud 가 재배포
    3. 재배포 = 새 컨테이너 = git 저장소 내용만 존재 -> 1번의 CSV 없음

컨테이너 파일시스템은 git 에 커밋되지 않습니다. 그래서 저장 위치가
컨테이너 **밖**이어야 합니다. 슬립 복귀 때도 같은 일이 일어납니다.

백엔드 셋을 같은 인터페이스로 제공합니다. `st.secrets` 에 설정된 것을
자동으로 고르고, 아무것도 없으면 기존처럼 로컬 파일을 씁니다(동작은
그대로, 대신 경고를 냅니다).

    from lnp_store import get_store
    store = get_store(st)
    df = store.load()          # 앱 시작 시
    store.save(df)             # 편집/추가 후

어느 것을 고를지
----------------
    gsheet : 본인과 소수 동료가 쓰고, 표를 브라우저에서 직접 보고
             싶을 때. 무료. Google 계정만 있으면 됩니다.
    github : 본인만 편집하고 **버전 이력**을 남기고 싶을 때. 커밋으로
             쌓이므로 되돌리기가 쉽습니다. 다만 커밋마다 재배포가
             일어날 수 있어 (아래 주의) 데이터 파일은 별도 저장소를
             쓰는 편이 안전합니다.
    supabase : 여러 사람이 동시에 편집할 때. 동시 쓰기가 안전한 것은
             이것뿐입니다.

주의: gsheet / github / s3 는 모두 '마지막에 저장한 사람이 이깁니다'.
두 사람이 동시에 편집하면 한쪽 편집이 조용히 사라집니다. 여러 사람이
쓸 계획이면 supabase 를 쓰십시오.
"""
import io
import json

import pandas as pd


# ==========================================================================
# 공통 인터페이스
# ==========================================================================
class BaseStore:
    name = "base"
    is_persistent = False

    def load(self):
        raise NotImplementedError

    def save(self, df):
        raise NotImplementedError

    def describe(self):
        return self.name


class LocalStore(BaseStore):
    """지금과 같은 동작 — 컨테이너 파일. 재배포 시 사라집니다."""
    name = "local"
    is_persistent = False

    def __init__(self, path="lnp_web_data.csv"):
        self.path = path

    def load(self):
        import os
        if os.path.exists(self.path):
            return pd.read_csv(self.path, encoding="utf-8-sig")
        return None

    def save(self, df):
        df.to_csv(self.path, index=False, encoding="utf-8-sig")

    def describe(self):
        return f"컨테이너 파일 ({self.path}) — 재배포·슬립 복귀 시 사라집니다"


# ==========================================================================
# Google Sheets
# ==========================================================================
class GSheetStore(BaseStore):
    """Google Sheets 한 장을 데이터베이스로 씁니다.

    설정 (한 번만):
      1. https://console.cloud.google.com 에서 프로젝트를 만들고
         'Google Sheets API' 와 'Google Drive API' 를 사용 설정합니다.
      2. 서비스 계정을 만들고 JSON 키를 내려받습니다.
      3. Google Sheets 에서 새 시트를 만들고, 서비스 계정 이메일
         (JSON 의 client_email)에 **편집자** 권한으로 공유합니다.
      4. Streamlit Cloud 앱 설정 -> Secrets 에 붙여넣습니다:

            [gsheet]
            sheet_id = "시트 URL 의 /d/ 와 /edit 사이 문자열"
            worksheet = "data"

            [gsheet.service_account]
            type = "service_account"
            project_id = "..."
            private_key_id = "..."
            private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
            client_email = "...@....iam.gserviceaccount.com"
            client_id = "..."
            token_uri = "https://oauth2.googleapis.com/token"

      5. requirements.txt 에 `gspread` 와 `google-auth` 를 추가합니다.

    private_key 는 줄바꿈이 \\n 으로 들어가야 합니다. 이것이 가장 흔한
    실패 원인입니다.
    """
    name = "gsheet"
    is_persistent = True

    def __init__(self, cfg):
        self.sheet_id = cfg["sheet_id"]
        self.worksheet = cfg.get("worksheet", "data")
        self._sa = dict(cfg["service_account"])
        self._ws = None

    def _connect(self):
        if self._ws is not None:
            return self._ws
        import gspread
        from google.oauth2.service_account import Credentials
        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(self._sa, scopes=scopes)
        sh = gspread.authorize(creds).open_by_key(self.sheet_id)
        try:
            self._ws = sh.worksheet(self.worksheet)
        except Exception:
            self._ws = sh.add_worksheet(self.worksheet, rows=1000, cols=40)
        return self._ws

    def load(self):
        ws = self._connect()
        rows = ws.get_all_values()
        if not rows or len(rows) < 2:
            return None
        df = pd.DataFrame(rows[1:], columns=rows[0])
        return df.replace("", pd.NA)

    def save(self, df):
        ws = self._connect()
        body = [list(df.columns)] + df.astype(object).where(
            df.notna(), "").astype(str).values.tolist()
        ws.clear()
        ws.update(body, "A1")

    def describe(self):
        return f"Google Sheets ({self.sheet_id[:12]}…/{self.worksheet})"


# ==========================================================================
# GitHub 저장소에 커밋
# ==========================================================================
class GitHubStore(BaseStore):
    """CSV 를 GitHub 저장소에 커밋합니다. 버전 이력이 남습니다.

    설정:
      1. GitHub -> Settings -> Developer settings -> Personal access tokens
         -> Fine-grained token. 대상 저장소에 **Contents: Read and write**
         권한만 주십시오.
      2. Streamlit Secrets:

            [github]
            token = "github_pat_..."
            repo  = "사용자명/저장소명"
            path  = "data/lnp_web_data.csv"
            branch = "main"

      3. requirements.txt 에 `PyGithub` 를 추가합니다.

    **중요:** 앱 코드와 같은 저장소·같은 브랜치에 커밋하면 커밋마다
    Streamlit Cloud 가 재배포를 시작합니다. 저장할 때마다 앱이 재시작되어
    느려집니다. 데이터 전용 저장소를 따로 만들거나, 최소한 별도 브랜치를
    쓰십시오.
    """
    name = "github"
    is_persistent = True

    def __init__(self, cfg):
        self.token = cfg["token"]
        self.repo_name = cfg["repo"]
        self.path = cfg.get("path", "data/lnp_web_data.csv")
        self.branch = cfg.get("branch", "main")
        self._repo = None

    def _connect(self):
        if self._repo is None:
            from github import Github
            self._repo = Github(self.token).get_repo(self.repo_name)
        return self._repo

    def load(self):
        repo = self._connect()
        try:
            f = repo.get_contents(self.path, ref=self.branch)
        except Exception:
            return None
        return pd.read_csv(io.BytesIO(f.decoded_content), encoding="utf-8-sig")

    def save(self, df):
        repo = self._connect()
        csv = df.to_csv(index=False, encoding="utf-8-sig")
        msg = f"data: {len(df)} rows"
        try:
            f = repo.get_contents(self.path, ref=self.branch)
            repo.update_file(self.path, msg, csv, f.sha, branch=self.branch)
        except Exception:
            repo.create_file(self.path, msg, csv, branch=self.branch)

    def describe(self):
        return f"GitHub ({self.repo_name}/{self.path}@{self.branch})"


# ==========================================================================
# Supabase (Postgres) — 여러 사람이 동시에 쓸 때
# ==========================================================================
class SupabaseStore(BaseStore):
    """Postgres 테이블에 저장합니다. 동시 쓰기가 안전한 유일한 선택지입니다.

    설정:
      1. https://supabase.com 에서 무료 프로젝트를 만듭니다.
      2. SQL Editor 에서 테이블을 만듭니다 (컬럼은 모두 text 로 두고
         읽을 때 변환하는 편이 스키마 변경에 강합니다):

            create table lnp_rows (
              id bigserial primary key,
              payload jsonb not null,
              updated_at timestamptz default now()
            );

      3. Streamlit Secrets:

            [supabase]
            url = "https://xxxx.supabase.co"
            key = "service_role 또는 anon 키"
            table = "lnp_rows"

      4. requirements.txt 에 `supabase` 를 추가합니다.

    anon 키를 쓰면 Row Level Security 정책을 반드시 설정하십시오.
    설정하지 않으면 누구나 읽고 쓸 수 있습니다.
    """
    name = "supabase"
    is_persistent = True

    def __init__(self, cfg):
        self.url = cfg["url"]
        self.key = cfg["key"]
        self.table = cfg.get("table", "lnp_rows")
        self._cl = None

    def _connect(self):
        if self._cl is None:
            from supabase import create_client
            self._cl = create_client(self.url, self.key)
        return self._cl

    def load(self):
        cl = self._connect()
        res = cl.table(self.table).select("payload").execute()
        if not res.data:
            return None
        return pd.DataFrame([r["payload"] for r in res.data])

    def save(self, df):
        cl = self._connect()
        # 전체 교체. 행 단위 갱신이 필요하면 id 를 함께 관리하십시오.
        cl.table(self.table).delete().neq("id", 0).execute()
        recs = json.loads(df.to_json(orient="records"))
        for i in range(0, len(recs), 500):          # 배치 삽입
            cl.table(self.table).insert(
                [{"payload": r} for r in recs[i:i + 500]]).execute()

    def describe(self):
        return f"Supabase ({self.url.split('//')[-1][:18]}…/{self.table})"


# ==========================================================================
# 자동 선택
# ==========================================================================
_ORDER = [("supabase", SupabaseStore), ("gsheet", GSheetStore),
          ("github", GitHubStore)]


def get_store(st, local_path="lnp_web_data.csv"):
    """st.secrets 에 설정된 백엔드를 자동으로 고릅니다.

    아무것도 없으면 LocalStore 를 돌려줍니다 — 지금과 같은 동작이므로
    설정하기 전에도 앱이 그대로 돕니다.

    우선순위는 supabase > gsheet > github 입니다(동시 쓰기 안전성 순).
    """
    try:
        secrets = st.secrets
    except Exception:
        return LocalStore(local_path)

    for key, cls in _ORDER:
        try:
            if key in secrets:
                return cls(secrets[key])
        except Exception as e:
            # 설정이 있는데 실패하면 조용히 로컬로 떨어지지 않고 알립니다.
            # 조용한 실패는 '저장한 줄 알았는데 안 된' 상황을 만듭니다.
            st.error(f"{key} 저장소 설정을 읽지 못했습니다: {e}")
    return LocalStore(local_path)


def show_store_status(st, store, where=None):
    """어디에 저장되는지 사용자에게 명시합니다.

    `where` 를 주면 그곳에 그립니다 (`st.sidebar`, `st.container()` 등).
    기본은 `st.sidebar` 이고, 사이드바가 없는 객체면 `st` 자신에 그립니다.
    """
    tgt = where if where is not None else getattr(st, "sidebar", st)
    if store.is_persistent:
        tgt.success(f"저장 위치: **{store.describe()}**\n\n"
                    "코드를 수정해 재배포해도 데이터가 남습니다.")
    else:
        tgt.warning(
            f"저장 위치: **{store.describe()}**\n\n"
            "**GitHub 에서 코드를 수정하면 입력한 데이터가 사라집니다.** "
            "재배포가 새 컨테이너를 만들고, 그 안에는 git 저장소 내용만 "
            "있기 때문입니다. 영구 저장은 `lnp_store.py` 의 설정 안내를 "
            "보십시오. 그때까지는 사이드바에서 CSV 를 내려받아 보관하십시오.")
