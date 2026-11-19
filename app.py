"""お好み焼き店専用 Instagram「運用パック」

起動コマンド: python3 -m streamlit run app.py
"""
import calendar
import concurrent.futures
import datetime as dt
import io
import json
import os
import re
import time
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# ===== 設定(変えたいときは、ここを書きかえます) =====
# 使うGeminiのモデル名(混み合っていて使えないときは、次のモデルを自動で試す)
MODEL_NAMES = ["gemini-3.6-flash", "gemini-3.8-flash"]
MODEL_TIME_LIMIT = 180  # 1つのモデルを待つ時間(秒)。これを過ぎたら次のモデルに切りかえる
NO_SEARCH_NOTICE = "検索なしの一般的な傾向です。最新ではないので、確認してください"
RETRY_WAIT = 5  # 混雑していたときに、次を試すまで待つ時間(秒)
MAX_STORES = 3  # 登録できるお店の数
TOP_N = 5  # 振り返りで見る「上位・下位の投稿」の数
BIO_MAX_CHARS = 150  # Instagramのプロフィール文の文字数の上限
MAX_IMAGE_MB = 15  # アップロードできる画像の大きさの上限
PLAN_COUNTS = {"ストーリー": 30, "リール": 10, "フィード": 10}  # 月次企画の本数
WEEKDAYS = "月火水木金土日"

# ファイルの場所は app.py と同じフォルダを基準にする(どこから起動しても動くように)
BASE_DIR = Path(__file__).resolve().parent
STORES_FILE = BASE_DIR / "stores.json"
TREND_MEMO_FILE = BASE_DIR / "trend_memo.json"
TREND_MEMO_MAX_DAYS = 30  # この日数より古いメモは「古い」と表示する
TREND_MEMO_KEEP = 24  # trend_memo.json に残すメモの数(古いものから消す)
TREND_MEMO_OLD_NOTICE = "トレンドのメモが古いです。更新してください"

# .env から APIキーを読み込む
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# お店の情報の項目(保存するときの名前: 画面に出す名前, 入力例)
STORE_FIELDS = {
    "name": ("店名", "例:お好み焼き たろう"),
    "location": ("場所", "例:大阪市〇〇区〇〇 〇〇駅から歩いて5分"),
    "hours": ("営業時間(お休みの日も)", "例:11:00〜22:00 火曜はお休み"),
    "reservation": ("予約方法", "例:電話 06-0000-0000 / ネット予約あり"),
    "kodawari": ("こだわり", "例:山いもたっぷりのふわふわ生地、自家製ソース"),
    "ng_words": ("NGワード(使わない言葉)", "例:激安、最安値"),
}

# 振り返りCSVの列の役割と、自動で見つけるためのキーワード
COLUMN_ROLES = {
    "投稿日時": ["日時", "日付", "date", "time", "公開"],
    "投稿内容(キャプション)": ["説明", "キャプション", "caption", "本文", "description"],
    "投稿形式": ["タイプ", "形式", "type", "種類"],
    "いいね": ["いいね", "like"],
    "保存数": ["保存", "save"],
    "コメント": ["コメント", "comment"],
    "シェア": ["シェア", "share"],
    "リーチ": ["リーチ", "reach"],
}
NUMERIC_ROLES = ["いいね", "保存数", "コメント", "シェア", "リーチ"]
REACTION_ROLES = ["いいね", "保存数", "コメント", "シェア"]
NONE_LABEL = "(使わない)"
ALL_MONTHS_LABEL = "ぜんぶの期間"

# AIに毎回わたす共通のルール
SYSTEM_INSTRUCTION = """あなたは、お好み焼き店のInstagram運用を専門にするプロです。
鉄板の音、生地を焼く様子、ソースやマヨネーズをかける瞬間、かつお節がおどる様子、
ヘラで切る場面など、お好み焼きならではの魅力を生かした提案をしてください。

必ず守るルール:
- 日本語で、すぐ使える具体的な内容を書く。
- 「お店の情報」に書かれていないこと(メニュー名、値段、受賞歴、キャンペーンなど)を作らない。
  必要なときは「〇〇」と書いて、あとから入れられるようにする。
- NGワードは絶対に使わない。
- 営業時間やお休みの日と矛盾する内容にしない。
- 前置きやあいさつは書かず、すぐ本題に入る。
"""


class AppError(Exception):
    """画面にそのまま見せる、わかりやすいエラー"""


# ===== お店の情報の読み書き =====
def load_stores():
    """stores.json を読み込む(ファイルがなければ空のリスト)"""
    if not STORES_FILE.exists():
        return []
    try:
        data = json.loads(STORES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AppError(
            f"お店の情報のファイル(stores.json)を読めません。\n\n"
            f"ファイルの中身がこわれているかもしれません。直すか、別の名前に変えてから、"
            f"このページを開き直してください。\n\nくわしい内容:{e}"
        )
    if not isinstance(data, list):
        raise AppError("stores.json の形がちがいます。直すか、別の名前に変えてください。")
    stores = []
    for item in data[:MAX_STORES]:
        if isinstance(item, dict):
            stores.append({k: str(item.get(k, "")) for k in STORE_FIELDS})
    return stores


def save_stores(stores):
    """stores.json に保存する(とちゅうで失敗しても元のファイルがこわれないように、一度別名で書く)"""
    tmp = STORES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(stores, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, STORES_FILE)


# ===== トレンドのメモの読み書き =====
def load_trend_memos():
    """trend_memo.json を読み込む(ファイルがなければ空のリスト。古い順に並んでいる)"""
    if not TREND_MEMO_FILE.exists():
        return []
    try:
        data = json.loads(TREND_MEMO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise AppError(
            f"トレンドのメモのファイル(trend_memo.json)を読めません。\n\n"
            f"ファイルの中身がこわれているかもしれません。直すか、別の名前に変えてください。\n\nくわしい内容:{e}"
        )
    if not isinstance(data, list):
        raise AppError("trend_memo.json の形がちがいます。直すか、別の名前に変えてください。")
    memos = []
    for item in data:
        if isinstance(item, dict) and parse_date(item.get("date")) and str(item.get("text", "")).strip():
            memos.append({"date": str(item["date"]), "text": str(item["text"])})
    return memos


def save_trend_memo(text):
    """今日の日付をつけて、メモを trend_memo.json に追加する"""
    memos = load_trend_memos()
    memos.append({"date": dt.date.today().isoformat(), "text": text})
    memos = memos[-TREND_MEMO_KEEP:]
    tmp = TREND_MEMO_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(memos, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, TREND_MEMO_FILE)


def memo_status(memo):
    """メモの日付から「〇月〇日時点の情報」の文と、古いかどうかを返す"""
    date = parse_date(memo["date"])
    label = f"{date.month}月{date.day}日時点の情報"
    is_old = (dt.date.today() - date).days > TREND_MEMO_MAX_DAYS
    return date, label, is_old


def split_ng_words(store):
    """NGワードの文字を、1語ずつのリストに分ける"""
    return [w for w in re.split(r"[、,,\s]+", store.get("ng_words", "")) if w]


def find_ng_words(text, store):
    """文章の中に入ってしまったNGワードを探す"""
    lower = text.lower()
    return [w for w in split_ng_words(store) if w.lower() in lower]


def store_block(store):
    """AIにわたす「お店の情報」の文章を作る"""
    lines = ["# お店の情報"]
    for key, (label, _) in STORE_FIELDS.items():
        lines.append(f"- {label}:{store.get(key) or '未登録'}")
    return "\n".join(lines)


# ===== Gemini を呼び出す部分 =====
@st.cache_resource(show_spinner=False)
def get_client(api_key):
    """Geminiにつなぐ準備"""
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=MODEL_TIME_LIMIT * 1000),
    )


def generate_with_time_limit(client, **kwargs):
    """Geminiを呼び出す。返事が来ないまま止まっても、決めた時間で必ず打ち切る"""
    # 通信側の時間切れ設定だけでは何分も止まることがあったので、別の作業として動かして時計で見張る
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(client.models.generate_content, **kwargs).result(
            timeout=MODEL_TIME_LIMIT
        )
    finally:
        pool.shutdown(wait=False)  # 止まった通信の終わりを待たない


def call_gemini(prompt, search=False, schema=None, image=None):
    """Geminiを呼び出して、返事(文章と元の応答)を受け取る"""
    if not API_KEY:
        raise AppError(
            "APIキーが見つかりません。\n\n"
            ".env ファイルに GEMINI_API_KEY=(あなたのキー) と書いて、アプリを起動し直してください。"
        )
    config = {"system_instruction": SYSTEM_INSTRUCTION}
    if search:
        config["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if schema:
        config["response_mime_type"] = "application/json"
        config["response_schema"] = schema

    contents = [prompt]
    if image:
        contents.insert(
            0, types.Part.from_bytes(data=image["data"], mime_type=image["mime"])
        )

    # 混雑(500番台)、提供終了(404)、時間切れのときだけ、次のモデルに切りかえる
    # 混雑はすぐ直ることが多いので、少し待ちながら全部のモデルをもう一周試す
    client = get_client(API_KEY)
    tries = MODEL_NAMES * 2
    for i, model_name in enumerate(tries):
        is_last = i == len(tries) - 1
        try:
            response = generate_with_time_limit(
                client,
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(**config),
            )
            break
        except genai_errors.APIError as e:
            if is_last or not (e.code == 404 or (e.code and e.code >= 500)):
                raise
            time.sleep(RETRY_WAIT)
        except (concurrent.futures.TimeoutError, httpx.TimeoutException):
            # 時間切れは待ち時間が長いので、全部のモデルを一周したらあきらめる
            if i >= len(MODEL_NAMES) - 1:
                raise AppError("AIの返事に時間がかかりすぎました。少し待ってから、もう一度ためしてください。")

    # 返事が空だったり、とちゅうで切れたりしていないか確かめる
    candidate = response.candidates[0] if response.candidates else None
    reason = candidate.finish_reason.name if candidate and candidate.finish_reason else ""
    if reason == "MAX_TOKENS":
        raise AppError("AIの返事が長くなりすぎて、とちゅうで切れました。もう一度ためしてください。")
    if not response.text:
        raise AppError(
            "AIから返事がありませんでした。入力した言葉を少し変えて、もう一度ためしてください。"
            + (f"(理由:{reason})" if reason else "")
        )
    return response.text, response


def call_gemini_checked(prompt, schema, check, image=None, attempts=3):
    """表などの決まった形で受け取り、本数などが正しいか確かめる。ちがえば理由を伝えて作り直す"""
    best, best_problems = None, None
    extra = ""
    for _ in range(attempts):
        text, _resp = call_gemini(prompt + extra, schema=schema, image=image)
        try:
            data = json.loads(text)
        except ValueError:
            extra = "\n\n# 注意\n前回はJSONの形がこわれていました。正しいJSONで出力してください。"
            continue
        problems = check(data)
        if best is None or len(problems) < len(best_problems):
            best, best_problems = data, problems
        if not problems:
            break
        extra = "\n\n# 注意\n前回の案には次の問題がありました。直して全部を作り直してください。\n- " + "\n- ".join(problems)
    if best is None:
        raise AppError("AIの返事を読み取れませんでした。もう一度ためしてください。")
    return best, best_problems


def friendly_error(e):
    """エラーを、わかりやすい言葉に直す"""
    if isinstance(e, AppError):
        return str(e), None
    detail = str(e).replace(API_KEY, "***") if API_KEY else str(e)
    if isinstance(e, genai_errors.APIError):
        message = str(e.message or "")
        if e.code in (401, 403) or "API key" in message or "API_KEY" in message:
            return "APIキーが正しくないようです。.env の GEMINI_API_KEY を確かめて、アプリを起動し直してください。", detail
        if e.code == 404:
            return f"AIのモデル({'、'.join(MODEL_NAMES)})が見つかりません。app.py の上の方にある MODEL_NAMES を新しい名前に変えてください。", detail
        if e.code == 429:
            return "AIを使える回数の上限に届きました。しばらく待ってから、もう一度ためしてください。", detail
        if e.code and e.code >= 500:
            return "AIが今とても混み合っています。少し待ってから、もう一度ためしてください。", detail
        return "AIへのお願いがうまくいきませんでした。入力を見直して、もう一度ためしてください。", detail
    if isinstance(e, httpx.TimeoutException):
        return "AIの返事に時間がかかりすぎました。もう一度ためしてください。", detail
    if isinstance(e, httpx.TransportError):
        return "インターネットにつながらないようです。接続を確かめて、もう一度ためしてください。", detail
    return "思わぬエラーが起きました。もう一度ためしてください。直らないときは、下のくわしい内容を確かめてください。", detail


def show_error(e):
    """エラーを赤い枠で画面に出す"""
    message, detail = friendly_error(e)
    st.error(message)
    if detail:
        with st.expander("くわしい内容(困ったときに見てください)"):
            st.code(detail, language=None)


# ===== 結果の表示(コピー・保存) =====
def copy_button(text, key):
    """「結果をコピー」ボタン(Streamlitにないので、小さなHTMLで作る)"""
    # 文章の中の記号でHTMLがこわれないように、安全な形に変える
    payload = json.dumps(text).replace("<", "\\u003c")
    components.html(
        f"""
        <button id="copy-{key}" style="width:100%;height:40px;border:1px solid #d44a1c;border-radius:8px;
          background:#d44a1c;color:#fff;font-size:15px;cursor:pointer;font-family:sans-serif;">
          📋 結果をコピー</button>
        <script>
          const btn = document.getElementById("copy-{key}");
          const text = {payload};
          function fallback() {{
            const area = document.createElement("textarea");
            area.value = text;
            document.body.appendChild(area);
            area.select();
            const ok = document.execCommand("copy");
            area.remove();
            return ok;
          }}
          btn.addEventListener("click", async () => {{
            let ok = false;
            try {{ await navigator.clipboard.writeText(text); ok = true; }}
            catch (err) {{ ok = fallback(); }}
            btn.textContent = ok ? "✅ コピーしました" : "⚠️ コピーできません。保存ボタンを使ってね";
            setTimeout(() => {{ btn.textContent = "📋 結果をコピー"; }}, 2500);
          }});
        </script>
        """,
        height=48,
    )


def safe_filename(name):
    """ファイル名に使えない文字を「_」に置きかえる"""
    return re.sub(r'[\\/:*?"<>|\s]+', "_", name).strip("_") or "結果"


def save_result(key, result):
    """作った結果を覚えておく(メニューを切りかえても消えないように)"""
    st.session_state.setdefault("results", {})[key] = result


def escape_markdown(text):
    """「~」や「$」が取り消し線や数式に化けないようにする(コードブロックの中はそのまま)"""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("~", "\\~").replace("$", "\\$")
    return "".join(parts)


def show_result(key, file_label):
    """覚えておいた結果と、コピー・保存ボタンを表示する"""
    result = st.session_state.get("results", {}).get(key)
    if not result:
        return
    st.divider()
    st.subheader(result.get("title", "できあがり"))
    for message in result.get("warnings", []):
        st.warning(message)

    left, right = st.columns(2)
    with left:
        copy_button(result["text"], str(abs(hash(key))))
    with right:
        st.download_button(
            "💾 テキストで保存",
            data=result["text"].encode("utf-8"),
            file_name=safe_filename(f"{file_label}_{dt.date.today():%Y%m%d}") + ".txt",
            mime="text/plain",
            key=f"dl_{key}",
            on_click="ignore",
            use_container_width=True,
        )

    if result.get("table") is not None:
        # 長い文も全部読めるように、折り返して表示する表を使う
        st.table(result["table"].set_index("日付"))
    else:
        st.markdown(escape_markdown(result.get("display", result["text"])))

    if result.get("entry_html"):
        components.html(result["entry_html"], height=90, scrolling=True)


def run_and_save(key, title, store, work):
    """AIの作業を動かして、結果を保存する。エラーは画面に出す"""
    with st.spinner("AIが考えています。数分かかることがあります…"):
        try:
            result = work()
        except Exception as e:  # どんなエラーでも、画面にわかりやすく出す
            show_error(e)
            return
    result["title"] = title
    if store:
        hits = find_ng_words(result["text"], store)
        if hits:
            result.setdefault("warnings", []).append(
                f"NGワード「{'、'.join(hits)}」が入っています。使う前に直してください。"
            )
    save_result(key, result)


# ===== 画面の部品 =====
def month_picker(key):
    """作る月を選ぶ(最初は来月になっている)"""
    today = dt.date.today()
    months = [
        ((today.year * 12 + today.month - 1 + i) // 12, (today.month - 1 + i) % 12 + 1)
        for i in range(13)
    ]
    return st.selectbox(
        "いつの月の分を作りますか?",
        months,
        index=1,
        format_func=lambda m: f"{m[0]}年{m[1]}月",
        key=key,
    )


def trend_prompt_block(key):
    """トレンド調査の結果があれば、使うかどうかを選べるようにする"""
    trend = st.session_state.get("results", {}).get("trend")
    if not trend:
        st.caption("先に「1. トレンド調査」をすると、その結果も使えます。")
        return ""
    if trend.get("no_search"):
        st.warning(NO_SEARCH_NOTICE)
    if st.checkbox("「トレンド調査」の結果も使う", value=True, key=key):
        heading = "参考にする一般的な傾向(検索なし。最新とは限らない)" if trend.get("no_search") else "参考にする最新の傾向"
        return f"\n# {heading}\n{trend['text']}\n"
    return ""


def trend_memo_prompt_block():
    """保存した最新のトレンドのメモを、AIにわたす文にする(古ければ注意を出す)"""
    try:
        memos = load_trend_memos()
    except AppError as e:
        st.error(str(e))
        return ""
    if not memos:
        st.caption("「1. トレンド調査」でトレンドのメモを保存すると、その内容もAIにわたします。")
        return ""
    memo = memos[-1]
    date, label, is_old = memo_status(memo)
    if is_old:
        st.warning(f"{TREND_MEMO_OLD_NOTICE}({label})")
    else:
        st.info(f"📝 トレンドのメモ({label})をAIにわたします。")
    return f"""
# 最新トレンドのメモ({date:%Y年%m月%d日}時点。運用者が調べて貼り付けたもの)
{memo['text'].strip()}
"""


def need_store(store):
    """お店が選ばれていないときに案内を出す"""
    if store:
        return True
    st.info("👈 まず、左の「お店を登録・直す」からお店を登録してください。")
    return False


def sidebar(stores):
    """左のメニュー(お店の選択・登録と、機能の選択)"""
    state = st.session_state
    with st.sidebar:
        st.title("🥢 お好み焼き 運用パック")

        # --- お店をえらぶ ---
        st.header("① お店をえらぶ")
        names = [s["name"] for s in stores]
        if "pending_store" in state:
            state["current_store"] = state.pop("pending_store")
        if state.get("current_store") not in names:
            state.pop("current_store", None)
        store = None
        if stores:
            current = st.selectbox("使うお店", names, key="current_store")
            store = stores[names.index(current)]
        else:
            st.caption("まだお店が登録されていません。")

        with st.expander("お店を登録・直す", expanded=not stores):
            store_editor(stores)

        # --- 機能をえらぶ ---
        st.header("② やることをえらぶ")
        # 「1.」がマークダウンの箇条書きと見なされて消えないように「1\.」にして表示する
        page = st.radio(
            "メニュー",
            list(PAGES.keys()),
            format_func=lambda name: name.replace(". ", "\\. ", 1),
            label_visibility="collapsed",
        )

        st.caption(f"使っているAI:{MODEL_NAMES[0]}(混んでいるときは {MODEL_NAMES[-1]})")
        if not API_KEY:
            st.warning(".env の GEMINI_API_KEY が見つかりません。")
    return store, page


def store_editor(stores):
    """お店を登録・修正・削除するフォーム"""
    state = st.session_state
    nonce = state.setdefault("form_nonce", 0)
    slots = list(range(len(stores)))
    if len(stores) < MAX_STORES:
        slots.append(len(stores))  # 空いているときは「新しいお店」を選べる
    else:
        st.caption(f"お店は{MAX_STORES}つまで登録できます。")
    slot = st.selectbox(
        "どのお店を?",
        slots,
        format_func=lambda i: stores[i]["name"] if i < len(stores) else "＋ 新しいお店",
    )
    current = stores[slot] if slot < len(stores) else {}

    with st.form(f"store_form_{slot}_{nonce}"):
        values = {}
        for field, (label, example) in STORE_FIELDS.items():
            widget = st.text_area if field in ("kodawari", "hours") else st.text_input
            values[field] = widget(
                label, value=current.get(field, ""), placeholder=example
            ).strip()
        submitted = st.form_submit_button("保存する", type="primary")

    if submitted:
        others = [s["name"] for i, s in enumerate(stores) if i != slot]
        if not values["name"]:
            st.error("店名を入れてください。")
        elif values["name"] in others:
            st.error("同じ名前のお店がすでにあります。")
        else:
            updated = list(stores)
            updated[slot:slot + 1] = [values]
            write_stores(updated, values["name"], "保存しました")

    # まちがえて消さないように、チェックを入れないと押せないようにする
    if slot < len(stores):
        sure = st.checkbox("このお店を消す", key=f"delete_check_{slot}_{nonce}")
        if st.button("消す", disabled=not sure, key=f"delete_btn_{slot}_{nonce}"):
            write_stores(stores[:slot] + stores[slot + 1:], None, "消しました")


def write_stores(stores, select_name, message):
    """保存して画面を作り直す。保存できなければエラーを出す"""
    try:
        save_stores(stores)
    except OSError as e:
        st.error(f"stores.json に保存できませんでした:{e}")
        return
    if select_name:
        st.session_state["pending_store"] = select_name
    st.session_state["form_nonce"] += 1
    st.session_state["flash"] = message
    st.rerun()


# ===== 1. トレンド調査 =====
def page_trend(store):
    st.header("1. トレンド調査")
    trend_memo_section()
    st.divider()
    st.subheader("AIで調べる")
    st.write("インターネットで調べて、飲食店のInstagramで今はやっていることをまとめます。")
    focus = st.text_input(
        "とくに知りたいこと(なくてもOK)", placeholder="例:リールの長さ、人気の音楽"
    )
    if st.button("🔍 調べる", type="primary"):
        def work():
            today = dt.date.today()
            prompt = f"""今日は{today:%Y年%m月%d日}です。
Googleで検索して、日本の飲食店のInstagram運用の最新の傾向を調べ、要約してください。
{f'とくに知りたいこと:{focus}' if focus else ''}

次の見出しで、Markdownで書いてください。
## 今の傾向(5つまで)
## フィード・リール・ストーリーそれぞれのポイント
## お好み焼き店で今すぐ試せること(3つ)

条件:
- できるだけ最近の情報を使い、古い情報は使わない。
- 検索で確かめられなかったことは、言い切らずに「確かめられなかった」と書く。
"""
            try:
                text, response = call_gemini(prompt, search=True)
            except genai_errors.APIError as e:
                if e.code != 429:
                    raise
                # 検索の上限のときは、検索なしで作り直す。「最新情報」に見えないよう必ず注意を付ける
                text, _resp = call_gemini(f"""今日は{today:%Y年%m月%d日}です。
日本の飲食店のInstagram運用の一般的な傾向を、あなたの知識の範囲でまとめてください。
インターネットでの検索は使えません。
{f'とくに知りたいこと:{focus}' if focus else ''}

次の見出しで、Markdownで書いてください。
## 一般的な傾向(5つまで)
## フィード・リール・ストーリーそれぞれのポイント
## お好み焼き店で今すぐ試せること(3つ)

条件:
- 「最新」「今年」など、新しい情報だと受け取られる書き方をしない。
- 数字、調査名、出典、URLを書かない。
""")
                return {
                    "text": f"※{NO_SEARCH_NOTICE}\n\n{text}",  # コピー・保存のときも注意が残るように
                    "display": text,
                    "warnings": [NO_SEARCH_NOTICE],
                    "no_search": True,
                }
            result = {"text": text, "warnings": []}
            meta = response.candidates[0].grounding_metadata
            chunks = (meta.grounding_chunks or []) if meta else []
            links = []
            for chunk in chunks:
                if chunk.web and chunk.web.uri and chunk.web.uri not in [u for _, u in links]:
                    links.append((chunk.web.title or chunk.web.uri, chunk.web.uri))
            if links:
                result["text"] += "\n\n## 調べた先\n" + "\n".join(
                    f"- [{title}]({uri})" for title, uri in links
                )
            else:
                result["warnings"].append(
                    "今回はインターネットの検索結果を確認できませんでした。内容が古いかもしれないので、もう一度ためすことをおすすめします。"
                )
            # Googleの検索機能を使ったときに表示する決まりの「検索候補」
            entry = meta.search_entry_point if meta else None
            if entry and entry.rendered_content:
                result["entry_html"] = entry.rendered_content
            return result

        run_and_save("trend", "トレンド調査の結果", None, work)
    show_result("trend", "トレンド調査")


def trend_memo_section():
    """最新トレンドのメモを貼り付けて保存する欄"""
    st.subheader("📝 最新トレンドのメモ(貼り付け)")
    st.write("月初などに自分で調べた最新の情報を貼り付けて保存すると、「2. 月次企画」「3. 動画企画」「7. 共通テンプレ」でAIにわたします。")
    try:
        memos = load_trend_memos()
    except AppError as e:
        st.error(str(e))
        return  # こわれたファイルを上書きしないように、保存できないようにする

    latest = memos[-1] if memos else None
    if latest:
        _date, label, is_old = memo_status(latest)
        if is_old:
            st.warning(f"{TREND_MEMO_OLD_NOTICE}({label})")
        else:
            st.success(label)
    else:
        st.caption("まだ保存したメモはありません。")

    text = st.text_area(
        "最新トレンドのメモ",
        value=latest["text"] if latest else "",
        height=220,
        max_chars=10000,
        placeholder="例:・リールは7〜15秒の短いものが伸びている\n・〇〇の音源がはやっている",
        key="trend_memo_text",
    )
    if st.button("💾 保存", key="trend_memo_save"):
        if not text.strip():
            st.error("メモが空です。内容を貼り付けてから保存してください。")
        else:
            try:
                save_trend_memo(text.strip())
            except (OSError, AppError) as e:
                st.error(f"trend_memo.json に保存できませんでした:{e}")
            else:
                st.session_state["flash"] = "トレンドのメモを保存しました"
                st.rerun()

    if len(memos) > 1:
        with st.expander(f"これまでのメモ({len(memos)}件)"):
            for memo in reversed(memos):
                st.markdown(f"**{memo_status(memo)[1]}**")
                st.text(memo["text"])


# ===== 2. 月次企画 =====
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "format": {"type": "string", "enum": list(PLAN_COUNTS.keys())},
                    "theme": {"type": "string"},
                    "materials": {"type": "string"},
                },
                "required": ["date", "format", "theme", "materials"],
            },
        }
    },
    "required": ["items"],
}


def parse_date(text):
    """「2026-12-03」の形の文字を日付に直す。直せなければ None"""
    try:
        return dt.datetime.strptime(str(text).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def check_plan(data, year, month):
    """本数と日付が合っているかを、AIまかせにせずプログラムで確かめる"""
    items = data.get("items", [])
    problems = []
    for name, want in PLAN_COUNTS.items():
        got = sum(1 for item in items if item.get("format") == name)
        if got != want:
            problems.append(f"{name}は{want}本必要なのに{got}本になっています")
    wrong = []
    for item in items:
        date = parse_date(item.get("date"))
        if not date or (date.year, date.month) != (year, month):
            wrong.append(str(item.get("date")))
    if wrong:
        problems.append(f"{year}年{month}月ではない日付があります:{'、'.join(wrong[:5])}")
    return problems


def plan_to_table(items):
    """案のリストを、日付順の表に直す(曜日はプログラムで正しく計算する)"""
    def sort_key(item):
        return parse_date(item.get("date")) or dt.date.max

    rows = []
    for item in sorted(items, key=sort_key):
        date = parse_date(item.get("date"))
        label = f"{date:%m/%d}({WEEKDAYS[date.weekday()]})" if date else str(item.get("date"))
        rows.append(
            {
                "日付": label,
                "形式": item.get("format", ""),
                "テーマ": item.get("theme", ""),
                "撮る素材": item.get("materials", ""),
            }
        )
    return pd.DataFrame(rows, columns=["日付", "形式", "テーマ", "撮る素材"])


def page_plan(store):
    st.header("2. 月次企画")
    st.write("1か月分の投稿の案を、表にして作ります。ストーリー30本・リール10本・フィード10本です。")
    if not need_store(store):
        return
    year, month = month_picker("plan_month")
    wish = st.text_area(
        "この月に伝えたいこと(なくてもOK)",
        placeholder="例:新メニューの〇〇を広めたい。12/20に〇〇のイベントがある。",
        key="plan_wish",
    )
    trend = trend_memo_prompt_block() + trend_prompt_block("plan_trend")

    key = f"plan|{store['name']}"
    if st.button("🗓️ 企画を作る", type="primary"):
        def work():
            last_day = calendar.monthrange(year, month)[1]
            prompt = f"""{year}年{month}月(1日〜{last_day}日)のInstagram投稿の企画を作ってください。

{store_block(store)}

# この月に伝えたいこと
{wish or '特になし'}
{trend}
# 本数(ぴったり守る)
- ストーリー:30本
- リール:10本
- フィード:10本
合計50件をitemsに入れてください。

# 各項目の書き方
- date:投稿日を YYYY-MM-DD で。すべて{year}年{month}月の中にして、月の中でバランスよく散らす。
- format:ストーリー / リール / フィード のどれか。
- theme:何を伝える投稿か、1文で具体的に。同じ内容をくり返さない。
- materials:撮影やデータ準備が必要な素材を具体的に(例:ソースをかける瞬間のアップ動画10秒)。

# 考え方
- 季節の行事やその月らしさを生かす。
- 来店につながる投稿と、保存・シェアされやすい投稿を組み合わせる。
- お休みの日は「今日も営業中」のような内容にしない。
"""
            data, problems = call_gemini_checked(
                prompt, PLAN_SCHEMA, lambda d: check_plan(d, year, month)
            )
            table = plan_to_table(data.get("items", []))
            # タブ区切りにしておくと、Excelやスプレッドシートにそのまま貼れる
            clean = table.replace(r"[\t\r\n]+", " ", regex=True)
            text = clean.to_csv(sep="\t", index=False)
            warnings = [f"条件どおりに作れなかったところがあります:{p}" for p in problems]
            return {"text": text, "table": table, "warnings": warnings}

        run_and_save(key, f"{year}年{month}月の企画", store, work)
    show_result(key, f"{store['name']}_月次企画")


# ===== 3. 動画企画 =====
def page_video(store):
    st.header("3. 動画企画(月4本)")
    st.write("1か月に出す動画4本の、台本・撮る順番・画面に出す文字・音楽のふんいきを作ります。")
    if not need_store(store):
        return
    year, month = month_picker("video_month")
    length = st.selectbox("動画の長さ", ["15秒くらい", "30秒くらい", "60秒くらい"], index=1)
    wish = st.text_area(
        "動画で伝えたいこと(なくてもOK)", placeholder="例:焼いているときの音を聞かせたい", key="video_wish"
    )
    trend = trend_memo_prompt_block() + trend_prompt_block("video_trend")

    key = f"video|{store['name']}"
    if st.button("🎬 動画の企画を作る", type="primary"):
        def work():
            prompt = f"""{year}年{month}月に出すInstagramリール動画の企画を、ちょうど4本作ってください。
動画の長さは{length}です。

{store_block(store)}

# 伝えたいこと
{wish or '特になし'}
{trend}
# 書き方(Markdownで、4本すべてに次の項目を書く)
## 動画1:(タイトル)
- **ねらい**:だれに、何を感じてほしいか
- **台本**:秒数ごとの表(秒数 / 映像 / 話す言葉やナレーション)
- **撮影順**:現場で撮る順番を番号つきで(焼き上がると元に戻せないなど、調理の流れを考える)
- **テロップ**:画面に出す文字を、出す順に(最初の3秒で目を引くものにする)
- **BGMの雰囲気**:テンポや雰囲気を言葉で(実在の曲名は書かない)

4本は内容が重ならないようにしてください。
"""
            text, _ = call_gemini(prompt)
            return {"text": text}

        run_and_save(key, f"{year}年{month}月の動画企画", store, work)
    show_result(key, f"{store['name']}_動画企画")


# ===== 4. 現地撮影の準備 =====
def page_shoot(store):
    st.header("4. 現地撮影の準備")
    st.write("お店で撮影する日のために、撮るものリスト・当日の流れ・お店への連絡文を作ります。")
    if not need_store(store):
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        shoot_date = st.date_input("撮影する日", value=dt.date.today() + dt.timedelta(days=14))
    with c2:
        start = st.time_input("始める時間", value=dt.time(14, 0), step=dt.timedelta(minutes=15))
    with c3:
        hours = st.number_input("かかる時間(時間)", min_value=0.5, max_value=12.0, value=2.0, step=0.5)
    sender = st.text_input("連絡文に書く、あなたの名前や会社名", placeholder="例:SNS担当の山田")
    wish = st.text_area(
        "撮りたいもの(なくてもOK)",
        placeholder="例:来月の動画4本ぶん。店長さんのインタビューも撮りたい。",
        key="shoot_wish",
    )

    key = f"shoot|{store['name']}"
    if st.button("📸 準備リストを作る", type="primary"):
        def work():
            date_label = f"{shoot_date:%Y年%m月%d日}({WEEKDAYS[shoot_date.weekday()]})"
            prompt = f"""お好み焼き店での現地撮影の準備資料を作ってください。

{store_block(store)}

# 撮影の予定
- 日にち:{date_label}
- 開始:{start:%H:%M}
- 所要時間:{hours}時間
- 撮影する人の名前:{sender or '〇〇'}
- 撮りたいもの:{wish or '特に指定なし。Instagramのフィード・リール・ストーリーで使える素材全般'}

# 書き方(Markdownで、次の3つの見出しで)
## 撮影リスト
表で書く(番号 / 撮るもの / 写真か動画か / 撮り方のコツ / 使い道)。
湯気、鉄板の音、ソースをかける瞬間など、撮り直しがしにくいものには「一発勝負」と印をつける。

## 当日の進行表
表で書く(時間 / やること / お店にお願いすること)。
{start:%H:%M}に始めて{hours}時間で終わるようにする。営業時間やお客さんの多い時間に配慮する。
料理の準備と片付けの時間も入れる。

## お店への事前連絡文
そのまま送れるていねいな文章で。日時、撮影内容、お願いしたい準備(食材の数、スペースなど)、
お客さんが映らないようにする配慮や、映るときは許可をとること、確認してほしいことを入れる。

もし予定の日がお休みの日や営業時間と重なって注意が必要なら、最初に「⚠️ 確認してください」と書いてください。
"""
            text, _ = call_gemini(prompt)
            return {"text": text}

        run_and_save(key, f"{date_label_short(shoot_date)} 撮影の準備", store, work)
    show_result(key, f"{store['name']}_撮影準備")


def date_label_short(date):
    """「12/03(木)」の形にする"""
    return f"{date:%m/%d}({WEEKDAYS[date.weekday()]})"


# ===== 5. 月次の振り返り =====
def read_csv(file):
    """CSVを読み込む(文字コードがちがっても読めるように順番に試す)"""
    raw = file.getvalue()
    if not raw.strip():
        raise AppError("ファイルの中身が空っぽです。")
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        # UTF-16のCSVはタブ区切りのことが多い
        first_line = raw.decode("utf-16", errors="ignore").split("\n", 1)[0]
        sep = "\t" if "\t" in first_line else ","
        return pd.read_csv(io.BytesIO(raw), encoding="utf-16", sep=sep)
    for enc in ["utf-8-sig", "cp932"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise AppError("ファイルの文字の形(文字コード)がわかりませんでした。UTF-8で保存し直してください。")


def guess_column(columns, keywords):
    """キーワードを含む列名を探して、候補の位置を返す"""
    for i, col in enumerate(columns):
        if any(k.lower() in str(col).lower() for k in keywords):
            return i + 1  # 先頭は「使わない」なので +1
    return 0


def build_data(df, mapping):
    """選んだ列だけを取り出し、わかりやすい列名にそろえる"""
    data = pd.DataFrame(index=df.index)
    for role, col in mapping.items():
        if col == NONE_LABEL:
            continue
        if role in NUMERIC_ROLES:
            # 「1,234」のような文字も数字に直す。直せないものは空にする(0にすると平均がくるうため)
            data[role] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )
        else:
            data[role] = df[col].fillna("").astype(str)
    return data


def fmt_num(value):
    """148.0 を 148 のように見やすくする"""
    if pd.isna(value):
        return "不明"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def make_summary(data):
    """AIにわたす集計結果を、プログラムで正しく計算して文章にする"""
    lines = [f"投稿数:{len(data)}件"]
    metrics = [m for m in NUMERIC_ROLES if m in data.columns]

    for m in metrics:
        lines.append(f"{m}:合計 {fmt_num(data[m].sum())} / 1投稿の平均 {fmt_num(data[m].mean())}")

    # 見た人のうち、反応した人の割合
    reactions = [m for m in REACTION_ROLES if m in data.columns]
    if "リーチ" in data.columns and reactions and data["リーチ"].sum() > 0:
        rate = data[reactions].sum().sum() / data["リーチ"].sum() * 100
        lines.append(f"反応率(反応の合計 ÷ リーチの合計):{rate:.1f}%")
    if "リーチ" in data.columns and "保存数" in data.columns and data["リーチ"].sum() > 0:
        lines.append(f"保存率(保存数 ÷ リーチ):{data['保存数'].sum() / data['リーチ'].sum() * 100:.1f}%")

    # 形式ごとの平均
    if "投稿形式" in data.columns and metrics:
        lines.append("\n【投稿形式ごとの件数と平均】")
        grouped = data.groupby("投稿形式")[metrics].mean().round(1)
        grouped.insert(0, "件数", data.groupby("投稿形式").size())
        lines.append(grouped.to_string())

    # 上位・下位の投稿
    sort_key = next((m for m in ["いいね", "保存数", "リーチ"] if m in data.columns), None)
    if sort_key:
        n = max(1, min(TOP_N, len(data) // 2))  # 投稿が少ないときに上位と下位が重ならないように
        cols = [c for c in data.columns if c != "投稿内容(キャプション)"]
        for label, part in [("上位", data.nlargest(n, sort_key)), ("下位", data.nsmallest(n, sort_key))]:
            lines.append(f"\n【{sort_key}の{label}{n}件】")
            for _, row in part.iterrows():
                info = " / ".join(
                    f"{c}:{fmt_num(row[c]) if c in NUMERIC_ROLES else row[c]}" for c in cols
                )
                caption = row.get("投稿内容(キャプション)")
                lines.append("- " + info + (f"\n  内容:{caption[:150]}" if caption else ""))
    return "\n".join(lines)


def page_report(store):
    st.header("5. 月次の振り返り")
    st.write(
        "Instagramの数字が入ったCSVファイル(表のファイル)を入れると、"
        "良かったこと・直すこと・来月のアイデアを報告書にします。"
    )
    if not need_store(store):
        return
    uploaded = st.file_uploader("CSVファイルをここに入れる", type="csv")
    if uploaded is None:
        return
    try:
        df = read_csv(uploaded)
    except Exception as e:
        st.error("CSVファイルを読めませんでした。ファイルが正しいか確かめてください。")
        with st.expander("くわしい内容"):
            st.code(str(e), language=None)
        return
    if df.empty:
        st.error("CSVファイルにデータの行がありません。")
        return
    st.success(f"{len(df)}行のデータを読み込みました。")

    st.subheader("列の確認")
    st.caption("表のどの列が何の数字かを、確かめてください。ちがっていたら選び直せます。")
    options = [NONE_LABEL] + list(df.columns)
    mapping = {}
    cols = st.columns(4)
    for i, (role, keywords) in enumerate(COLUMN_ROLES.items()):
        with cols[i % 4]:
            mapping[role] = st.selectbox(
                role, options, index=guess_column(df.columns, keywords), key=f"map_{role}"
            )

    data = build_data(df, mapping)
    metrics = [m for m in NUMERIC_ROLES if m in data.columns]
    if not metrics:
        st.warning("数字の列(いいね・リーチなど)を1つ以上選んでください。")
        return

    # CSVに何か月分も入っていても、1か月だけを選んで振り返れるようにする
    period = ALL_MONTHS_LABEL
    if "投稿日時" in data.columns:
        parsed = pd.to_datetime(data["投稿日時"], errors="coerce")
        months = sorted(parsed.dropna().dt.strftime("%Y-%m").unique(), reverse=True)
        if months:
            period = st.selectbox(
                "どの月を振り返りますか?", months + [ALL_MONTHS_LABEL], key="report_period"
            )
            if period != ALL_MONTHS_LABEL:
                data = data[parsed.dt.strftime("%Y-%m") == period]
        else:
            st.warning("日付を読み取れなかったので、すべての行をまとめて振り返ります。")

    blanks = {m: int(data[m].isna().sum()) for m in metrics if data[m].isna().any()}
    if blanks:
        detail = "、".join(f"{m} {n}件" for m, n in blanks.items())
        st.warning(f"数字として読めないところがありました({detail})。そこは計算から外しています。")

    st.subheader(f"データ({period}:{len(data)}件)")
    st.dataframe(data, hide_index=True, width="stretch")
    summary = make_summary(data)
    with st.expander("AIにわたす集計結果を見る"):
        st.text(summary)

    wish = st.text_input("報告書で気をつけてほしいこと(なくてもOK)", placeholder="例:店長さんに見せるので、やさしい言葉で")
    key = f"report|{store['name']}"
    if st.button("📝 報告書を作る", type="primary"):
        def work():
            prompt = f"""お好み焼き店のInstagram月次報告書を作ってください。

{store_block(store)}

# 対象の期間
{period}

# 集計結果(数字はプログラムで計算した正しい値)
{summary}

# 気をつけること
{wish or '特になし'}

# 書き方(Markdownで、次の見出しで)
## 今月のまとめ
3行くらいで。
## 良かった点(3つ)
それぞれ根拠の数字をそえる。
## 改善点(3つ)
それぞれ根拠の数字と、具体的な直し方をそえる。
## 次月の提案(3つ)
すぐ実行できる形で、形式(フィード/リール/ストーリー)も書く。

条件:
- 数字は集計結果にあるものだけを使い、作らない。
- 集計にない項目(フォロワー数や来店数など)について、良くなった・悪くなったと書かない。
- 投稿数が少ない形式については、言い切らずに「件数が少ないので参考程度」と書く。
"""
            text, _ = call_gemini(prompt)
            return {"text": text}

        run_and_save(key, f"月次報告書({period})", store, work)
    show_result(key, f"{store['name']}_月次報告書")


# ===== 6. プロフィール見直し =====
PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "bios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concept": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["concept", "text"],
            },
        },
        "image_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["bios", "image_points"],
}


def check_profile(data):
    """3案あるか、文字数がオーバーしていないかを確かめる"""
    bios = data.get("bios", [])
    problems = []
    if len(bios) != 3:
        problems.append(f"プロフィール文は3案必要なのに{len(bios)}案になっています")
    for i, bio in enumerate(bios, 1):
        if len(bio.get("text", "")) > BIO_MAX_CHARS:
            problems.append(f"案{i}が{len(bio['text'])}文字です。{BIO_MAX_CHARS}文字以内にしてください")
    return problems


def page_profile(store):
    st.header("6. プロフィール見直し")
    st.write("Instagramのプロフィール文を3つ考えて、プロフィール画像の直すところも教えます。")
    if not need_store(store):
        return
    current = st.text_area("今のプロフィール文(なくてもOK)", key="profile_current")
    target = st.text_input("来てほしいお客さん(なくてもOK)", placeholder="例:近くに住む家族づれ", key="profile_target")
    uploaded = st.file_uploader(
        "今のプロフィール画像(なくてもOK)。入れるとAIが実際に見て答えます。",
        type=["png", "jpg", "jpeg", "webp"],
    )
    image = None
    if uploaded is not None:
        if uploaded.size > MAX_IMAGE_MB * 1024 * 1024:
            st.error(f"画像が大きすぎます。{MAX_IMAGE_MB}MB以下の画像にしてください。")
            return
        st.image(uploaded, width=160)
        image = {"data": uploaded.getvalue(), "mime": uploaded.type or "image/jpeg"}

    key = f"profile|{store['name']}"
    if st.button("✨ 見直す", type="primary"):
        def work():
            image_note = (
                "最初につけた画像が今のプロフィール画像です。画像を実際に見て、良い点と直す点を具体的に書いてください。"
                if image
                else "画像はもらっていません。お好み焼き店のプロフィール画像で確かめるべき点を書いてください。"
            )
            prompt = f"""お好み焼き店のInstagramプロフィールを見直してください。

{store_block(store)}

# 今のプロフィール文
{current or 'なし'}

# 来てほしいお客さん
{target or '未設定'}

# 作るもの
- bios:プロフィール文をちょうど3案。conceptに案のねらい、textに本文を入れる。
  本文は1案{BIO_MAX_CHARS}文字以内(改行やスペースも数える)。場所・営業時間・予約方法のうち大事なものを入れ、
  改行で読みやすくする。3案はねらいを変える。
- image_points:プロフィール画像の改善ポイントを3〜5個。小さい丸の画像で見ても伝わるかを意識する。
  {image_note}
"""
            data, problems = call_gemini_checked(prompt, PROFILE_SCHEMA, check_profile, image=image)
            parts, shown = ["## プロフィール文の案"], ["## プロフィール文の案"]
            for i, bio in enumerate(data.get("bios", []), 1):
                body, count = bio.get("text", ""), len(bio.get("text", ""))
                head = f"### 案{i}:{bio.get('concept', '')}({count}文字)"
                parts += [head, body]
                shown += [head, "\n".join(f"> {line}  " for line in body.splitlines())]
            title = "## プロフィール画像の改善ポイント" + ("" if image else "(画像なしの一般的なチェック項目)")
            points = [f"- {p}" for p in data.get("image_points", [])]
            parts += [title] + points
            shown += [title, "\n".join(points)]
            warnings = [f"条件どおりに作れなかったところがあります:{p}" for p in problems]
            return {"text": "\n\n".join(parts), "display": "\n\n".join(shown), "warnings": warnings}

        run_and_save(key, "プロフィールの見直し", store, work)
    show_result(key, f"{store['name']}_プロフィール")


# ===== 7. 共通テンプレ =====
def page_template(store):
    st.header("7. 共通テンプレ")
    st.write("フィードとリールの最後につける共通ページと、投稿の文の最後につけるお店の案内を作ります。")
    if not need_store(store):
        return
    account = st.text_input("Instagramのアカウント名(なくてもOK)", placeholder="例:@okonomiyaki_taro")
    wish = st.text_input("いちばんしてほしいこと(なくてもOK)", placeholder="例:電話で予約してほしい")
    trend = trend_memo_prompt_block()

    key = f"template|{store['name']}"
    if st.button("📄 テンプレを作る", type="primary"):
        def work():
            prompt = f"""お好み焼き店のInstagramで毎回使う、共通テンプレートを作ってください。

{store_block(store)}

# アカウント名
{account or '〇〇'}

# いちばんしてほしい行動
{wish or '来店・予約'}
{trend}

# 書き方(Markdownで、次の見出しで)
## フィード最後の共通ページ(3案)
それぞれ、ページに入れる文字と、配置や写真のアイデアを書く。文字は少なく、スマホで読める量にする。
## リール最後の共通ページ(3案)
数秒で読める短い文字と、見せ方を書く。
## キャプション末尾の店舗情報
登録されたお店の情報をそのまま使い、コピーして貼れる形でコードブロックに入れる。未登録の項目は書かない。
## キャプション末尾の案内文(5案)
保存・フォロー・予約などをやさしくうながす短い文を、ちがう言い方で。
Instagramのキャプション内のURLは押せないので、リンクは「プロフィールのリンクから」と案内する。
"""
            text, _ = call_gemini(prompt)
            return {"text": text}

        run_and_save(key, "共通テンプレ", store, work)
    show_result(key, f"{store['name']}_共通テンプレ")


# 左のメニューに出す機能(名前: 画面を作る関数)
PAGES = {
    "1. トレンド調査": page_trend,
    "2. 月次企画": page_plan,
    "3. 動画企画(月4本)": page_video,
    "4. 現地撮影の準備": page_shoot,
    "5. 月次の振り返り": page_report,
    "6. プロフィール見直し": page_profile,
    "7. 共通テンプレ": page_template,
}


# ===== ここからアプリの本体 =====
def main():
    st.set_page_config(page_title="お好み焼き 運用パック", page_icon="🥢", layout="wide")

    try:
        stores = load_stores()
    except AppError as e:
        st.error(str(e))
        st.stop()

    store, page = sidebar(stores)

    # 保存や削除のあとに、小さなお知らせを出す
    flash = st.session_state.pop("flash", None)
    if flash:
        st.toast(flash, icon="✅")

    if store:
        st.caption(f"今のお店:{store['name']}")
    PAGES[page](store)


main()
