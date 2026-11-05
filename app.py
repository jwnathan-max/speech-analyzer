"""
app.py
여호와의 증인 공개강연 AI 분석 Streamlit 웹 앱.

배포 전 준비 사항:
  1. .streamlit/secrets.toml 에 아래 키 추가:
       ANTHROPIC_API_KEY = "sk-ant-..."
       GEMINI_API_KEY    = "AQ...."   (음성 → 텍스트)
       NOTION_TOKEN      = "ntn_..."
       NOTION_TALKS_DB_ID = "..."
  2. outlines_pdf/ 폴더에 S-34_KO_{번호}.pdf 파일 저장.
"""

import base64
import difflib
import io
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import anthropic
import pdfplumber
import pypdfium2 as pdfium
import streamlit as st
from docx import Document
from dotenv import load_dotenv

# 로컬 개발 환경: .env 파일 자동 로드
load_dotenv()

from analyzer import analyze_speech
from notion_db import create_talk
from transcriber import transcribe

# 배포 확인용 버전 — 코드를 바꿔 배포할 때마다 올린다 (화면 상단에 표시됨)
APP_VERSION = "v2.4 (2026-11-05) · 파일 이름으로 강연 정보 자동 입력 + 주제 확인"

# ── 페이지 설정 ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="공개강연 AI 분석기",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── 모바일 친화적 CSS ────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* 전체 패딩 조정 */
.block-container { padding: 2.5rem 1rem 3rem; max-width: 780px; }

/* 버튼 전체 너비 */
.stButton > button { width: 100%; font-size: 1rem; }

/* 독특한 통찰 콜아웃 */
.fresh-callout {
    background: linear-gradient(135deg, #fffbea 0%, #fff3cd 100%);
    border-left: 4px solid #f0c040;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.75rem;
    line-height: 1.7;
}

/* 태그 칩 */
.tag-chip {
    display: inline-block;
    background: #f0f2f6;
    color: #444;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.8rem;
    margin: 2px 3px 2px 0;
}

/* 모바일 폰트 축소 */
@media (max-width: 600px) {
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.15rem !important; }
    h3 { font-size: 1rem !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Secrets 헬퍼 (st.secrets → os.environ 순서로 폴백) ───────────────────────

def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


ANTHROPIC_KEY = _secret("ANTHROPIC_API_KEY")
GEMINI_KEY    = _secret("GEMINI_API_KEY")
NOTION_TOKEN  = _secret("NOTION_TOKEN")
NOTION_DB_ID  = _secret("NOTION_TALKS_DB_ID")

OUTLINE_DIR   = Path("outlines_pdf")

# ── 오디오 전사 (Gemini) ──────────────────────────────────────────────────────


def transcribe_audio(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    text = transcribe(uploaded_file.getvalue(), suffix, api_key=GEMINI_KEY)
    return _clean_text(text)


# ── 골자 PDF 로드 ─────────────────────────────────────────────────────────────

def load_outline(outline_number: str) -> tuple[str, str]:
    """로컬 outlines_pdf/ 폴더에서 골자 파일(PDF 또는 DOCX)을 읽는다.
    반환: (골자 본문 텍스트, 골자 제목)"""
    pdf_path = OUTLINE_DIR / f"S-34_KO_{outline_number}.pdf"
    docx_path = OUTLINE_DIR / f"S-34_KO_{outline_number}.docx"
    if pdf_path.exists():
        return _read_outline_file(pdf_path)
    if docx_path.exists():
        return _read_outline_file(docx_path)
    raise FileNotFoundError(
        f"`outlines_pdf/S-34_KO_{outline_number}.pdf` 또는 `.docx` 파일을 찾을 수 없습니다. "
        "해당 파일을 outlines_pdf/ 폴더에 추가해 주세요."
    )


def load_uploaded_outline(uploaded_file) -> tuple[str, str]:
    """화면에서 직접 올린 골자 파일(PDF/Word)을 읽는다. 반환: (본문, 제목)"""
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)
    try:
        text, title = _read_outline_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if not text.strip():
        raise ValueError("올린 골자 파일에서 글자를 읽지 못했습니다. 다른 파일로 시도해 주세요.")
    return text, title


def _read_outline_file(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".docx":
        text = _extract_docx_text(path)
        return text, _clean_outline_title(text.splitlines()[0] if text else "")
    text = _extract_pdf_text(path)
    return text, _outline_title_from_pdf(path, text)


def _outline_title_from_pdf(path: Path, outline_text: str) -> str:
    """골자 제목을 찾는다. 본문 추출은 페이지 위쪽을 잘라내므로 첫 페이지 전체에서 따로 읽는다.
    글자가 깨진 PDF(이미지로 읽은 경우)는 읽어 낸 본문의 첫 줄을 쓴다."""
    try:
        with pdfplumber.open(str(path)) as pdf:
            first = pdf.pages[0].extract_text() or ""
        if first.strip() and not _is_garbled(first):
            return _clean_outline_title(first.strip().splitlines()[0])
    except Exception:
        pass
    for line in outline_text.splitlines():
        if line.strip():
            return _clean_outline_title(line)
    return ""


def _clean_outline_title(line: str) -> str:
    line = re.sub(r"[*#]", "", line).strip()
    return re.sub(r"^제\s*\d+\s*번\s*", "", line).strip()


# ── 파일 이름에서 강연 정보 읽기 ─────────────────────────────────────────────

# 예) 공개 강연_ (166) 진정한 믿음이란 무엇이며 어떻게 나타낼 수 있는가_ - 김정일 2026. 10. 3.mp3
_KO_NAME = re.compile(
    r"\((\d{1,3})\)\s*(.+?)\s+[-–—]\s+(.+?)\s+(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})\.?\s*$"
)
# 예) Public Talks_ (115) Protect Yourself ... - Kenneth Varni 1_17_2026.mp3
_EN_NAME = re.compile(
    r"\((\d{1,3})\)\s*(.+?)\s+[-–—]\s+(.+?)\s+(\d{1,2})_(\d{1,2})_(\d{4})\s*$"
)


def parse_talk_filename(filename: str) -> dict | None:
    """정해진 형식의 파일 이름에서 골자 번호·제목·연사·날짜를 읽는다. 형식이 아니면 None."""
    stem = Path(filename).stem
    m = _KO_NAME.search(stem)
    if m:
        num, title, speaker, y, mo, d = m.groups()
    else:
        m = _EN_NAME.search(stem)
        if not m:
            return None
        num, title, speaker, mo, d, y = m.groups()
    try:
        talk_date = date(int(y), int(mo), int(d))
    except ValueError:
        talk_date = None
    # 파일 이름에 못 쓰는 '?'가 '_'로 바뀌어 있으므로 되돌린다
    title = re.sub(r"_\s*$", "?", title.strip()).replace("_ ", "? ")
    return {
        "number": num.zfill(3),
        "title": title,
        "speaker": speaker.strip(),
        "date": talk_date,
    }


def check_title_match(file_title: str, outline_title: str) -> tuple[str, str]:
    """파일 이름의 주제와 골자 제목을 비교한다. 반환: (상태, 안내 문구)
    상태: match / mismatch / unknown"""
    if not file_title or not outline_title:
        return "unknown", "골자 제목을 확인하지 못해 주제 비교를 건너뛰었습니다."
    if not re.search(r"[가-힣]", file_title):
        return "unknown", (
            f"파일 제목이 한국어가 아니어서 자동 비교하지 않았습니다. "
            f"골자 제목: 「{outline_title}」 — 같은 강연인지 확인해 주세요."
        )
    norm = lambda s: re.sub(r"[^0-9A-Za-z가-힣]", "", s)
    a, b = norm(file_title), norm(outline_title)
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    if ratio >= 0.8 or (a and b and (a in b or b in a)):
        return "match", f"주제가 골자와 일치합니다: 「{outline_title}」"
    return "mismatch", (
        f"파일 제목 「{file_title}」과 골자 제목 「{outline_title}」이 다릅니다. "
        "골자 번호가 맞는지 확인해 주세요."
    )


def _extract_docx_text(path: Path) -> str:
    doc = Document(str(path))
    parts: list[str] = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append("\t".join(cells))
    return _clean_text("\n".join(parts))


def _extract_pdf_text(path: Path) -> str:
    """pdfplumber로 먼저 추출하고, JW 커스텀폰트로 깨진 경우 Vision OCR로 전환한다."""
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            h = page.height
            cropped = page.within_bbox((0, h * 0.10, page.width, h * 0.90))
            text = cropped.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                pages.append(text)

    raw = "\n".join(pages)
    if _is_garbled(raw):
        return _vision_ocr(str(path))
    return _clean_text(raw)


def _is_garbled(text: str) -> bool:
    """Latin-1 보충 문자가 15% 이상이면 커스텀폰트 인코딩으로 판단한다."""
    if not text:
        return False
    non_ascii = [c for c in text if ord(c) > 0x7F]
    if not non_ascii:
        return False
    latin1 = sum(1 for c in non_ascii if 0x0080 <= ord(c) <= 0x00FF)
    return (latin1 / len(non_ascii)) > 0.15


def _vision_ocr(pdf_path: str) -> str:
    """pypdfium2로 PDF를 이미지로 렌더링한 뒤 Claude Vision으로 OCR 처리한다."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    doc = pdfium.PdfDocument(pdf_path)
    texts: list[str] = []

    for i in range(len(doc)):
        page = doc[i]
        bmp = page.render(scale=1.75)
        buf = io.BytesIO()
        bmp.to_pil().save(buf, format="PNG")
        img_b64 = base64.standard_b64encode(buf.getvalue()).decode()

        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "이 이미지는 여호와의 증인 한국어 연설 골자(강연 원고)입니다. "
                                "이미지에 보이는 모든 텍스트를 정확히 추출하세요. "
                                "성구 참조(예: 마태 5:3)도 포함하고 들여쓰기·줄바꿈을 보존하세요. "
                                "'No.X-KO' 코드와 저작권 줄은 제외하세요. "
                                "추출한 텍스트만 출력하고 다른 설명은 하지 마세요."
                            ),
                        },
                    ],
                }
            ],
        )
        texts.append(msg.content[0].text.strip())

    doc.close()
    return _clean_text("\n\n".join(texts))


def _clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


# ── 결과 화면 렌더링 ──────────────────────────────────────────────────────────

_LENS_LABELS = {
    "expression": "🔍 표현",
    "author": "🔍 필자",
    "context": "🔍 배경(심층)",
    "emotion": "🔍 감정",
    "timeline": "🔍 연대",
    "jehovah": "🔍 여호와",
}

_ILLUS_ICONS = {"예": "🧩", "비유": "💡", "경험담": "🗣️", "질문": "❓", "시각자료": "🖼️"}


def _render_point(pt: dict) -> None:
    """요점 하나를 카드로 표시한다: 요점 → 설명 → 예·비유 → 성구 → 연결."""
    level = pt.get("point_level", "")
    is_sub = level == "부가요점"
    with st.container(border=True):
        prefix = "↳ " if is_sub else "● "
        level_tag = f" <span class='tag-chip'>{level}</span>" if level else ""
        st.markdown(
            f"**{prefix}{pt.get('point', '')}**{level_tag}",
            unsafe_allow_html=True,
        )
        if pt.get("explanation"):
            st.markdown(f"**설명:** {pt['explanation']}")

        for il in pt.get("illustrations") or []:
            kind = il.get("type", "예")
            line = f"{_ILLUS_ICONS.get(kind, '💡')} **{kind}:** {il.get('content', '')}"
            if il.get("effect"):
                line += f"  \n↳ *효과:* {il['effect']}"
            st.markdown(line)

        for s in pt.get("scriptures") or []:
            _render_scripture(s)

        if pt.get("transition"):
            st.markdown(f"➡️ **연결:** {pt['transition']}")


def _render_scripture(s: dict) -> None:
    """성구 하나를 접이식 칸으로 표시한다."""
    ref          = s.get("reference", "")
    is_mandatory = s.get("is_mandatory", False)
    is_fresh     = s.get("is_fresh_perspective", False)
    context_bg   = s.get("context_background", "")
    translation  = s.get("translation_notes")
    meditation   = s.get("deep_meditation")
    explanation  = s.get("detailed_explanation", "")
    illus_detail = s.get("illustration_detail")
    insight      = s.get("insight_point")
    application  = s.get("application", "")
    process      = s.get("process_applied") or {}
    delivery     = s.get("delivery_technique")
    six_lens     = s.get("six_lens_analysis") or {}

    mandatory_badge = " ★낭독" if is_mandatory else ""
    fresh_badge = "🌟 " if is_fresh else ""
    label = f"{fresh_badge}📖 {ref}{mandatory_badge}"

    with st.expander(label, expanded=False):
        if process:
            def _mark(v): return "✅" if v else "▫️"
            st.markdown(
                f"**낭설예적:** "
                f"{_mark(process.get('read_aloud'))} 낭독  "
                f"{_mark(process.get('explained'))} 설명  "
                f"{_mark(process.get('example_used'))} 예  "
                f"{_mark(process.get('application_made'))} 적용"
            )
        if delivery:
            st.markdown(f"**전달기법 평가:** {delivery}")

        if context_bg:
            st.markdown(f"**배경:** {context_bg}")
        if translation:
            st.markdown(f"**번역판 활용:** {translation}")
        if meditation:
            st.info(f"🪞 **깊은 묵상점**\n\n{meditation}")

        if is_fresh:
            parts: list[str] = []
            if illus_detail:
                parts.append(f"▸ 비유: {illus_detail}")
            if insight:
                parts.append(f"▸ 통찰: {insight}")
            body = "<br>".join(parts)
            if body:
                st.markdown(
                    f'<div class="fresh-callout">'
                    f'🌟 <strong>독특한 통찰</strong><br>{body}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        elif illus_detail:
            st.markdown(f"> 💡 **비유:** {illus_detail}")

        if explanation:
            st.markdown(f"**해설:** {explanation}")
        if application:
            st.markdown(f"✅ **적용:** {application}")

        lens_items = [
            (lbl, six_lens.get(key)) for key, lbl in _LENS_LABELS.items() if six_lens.get(key)
        ]
        if lens_items:
            st.markdown("**🧩 6렌즈 심층 분석**")
            for lbl, content in lens_items:
                st.markdown(f"- **{lbl}:** {content}")


def render_results(analysis: dict, notion_url: str = "") -> None:
    topic       = analysis.get("topic", "분석 결과")
    adherence   = analysis.get("outline_adherence", {})
    score       = int(adherence.get("score", 0))
    notes       = adherence.get("notes", "")
    coaching    = analysis.get("ai_coaching_summary", {})
    search_tags = analysis.get("search_tags", {})
    scriptures  = search_tags.get("scriptures", [])
    illus_tags  = search_tags.get("illustrations", [])
    talk_flow   = analysis.get("talk_flow") or []

    st.divider()

    # ── 제목 & 점수 ──────────────────────────────────────────────────────────
    has_adherence = bool(adherence)
    if has_adherence:
        col_title, col_score = st.columns([3, 1])
        with col_title:
            st.header(f"🎙️ {topic}")
        with col_score:
            score_color = (
                "#28a745" if score >= 80 else "#fd7e14" if score >= 60 else "#dc3545"
            )
            st.markdown(
                f"<div style='text-align:center;margin-top:1.1rem'>"
                f"<span style='font-size:2rem;font-weight:bold;color:{score_color}'>{score}</span>"
                f"<span style='color:#888;font-size:1rem'>점</span></div>",
                unsafe_allow_html=True,
            )
        st.progress(score / 100)
        if notes:
            st.caption(f"📝 {notes}")
    else:
        st.header(f"🎙️ {topic}")

    # ── 태그 칩 ──────────────────────────────────────────────────────────────
    if scriptures:
        disp = scriptures[:12]
        rest = len(scriptures) - len(disp)
        chips = "".join(f'<span class="tag-chip">📖 {s}</span>' for s in disp)
        if rest > 0:
            chips += f'<span class="tag-chip">+{rest}개</span>'
        st.markdown(chips, unsafe_allow_html=True)

    if illus_tags:
        chips = "".join(f'<span class="tag-chip">🖼️ {t}</span>' for t in illus_tags)
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("")

    # ── AI 강연 심층 리뷰 ─────────────────────────────────────────────────────
    if coaching:
        st.subheader("💡 AI 강연 심층 리뷰")
        if coaching.get("strengths"):
            st.success(f"⭐ **강점**\n\n{coaching['strengths']}")
        if coaching.get("areas_for_improvement"):
            st.info(f"🌱 **개선 제안**\n\n{coaching['areas_for_improvement']}")

    # ── 강연 흐름 분석 (서론 → 본론 → 결론, 요점별 설명·예·성구) ────────────
    if talk_flow:
        st.subheader("🗺️ 강연 흐름 분석")
        st.caption("요점마다 어떻게 설명하고, 어떤 예·비유·성구로 강조하며, 다음 요점으로 연결했는지")

        section_icons = {"서론": "🔰", "본론": "📖", "결론": "🏁"}
        for sec in talk_flow:
            name = sec.get("section", "")
            st.markdown(f"### {section_icons.get(name, '▪️')} {name}")
            if sec.get("section_summary"):
                st.caption(sec["section_summary"])

            for pt in sec.get("points") or []:
                _render_point(pt)

    # ── 노션 링크 ─────────────────────────────────────────────────────────────
    if notion_url:
        st.success(f"✅ **노션에 저장됨** — [노션에서 열기 →]({notion_url})")


# ── 메인 헤더 ─────────────────────────────────────────────────────────────────

st.title("🎙️ 공개강연 AI 분석기")
st.caption(f"공개강연 음성 분석 · Notion 자동 저장 · **{APP_VERSION}**")
st.divider()

# ── 입력 폼 ───────────────────────────────────────────────────────────────────

st.subheader("📋 강연 정보 입력")

# 음성 파일은 폼 밖에 둔다 — 올리자마자 파일 이름을 읽어 아래 칸을 자동으로 채우기 위해
audio_file = st.file_uploader(
    "① 음성 파일 업로드",
    type=["mp3", "m4a", "wav", "ogg", "flac", "webm"],
    help="Gemini로 자동 전사합니다. 최대 200 MB.\n\n"
         "파일 이름이 `공개 강연_ (166) 제목 - 연사 2026. 10. 3.mp3` 형식이면 "
         "골자 번호·연사·날짜를 자동으로 채웁니다.",
)

file_info = parse_talk_filename(audio_file.name) if audio_file else None
if audio_file and st.session_state.get("_autofill_for") != audio_file.name:
    # 새 파일을 올렸을 때 한 번만 채운다 (이후 직접 고친 값은 유지)
    st.session_state["_autofill_for"] = audio_file.name
    if file_info:
        st.session_state["outline_number"] = file_info["number"]
        st.session_state["speaker"] = file_info["speaker"]
        st.session_state["talk_date"] = file_info["date"]

if file_info:
    date_txt = file_info["date"].strftime("%Y-%m-%d") if file_info["date"] else "날짜 없음"
    st.caption(
        f"🪄 파일 이름에서 자동 입력: 골자 {file_info['number']} · "
        f"{file_info['speaker'] or '연사 없음'} · {date_txt} — 아래에서 고칠 수 있습니다."
    )
elif audio_file:
    st.caption("ℹ️ 파일 이름이 정해진 형식이 아니어서 자동 입력하지 않았습니다. 필요하면 직접 입력하거나 비워 두세요.")

with st.form("main_form"):
    outline_number = st.text_input(
        "② 골자 번호",
        placeholder="예: 001",
        help="`outlines_pdf/S-34_KO_{번호}.pdf` 파일을 자동으로 불러옵니다.",
        key="outline_number",
    )
    outline_file = st.file_uploader(
        "②-1 골자 파일 직접 올리기 (선택)",
        type=["pdf", "docx"],
        help="저장된 골자가 없는 강연이면 골자 PDF(또는 Word)를 여기에 올려 주세요. "
             "올린 파일이 골자 번호보다 우선 사용됩니다.",
    )

    col1, col2 = st.columns(2)
    with col1:
        speaker = st.text_input(
            "③ 연사 이름",
            placeholder="홍길동 (선택 사항)",
            key="speaker",
        )
    with col2:
        talk_date = st.date_input(
            "④ 연설 날짜",
            value=st.session_state.get("talk_date"),
            help="선택 사항 — 비워두면 노션에 날짜가 저장되지 않습니다.",
            key="talk_date",
        )

    submitted = st.form_submit_button(
        "🔍 분석 시작",
        use_container_width=True,
        type="primary",
    )

# ── 분석 파이프라인 ───────────────────────────────────────────────────────────

if submitted:
    # 유효성 검사
    if not audio_file:
        st.error("음성 파일을 업로드해 주세요.")
        st.stop()

    num_clean = outline_number.strip().zfill(3) if outline_number.strip() else ""
    analysis: dict = {}
    notion_url = ""
    pipeline_error = ""  # status 밖에서 표시할 에러 메시지
    outline_text = ""
    title_warning = ""  # 주제-골자 불일치 시 결과 위에 다시 보여줄 문구

    with st.status("📡 분석을 진행하고 있습니다...", expanded=True) as status:

        # Step 1: 음성 전사
        st.write("🎤 음성 텍스트 변환 중...")
        try:
            speech_text = transcribe_audio(audio_file)
            st.write("✅ 음성 변환 완료")
        except Exception as exc:
            pipeline_error = f"음성 변환 실패: {exc}"
            status.update(label="❌ 음성 변환 실패", state="error")

        # Step 2: 골자 로드 (올린 파일 우선, 없으면 골자 번호로 저장된 파일)
        if not pipeline_error and (outline_file or num_clean):
            st.write("📄 골자 불러오는 중...")
            try:
                if outline_file:
                    outline_text, outline_title = load_uploaded_outline(outline_file)
                else:
                    outline_text, outline_title = load_outline(num_clean)
                st.write("✅ 골자 불러오기 완료")

                # 파일 이름의 주제와 골자 제목 비교 (다르면 경고만 하고 분석은 계속)
                if file_info:
                    title_state, title_msg = check_title_match(file_info["title"], outline_title)
                    if title_state == "match":
                        st.write(f"✅ {title_msg}")
                    elif title_state == "mismatch":
                        title_warning = title_msg
                        st.warning(f"⚠️ {title_msg}")
                    else:
                        st.write(f"ℹ️ {title_msg}")
            except FileNotFoundError as exc:
                pipeline_error = (
                    f"{exc}\n\n💡 저장된 골자가 없는 강연이면 "
                    "'②-1 골자 파일 직접 올리기'에 골자 PDF를 올려 주세요."
                )
                status.update(label="❌ 골자 로드 실패", state="error")
            except Exception as exc:
                pipeline_error = f"골자 불러오기 실패: {exc}"
                status.update(label="❌ 골자 로드 실패", state="error")

        # Step 3: AI 분석
        if not pipeline_error:
            st.write("🤖 Claude AI 심층 분석 중... (보통 3~5분 소요)")
            progress_line = st.empty()
            try:
                analysis = analyze_speech(
                    speech_text,
                    outline_text,
                    anthropic_api_key=ANTHROPIC_KEY,
                    progress=lambda msg: progress_line.write(f"⏳ {msg}"),
                )
                progress_line.empty()
                st.write("✅ AI 분석 완료")
            except Exception as exc:
                pipeline_error = f"AI 분석 실패: {exc}"
                status.update(label="❌ AI 분석 실패", state="error")

        # Step 4: 노션 업로드 (실패해도 분석 결과는 표시)
        if not pipeline_error:
            st.write("📤 Notion에 저장 중...")
            try:
                metadata = {
                    "speaker":        speaker.strip(),
                    "date":           str(talk_date) if talk_date else "",
                    "source":         "audio",
                    "outline_number": int(num_clean) if num_clean else None,
                }
                notion_url = create_talk(
                    metadata,
                    analysis,
                    notion_token=NOTION_TOKEN,
                    db_id=NOTION_DB_ID,
                    transcript_text=speech_text,
                )
                st.write("✅ Notion 업로드 완료")
                status.update(label="✅ 분석 완료!", state="complete")
            except Exception as exc:
                st.warning(f"⚠️ Notion 업로드 실패: {exc}")
                status.update(label="⚠️ 분석 완료 (Notion 업로드 실패)", state="error")

    # ── status 바깥에서 에러/결과 표시 ──────────────────────────────────────
    if pipeline_error:
        st.error(f"❌ {pipeline_error}")
        st.stop()

    if title_warning:
        st.warning(f"⚠️ {title_warning}")
    render_results(analysis, notion_url)
