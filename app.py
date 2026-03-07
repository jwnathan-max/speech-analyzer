"""
app.py
여호와의 증인 공개강연 AI 분석 Streamlit 웹 앱.

배포 전 준비 사항:
  1. .streamlit/secrets.toml 에 아래 키 추가:
       ANTHROPIC_API_KEY = "sk-ant-..."
       OPENAI_API_KEY    = "sk-proj-..."
       NOTION_TOKEN      = "ntn_..."
       NOTION_TALKS_DB_ID = "..."
  2. outlines_pdf/ 폴더에 S-34_KO_{번호}.pdf 파일 저장.
"""

import base64
import io
import os
import re
import tempfile
from pathlib import Path

import anthropic
import pdfplumber
import pypdfium2 as pdfium
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# 로컬 개발 환경: .env 파일 자동 로드
load_dotenv()

from analyzer import analyze_speech
from notion_db import create_talk

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
.block-container { padding: 1rem 1rem 3rem; max-width: 780px; }

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
OPENAI_KEY    = _secret("OPENAI_API_KEY")
NOTION_TOKEN  = _secret("NOTION_TOKEN")
NOTION_DB_ID  = _secret("NOTION_TALKS_DB_ID")

OUTLINE_DIR   = Path("outlines_pdf")

# ── 오디오 전사 (OpenAI Whisper) ─────────────────────────────────────────────

_WHISPER_LIMIT = 25 * 1024 * 1024  # 25 MB


def transcribe_audio(uploaded_file) -> str:
    if uploaded_file.size > _WHISPER_LIMIT:
        raise ValueError(
            f"파일이 너무 큽니다 ({uploaded_file.size / 1024 / 1024:.1f} MB). "
            "Whisper API 최대 허용 크기: 25 MB."
        )
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        client = OpenAI(api_key=OPENAI_KEY)
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ko",
                response_format="text",
            )
        return _clean_text(str(result))
    finally:
        os.unlink(tmp_path)


# ── 골자 PDF 로드 ─────────────────────────────────────────────────────────────

def load_outline(outline_number: str) -> str:
    """로컬 outlines_pdf/ 폴더에서 골자 PDF를 읽어 텍스트로 반환한다."""
    pdf_path = OUTLINE_DIR / f"S-34_KO_{outline_number}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"`outlines_pdf/S-34_KO_{outline_number}.pdf` 파일을 찾을 수 없습니다. "
            "해당 PDF를 outlines_pdf/ 폴더에 추가해 주세요."
        )
    return _extract_pdf_text(pdf_path)


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

def render_results(analysis: dict, notion_url: str = "") -> None:
    topic       = analysis.get("topic", "분석 결과")
    adherence   = analysis.get("outline_adherence", {})
    score       = int(adherence.get("score", 0))
    notes       = adherence.get("notes", "")
    flow        = analysis.get("structure_and_flow", {})
    coaching    = analysis.get("ai_coaching_summary", {})
    search_tags = analysis.get("search_tags", {})
    scriptures  = search_tags.get("scriptures", [])
    illus_tags  = search_tags.get("illustrations", [])

    st.divider()

    # ── 제목 & 점수 ──────────────────────────────────────────────────────────
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

    # ── 서론 ─────────────────────────────────────────────────────────────────
    intro = flow.get("intro", {})
    if intro.get("background") or intro.get("illustration"):
        st.subheader("🔰 서론")
        if intro.get("background"):
            st.markdown(intro["background"])
        if intro.get("illustration"):
            st.markdown(
                f'<div class="fresh-callout">💡 <strong>예화/비유</strong><br>'
                f'{intro["illustration"]}</div>',
                unsafe_allow_html=True,
            )

    # ── 본론 ─────────────────────────────────────────────────────────────────
    main_points = flow.get("main_points", [])
    if main_points:
        st.subheader("📖 본론")

        for i, point in enumerate(main_points, 1):
            summary   = point.get("point_summary", f"요점 {i}")
            has_fresh = any(
                s.get("is_fresh_perspective")
                for s in point.get("scriptures_used", [])
            )
            label = ("🌟 " if has_fresh else "") + f"{i}. {summary}"

            with st.expander(label, expanded=False):
                scriptures_used = point.get("scriptures_used", [])
                if not scriptures_used:
                    st.caption("성구 정보 없음")
                    continue

                for s in scriptures_used:
                    ref          = s.get("reference", "")
                    is_mandatory = s.get("is_mandatory", False)
                    is_fresh     = s.get("is_fresh_perspective", False)
                    context_bg   = s.get("context_background", "")
                    explanation  = s.get("detailed_explanation", "")
                    illus_detail = s.get("illustration_detail")
                    insight      = s.get("insight_point")
                    application  = s.get("application", "")

                    # 성구 헤더
                    mandatory_badge = " `★낭독`" if is_mandatory else ""
                    st.markdown(f"##### 📖 {ref}{mandatory_badge}")

                    if context_bg:
                        st.markdown(f"**배경:** {context_bg}")
                    if explanation:
                        st.markdown(f"**해설:** {explanation}")

                    # 독특한 통찰 vs 일반 비유
                    if is_fresh:
                        parts: list[str] = []
                        if illus_detail:
                            parts.append(f"▸ 비유: {illus_detail}")
                        if insight:
                            parts.append(f"▸ 통찰: {insight}")
                        body = "<br>".join(parts) if parts else (insight or illus_detail or "")
                        st.markdown(
                            f'<div class="fresh-callout">'
                            f'🌟 <strong>독특한 통찰</strong><br>{body}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        if illus_detail:
                            st.markdown(f"> 💡 **비유:** {illus_detail}")

                    if application:
                        st.markdown(f"✅ **적용:** {application}")

                    st.markdown("---")

    # ── 결론 ─────────────────────────────────────────────────────────────────
    conclusion = flow.get("conclusion", {})
    if conclusion.get("summary") or conclusion.get("illustration"):
        st.subheader("🏁 결론")
        if conclusion.get("summary"):
            st.markdown(conclusion["summary"])
        if conclusion.get("illustration"):
            st.markdown(
                f'<div class="fresh-callout">💡 <strong>예화/비유</strong><br>'
                f'{conclusion["illustration"]}</div>',
                unsafe_allow_html=True,
            )

    # ── 노션 링크 ─────────────────────────────────────────────────────────────
    if notion_url:
        st.success(f"✅ **노션에 저장됨** — [노션에서 열기 →]({notion_url})")


# ── 메인 헤더 ─────────────────────────────────────────────────────────────────

st.title("🎙️ 공개강연 AI 분석기")
st.caption("공개강연 음성 분석 · Notion 자동 저장")
st.divider()

# ── 입력 폼 ───────────────────────────────────────────────────────────────────

with st.form("main_form"):
    st.subheader("📋 강연 정보 입력")

    audio_file = st.file_uploader(
        "① 음성 파일 업로드",
        type=["mp3", "m4a", "wav", "ogg", "flac", "webm"],
        help="OpenAI Whisper로 자동 전사합니다. 최대 25 MB.",
    )

    outline_number = st.text_input(
        "② 골자 번호",
        placeholder="예: 001",
        help="`outlines_pdf/S-34_KO_{번호}.pdf` 파일을 자동으로 불러옵니다.",
    )

    col1, col2 = st.columns(2)
    with col1:
        speaker = st.text_input(
            "③ 연사 이름",
            placeholder="홍길동 (선택 사항)",
        )
    with col2:
        talk_date = st.date_input(
            "④ 연설 날짜",
            value=None,
            help="선택 사항 — 비워두면 노션에 날짜가 저장되지 않습니다.",
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
    if not outline_number.strip():
        st.error("골자 번호를 입력해 주세요 (예: 001).")
        st.stop()

    num_clean = outline_number.strip().zfill(3)
    analysis: dict = {}
    notion_url = ""
    pipeline_error = ""  # status 밖에서 표시할 에러 메시지

    with st.status("📡 분석을 진행하고 있습니다...", expanded=True) as status:

        # Step 1: 음성 전사
        st.write("🎤 음성 텍스트 변환 중...")
        try:
            speech_text = transcribe_audio(audio_file)
            st.write("✅ 음성 변환 완료")
        except Exception as exc:
            pipeline_error = f"음성 변환 실패: {exc}"
            status.update(label="❌ 음성 변환 실패", state="error")

        # Step 2: 골자 PDF 로드
        if not pipeline_error:
            st.write("📄 골자 PDF 불러오는 중...")
            try:
                outline_text = load_outline(num_clean)
                st.write("✅ 골자 불러오기 완료")
            except Exception as exc:
                pipeline_error = str(exc)
                status.update(label="❌ 골자 로드 실패", state="error")

        # Step 3: AI 분석
        if not pipeline_error:
            st.write("🤖 Claude AI 심층 분석 중... (30~90초 소요)")
            try:
                analysis = analyze_speech(
                    speech_text,
                    outline_text,
                    anthropic_api_key=ANTHROPIC_KEY,
                )
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
                    "outline_number": int(num_clean),
                }
                notion_url = create_talk(
                    metadata,
                    analysis,
                    notion_token=NOTION_TOKEN,
                    db_id=NOTION_DB_ID,
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

    render_results(analysis, notion_url)
