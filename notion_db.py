"""
notion_db.py
Notion API 통신 모듈.

Public API:
  create_talk(metadata, analysis, *, notion_token, db_id) -> page_url
"""

import time

import httpx
from notion_client import Client

_NOTION_VERSION = "2022-06-28"
_FILE_UPLOAD_API = "https://api.notion.com/v1/file_uploads"

# ── 노션 DB 속성명 기본값 ────────────────────────────────────────────────────
# 실제 Notion DB의 속성명과 정확히 일치해야 합니다.
# 속성명이 다른 경우 여기서 직접 수정하세요.

_PROPS = {
    "title":       "주제",
    "speaker":     "연사",
    "date":        "날짜",
    "score":       "일치율",
    "source":      "출처",
    "scripture":   "주요 성구",
    "outline_num": "골자 번호",
    "illustration":"비유 주제",
}

_MAX_RT = 1990  # Notion rich_text 단일 항목 최대 길이


# ── Public API ──────────────────────────────────────────────────────────────


def create_talk(
    metadata: dict,
    analysis: dict,
    *,
    notion_token: str,
    db_id: str,
    transcript_text: str = "",
) -> str:
    """
    Talks DB에 새 연설 페이지를 생성하고 page URL을 반환한다.

    metadata 예시:
      {
        "speaker":        "홍길동",   # str, 빈칸 허용
        "date":           "2026-03-01", # YYYY-MM-DD str, 빈칸이면 저장 안 함
        "source":         "audio",
        "outline_number": 1,           # int or None
      }
    analysis: analyze_speech()의 반환 dict
    transcript_text: 연설 원문 전체(녹취록). 비어 있으면 업로드하지 않음.
    """
    notion     = Client(auth=notion_token)
    topic      = analysis.get("topic", "제목 없음")
    score      = analysis.get("outline_adherence", {}).get("score", 0)
    search_tags = analysis.get("search_tags", {})
    speaker    = metadata.get("speaker", "")

    # ── 페이지 속성 구성 ──────────────────────────────────────────────────────
    properties: dict = {
        _PROPS["title"]: {
            "title": [{"text": {"content": _t(topic)}}]
        },
        _PROPS["speaker"]: {
            "rich_text": [{"text": {"content": _t(speaker)}}]
        },
        _PROPS["score"]: {
            "number": int(score)
        },
    }

    # 날짜 (입력된 경우에만)
    if metadata.get("date"):
        properties[_PROPS["date"]] = {
            "date": {"start": metadata["date"]}
        }

    # 출처
    if metadata.get("source"):
        properties[_PROPS["source"]] = {
            "select": {"name": metadata["source"]}
        }

    # 골자 번호
    if metadata.get("outline_number") is not None:
        properties[_PROPS["outline_num"]] = {
            "number": int(metadata["outline_number"])
        }

    # 주요 성구 (multi_select)
    scriptures = search_tags.get("scriptures", [])
    if scriptures:
        properties[_PROPS["scripture"]] = {
            "multi_select": [{"name": ref} for ref in scriptures[:50]]
        }

    # 비유 주제 (multi_select)
    illustrations = search_tags.get("illustrations", [])
    if illustrations:
        properties[_PROPS["illustration"]] = {
            "multi_select": [{"name": tag} for tag in illustrations[:20]]
        }

    # ── 본문 블록 생성 ────────────────────────────────────────────────────────
    all_blocks = _build_analysis_blocks(analysis)

    # 연설 원문(녹취록)을 TXT 파일로 업로드하여 본문 끝에 첨부
    if transcript_text and transcript_text.strip():
        file_upload_id = _upload_transcript_txt(
            notion_token, topic, transcript_text.strip()
        )
        all_blocks.append({"object": "block", "type": "divider", "divider": {}})
        all_blocks.append({
            "object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [
                {"type": "text", "text": {"content": "📜 연설 원문 (녹취록)"}}
            ]},
        })
        all_blocks.append({
            "object": "block", "type": "file",
            "file": {
                "type": "file_upload",
                "file_upload": {"id": file_upload_id},
            },
        })

    # 첫 100개와 함께 페이지 생성
    page = notion.pages.create(
        parent={"database_id": db_id},
        properties=properties,
        children=all_blocks[:100],
    )
    page_id = page["id"]

    # 100개 초과 블록 분할 전송 (502 에러 방지)
    for i in range(100, len(all_blocks), 100):
        time.sleep(1)
        notion.blocks.children.append(
            block_id=page_id,
            children=all_blocks[i : i + 100],
        )

    return page.get("url", "")


def _upload_transcript_txt(notion_token: str, topic: str, text: str) -> str:
    """
    연설 원문 텍스트를 Notion File Upload API로 .txt 파일로 업로드하고
    file_upload 객체 id를 반환한다.
    """
    headers = {
        "Authorization": f"Bearer {notion_token}",
        "Notion-Version": _NOTION_VERSION,
    }
    safe_name = (topic or "녹취록").strip()[:80] or "녹취록"

    create_resp = httpx.post(
        _FILE_UPLOAD_API,
        headers={**headers, "Content-Type": "application/json"},
        json={"filename": f"{safe_name}.txt", "content_type": "text/plain"},
        timeout=30,
    )
    create_resp.raise_for_status()
    upload_info = create_resp.json()

    send_resp = httpx.post(
        upload_info["upload_url"],
        headers=headers,
        files={"file": (f"{safe_name}.txt", text.encode("utf-8"), "text/plain")},
        timeout=60,
    )
    send_resp.raise_for_status()

    return upload_info["id"]


# ── Notion 블록 빌더 ────────────────────────────────────────────────────────


def _t(text: str) -> str:
    """2000자 제한 내로 자른다."""
    return (text or "")[:_MAX_RT]


def _build_analysis_blocks(analysis: dict) -> list:
    """분석 결과 dict를 Notion 페이지 본문 블록으로 변환한다."""
    blocks: list[dict] = []

    # ── 내부 블록 생성 헬퍼 ──────────────────────────────────────────────────

    def heading2(text: str) -> dict:
        return {
            "object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": _t(text)}}]},
        }

    def heading3(text: str) -> dict:
        return {
            "object": "block", "type": "heading_3",
            "heading_3": {"rich_text": [{"type": "text", "text": {"content": _t(text)}}]},
        }

    def paragraph(text: str) -> dict:
        return {
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": _t(text)}}]},
        }

    def paragraph_labeled(label: str, content: str) -> dict:
        return {
            "object": "block", "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": f"{label}: "},
                        "annotations": {"bold": True},
                    },
                    {"type": "text", "text": {"content": _t(content)}},
                ]
            },
        }

    def callout(text: str, emoji: str, color: str = "default") -> dict:
        block: dict = {
            "object": "block", "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": _t(text)}}],
                "icon": {"type": "emoji", "emoji": emoji},
            },
        }
        if color != "default":
            block["callout"]["color"] = color
        return block

    def fresh_callout(insight: str | None, illus_detail: str | None) -> dict:
        """🌟 독특한 통찰 — 비유와 통찰을 하나의 Callout으로 합산한다."""
        parts: list[str] = []
        if illus_detail:
            parts.append(f"▸ 비유: {illus_detail}")
        if insight:
            parts.append(f"▸ 통찰: {insight}")
        body = "\n".join(parts) if parts else (insight or illus_detail or "")
        return {
            "object": "block", "type": "callout",
            "callout": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {"content": "🌟 독특한 통찰\n"},
                        "annotations": {"bold": True},
                    },
                    {"type": "text", "text": {"content": _t(body)}},
                ],
                "icon": {"type": "emoji", "emoji": "🌟"},
                "color": "yellow_background",
            },
        }

    def divider() -> dict:
        return {"object": "block", "type": "divider", "divider": {}}

    # ── 골자 일치도 ───────────────────────────────────────────────────────────
    adherence = analysis.get("outline_adherence", {})
    score     = adherence.get("score", "-")
    notes     = adherence.get("notes", "")

    blocks.append(heading2("골자 일치도"))
    blocks.append(paragraph(f"점수: {score}점"))
    if notes:
        blocks.append(paragraph(_t(notes)))
    blocks.append(divider())

    # ── AI 강연 심층 리뷰 (상단으로 이동, Streamlit과 동일 순서) ──────────────
    coaching = analysis.get("ai_coaching_summary", {})
    if coaching:
        blocks.append(heading2("💡 AI 강연 심층 리뷰"))
        if coaching.get("strengths"):
            blocks.append(paragraph_labeled("⭐ 강점", coaching["strengths"]))
        if coaching.get("areas_for_improvement"):
            blocks.append(paragraph_labeled("🌱 개선 제안", coaching["areas_for_improvement"]))
        blocks.append(divider())

    # ── 골자 계층 분석 (서론/본론/결론 리포트) ───────────────────────────────
    hierarchy = analysis.get("outline_hierarchy_analysis") or {}
    intro_report = hierarchy.get("intro_report", "")
    body_report = hierarchy.get("body_report", "")
    conclusion_report = hierarchy.get("conclusion_report", "")
    if intro_report or body_report or conclusion_report:
        blocks.append(heading2("🧭 골자 계층 분석"))
        blocks.append(paragraph("각 부분의 요점이 어떤 예·비유·성구로 강조되고, 요점들이 어떻게 유기적으로 연결되는지 분석합니다."))
        if intro_report:
            blocks.append(heading3("🔰 서론"))
            blocks.append(paragraph(_t(intro_report)))
        if body_report:
            blocks.append(heading3("📖 본론"))
            blocks.append(paragraph(_t(body_report)))
        if conclusion_report:
            blocks.append(heading3("🏁 결론"))
            blocks.append(paragraph(_t(conclusion_report)))
        blocks.append(divider())

    # ── 성구 분석 (낭설예적 · 배경 · 번역판 · 묵상점) ────────────────────────
    scripture_analysis = analysis.get("scripture_analysis") or []
    if scripture_analysis:
        blocks.append(heading2("📖 성구 분석"))
        blocks.append(paragraph("각 성구를 낭설예적(낭독·설명·예·적용)으로 다뤘는지, 배경·다른 번역판·깊은 묵상점을 이끌어냈는지 분석합니다."))

        for s in scripture_analysis:
            ref          = s.get("reference", "")
            section      = s.get("belongs_to_section", "")
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

            if ref:
                suffix = " ★낭독" if is_mandatory else ""
                section_suffix = f"  ({section})" if section else ""
                blocks.append(heading3(_t(f"📖 {ref}{suffix}{section_suffix}")))

            if process:
                def _mk(v): return "✅" if v else "▫️"
                blocks.append(paragraph_labeled(
                    "낭설예적",
                    f"{_mk(process.get('read_aloud'))} 낭독  "
                    f"{_mk(process.get('explained'))} 설명  "
                    f"{_mk(process.get('example_used'))} 예  "
                    f"{_mk(process.get('application_made'))} 적용"
                ))
            if delivery:
                blocks.append(paragraph_labeled("전달기법 평가", delivery))

            if context_bg:
                blocks.append(paragraph_labeled("배경", context_bg))
            if translation:
                blocks.append(paragraph_labeled("번역판 활용", translation))
            if meditation:
                blocks.append(callout(_t(f"🪞 깊은 묵상점: {meditation}"), "🪞", "blue_background"))

            if is_fresh:
                if insight or illus_detail:
                    blocks.append(fresh_callout(insight, illus_detail))
            else:
                if illus_detail:
                    blocks.append(callout(_t(illus_detail), "💡"))

            if explanation:
                blocks.append(paragraph(_t(f"해설: {explanation}")))
            if application:
                blocks.append(callout(_t(application), "✅"))

            lens_labels = {
                "expression": "🔍 표현",
                "author": "🔍 필자",
                "context": "🔍 배경(심층)",
                "emotion": "🔍 감정",
                "timeline": "🔍 연대",
                "jehovah": "🔍 여호와",
            }
            lens_items = [
                (lens_labels[key], six_lens.get(key))
                for key in lens_labels
                if six_lens.get(key)
            ]
            if lens_items:
                blocks.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": [
                        {
                            "type": "text",
                            "text": {"content": "6렌즈 심층 분석"},
                            "annotations": {"bold": True, "italic": True},
                        }
                    ]},
                })
                for label, content in lens_items:
                    blocks.append(paragraph_labeled(label, content))

    return blocks
