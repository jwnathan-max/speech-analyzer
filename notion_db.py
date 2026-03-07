"""
notion_db.py
Notion API 통신 모듈.

Public API:
  create_talk(metadata, analysis, *, notion_token, db_id) -> page_url
"""

import time

from notion_client import Client

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


# ── Notion 블록 빌더 ────────────────────────────────────────────────────────


def _t(text: str) -> str:
    """2000자 제한 내로 자른다."""
    return (text or "")[:_MAX_RT]


def _build_analysis_blocks(analysis: dict) -> list:
    """분석 결과 dict를 Notion 페이지 본문 블록으로 변환한다."""
    blocks: list[dict] = []
    flow = analysis.get("structure_and_flow", {})

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

    # ── 서론 ─────────────────────────────────────────────────────────────────
    intro = flow.get("intro", {})
    if intro:
        blocks.append(heading2("서론"))
        if intro.get("background"):
            blocks.append(paragraph(_t(intro["background"])))
        if intro.get("illustration"):
            blocks.append(callout(_t(intro["illustration"]), "💡"))
        blocks.append(divider())

    # ── 본론 ─────────────────────────────────────────────────────────────────
    for i, point in enumerate(flow.get("main_points", []), 1):
        summary = point.get("point_summary", "")
        blocks.append(heading2(_t(f"{i}. {summary}")))

        for s in point.get("scriptures_used", []):
            ref          = s.get("reference", "")
            is_mandatory = s.get("is_mandatory", False)
            is_fresh     = s.get("is_fresh_perspective", False)
            context_bg   = s.get("context_background", "")
            explanation  = s.get("detailed_explanation", "")
            illus_detail = s.get("illustration_detail")
            insight      = s.get("insight_point")
            application  = s.get("application", "")

            if ref:
                suffix = " ★낭독" if is_mandatory else ""
                blocks.append(heading3(_t(f"📖 {ref}{suffix}")))

            if context_bg:
                blocks.append(paragraph_labeled("배경", context_bg))

            if is_fresh:
                # 비유·통찰 통합 Callout
                if insight or illus_detail:
                    blocks.append(fresh_callout(insight, illus_detail))
            else:
                # 일반 비유는 단순 Callout
                if illus_detail:
                    blocks.append(callout(_t(illus_detail), "💡"))

            if explanation:
                blocks.append(paragraph(_t(f"해설: {explanation}")))
            if application:
                blocks.append(callout(_t(application), "✅"))

        blocks.append(divider())

    # ── 결론 ─────────────────────────────────────────────────────────────────
    conclusion = flow.get("conclusion", {})
    if conclusion:
        blocks.append(heading2("결론"))
        if conclusion.get("summary"):
            blocks.append(paragraph(_t(conclusion["summary"])))
        if conclusion.get("illustration"):
            blocks.append(callout(_t(conclusion["illustration"]), "💡"))
        blocks.append(divider())

    # ── AI 코칭 요약 ──────────────────────────────────────────────────────────
    coaching = analysis.get("ai_coaching_summary", {})
    if coaching:
        blocks.append(heading2("💡 AI 강연 심층 리뷰"))
        if coaching.get("strengths"):
            blocks.append(paragraph_labeled("⭐ 강점", coaching["strengths"]))
        if coaching.get("areas_for_improvement"):
            blocks.append(paragraph_labeled("🌱 개선 제안", coaching["areas_for_improvement"]))

    return blocks
