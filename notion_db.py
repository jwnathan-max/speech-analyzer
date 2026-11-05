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

    # 요청당 최상위 블록 100개, 하위 포함 전체 블록 1000개 제한이 있으므로 나눠서 전송
    chunks = _chunk_blocks(all_blocks)
    page = notion.pages.create(
        parent={"database_id": db_id},
        properties=properties,
        children=chunks[0] if chunks else [],
    )
    page_id = page["id"]

    # 나머지 묶음 분할 전송 (502 에러 방지)
    for chunk in chunks[1:]:
        time.sleep(1)
        notion.blocks.children.append(block_id=page_id, children=chunk)

    return page.get("url", "")


def _count_blocks(block: dict) -> int:
    """블록 하나와 그 하위 블록의 총 개수."""
    inner = block.get(block.get("type", ""), {})
    children = inner.get("children", []) if isinstance(inner, dict) else []
    return 1 + sum(_count_blocks(c) for c in children)


def _chunk_blocks(blocks: list, max_top: int = 100, max_total: int = 800) -> list:
    """Notion 요청 제한에 맞춰 블록 목록을 여러 묶음으로 나눈다."""
    chunks: list[list] = []
    current: list = []
    total = 0
    for b in blocks:
        n = _count_blocks(b)
        if current and (len(current) >= max_top or total + n > max_total):
            chunks.append(current)
            current, total = [], 0
        current.append(b)
        total += n
    if current:
        chunks.append(current)
    return chunks


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

    def quote(text: str) -> dict:
        return {
            "object": "block", "type": "quote",
            "quote": {"rich_text": [{"type": "text", "text": {"content": _t(text)}}]},
        }

    def bold_paragraph(text: str) -> dict:
        return {
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{
                "type": "text", "text": {"content": _t(text)}, "annotations": {"bold": True},
            }]},
        }

    def scripture_toggle(s: dict) -> dict:
        """성구 하나를 토글 블록으로 만든다. 펼치면 상세 분석이 보인다."""
        fresh = "🌟 " if s.get("is_fresh_perspective") else ""
        mandatory = " ★낭독" if s.get("is_mandatory") else ""
        return {
            "object": "block", "type": "toggle",
            "toggle": {
                "rich_text": [{
                    "type": "text",
                    "text": {"content": _t(f"{fresh}📖 {s.get('reference', '')}{mandatory}")},
                    "annotations": {"bold": True},
                }],
                "children": _scripture_children(s)[:100],
            },
        }

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

    # ── 강연 흐름 분석 (서론 → 본론 → 결론, 요점별 설명·예·성구) ────────────
    talk_flow = analysis.get("talk_flow") or []
    if talk_flow:
        blocks.append(heading2("🗺️ 강연 흐름 분석"))
        blocks.append(paragraph("요점마다 어떻게 설명하고, 어떤 예·비유·성구로 강조하며, 다음 요점으로 연결했는지 강연 순서대로 정리합니다. 성구는 토글을 펼쳐 상세 분석을 볼 수 있습니다."))

        section_icons = {"서론": "🔰", "본론": "📖", "결론": "🏁"}
        illus_icons = {"예": "🧩", "비유": "💡", "경험담": "🗣️", "질문": "❓", "시각자료": "🖼️"}

        for sec in talk_flow:
            name = sec.get("section", "")
            blocks.append(heading3(f"{section_icons.get(name, '▪️')} {name}"))
            if sec.get("section_summary"):
                blocks.append(quote(sec["section_summary"]))

            for pt in sec.get("points") or []:
                level = pt.get("point_level", "")
                prefix = "↳ " if level == "부가요점" else "● "
                level_suffix = f"  [{level}]" if level else ""
                blocks.append(bold_paragraph(f"{prefix}{pt.get('point', '')}{level_suffix}"))

                if pt.get("explanation"):
                    blocks.append(paragraph_labeled("설명", pt["explanation"]))

                for il in pt.get("illustrations") or []:
                    kind = il.get("type", "예")
                    text = il.get("content", "")
                    if il.get("effect"):
                        text += f"\n↳ 효과: {il['effect']}"
                    blocks.append(callout(_t(f"{kind}: {text}"), illus_icons.get(kind, "💡"), "gray_background"))

                for s in pt.get("scriptures") or []:
                    blocks.append(scripture_toggle(s))

                if pt.get("transition"):
                    blocks.append(paragraph_labeled("➡️ 연결", pt["transition"]))

        blocks.append(divider())

    return blocks


def _scripture_children(s: dict) -> list:
    """성구 하나의 상세 분석 블록 목록 (토글 안에 들어감)."""
    children: list[dict] = []

    def paragraph(text: str) -> dict:
        return {
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": _t(text)}}]},
        }

    def paragraph_labeled(label: str, content: str) -> dict:
        return {
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [
                {"type": "text", "text": {"content": f"{label}: "}, "annotations": {"bold": True}},
                {"type": "text", "text": {"content": _t(content)}},
            ]},
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

    if process:
        def _mk(v): return "✅" if v else "▫️"
        children.append(paragraph_labeled(
            "낭설예적",
            f"{_mk(process.get('read_aloud'))} 낭독  "
            f"{_mk(process.get('explained'))} 설명  "
            f"{_mk(process.get('example_used'))} 예  "
            f"{_mk(process.get('application_made'))} 적용"
        ))
    if delivery:
        children.append(paragraph_labeled("전달기법 평가", delivery))

    if context_bg:
        children.append(paragraph_labeled("배경", context_bg))
    if translation:
        children.append(paragraph_labeled("번역판 활용", translation))
    if meditation:
        children.append(callout(f"🪞 깊은 묵상점: {meditation}", "🪞", "blue_background"))

    if is_fresh:
        parts: list[str] = []
        if illus_detail:
            parts.append(f"▸ 비유: {illus_detail}")
        if insight:
            parts.append(f"▸ 통찰: {insight}")
        if parts:
            children.append({
                "object": "block", "type": "callout",
                "callout": {
                    "rich_text": [
                        {"type": "text", "text": {"content": "🌟 독특한 통찰\n"}, "annotations": {"bold": True}},
                        {"type": "text", "text": {"content": _t("\n".join(parts))}},
                    ],
                    "icon": {"type": "emoji", "emoji": "🌟"},
                    "color": "yellow_background",
                },
            })
    elif illus_detail:
        children.append(callout(illus_detail, "💡"))

    if explanation:
        children.append(paragraph(f"해설: {explanation}"))
    if application:
        children.append(callout(application, "✅"))

    lens_labels = {
        "expression": "🔍 표현",
        "author": "🔍 필자",
        "context": "🔍 배경(심층)",
        "emotion": "🔍 감정",
        "timeline": "🔍 연대",
        "jehovah": "🔍 여호와",
    }
    lens_items = [(lbl, six_lens.get(key)) for key, lbl in lens_labels.items() if six_lens.get(key)]
    if lens_items:
        children.append({
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [{
                "type": "text", "text": {"content": "6렌즈 심층 분석"},
                "annotations": {"bold": True, "italic": True},
            }]},
        })
        for lbl, content in lens_items:
            children.append(paragraph_labeled(lbl, content))

    return children
