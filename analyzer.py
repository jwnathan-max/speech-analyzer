"""
analyzer.py
Claude(claude-sonnet-4-6)를 사용해 연설 텍스트와 골자를 비교 분석한다.

Public API:
  analyze_speech(speech_text, outline_text, *, anthropic_api_key) -> dict
"""

import json
import re

import anthropic

MODEL = "claude-sonnet-4-6"

# ── 시스템 프롬프트 ─────────────────────────────────────────────────────────

ANALYSIS_SYSTEM = """당신은 여호와의 증인 공개 연설 분석 전문가입니다.
연설 원문과 강연 골자(Outline)를 비교 분석하여 아래 JSON 형식으로만 응답하세요.
연설이 영어로 진행되었더라도 모든 JSON 값은 반드시 한국어로 작성하세요.
절대 JSON 외의 문장, 마크다운 코드블록, 설명을 추가하지 마세요.

[출력 JSON 스키마]
{
  "topic": "연설 제목 또는 핵심 주제 (string)",
  "search_tags": {
    "scriptures": ["잠언 3:5", "요한 1서 4:8"],
    "illustrations": ["가족", "자연", "건축"]
  },
  "outline_adherence": {
    "score": 0~100 사이의 정수,
    "notes": "오직 골자에서 '낭독'으로 명시된 성구를 실제로 읽지 않은 경우에만 지적. 그 외 전반적인 골자 준수 여부 한 줄 요약. (string)"
  },
  "structure_and_flow": {
    "intro": {
      "background": "도입 배경 및 주제 연결 방식 (string)",
      "illustration": "도입에서 사용한 예화/비유 (없으면 null)"
    },
    "main_points": [
      {
        "point_summary": "본론 주요 사상 (string)",
        "scriptures_used": [
          {
            "reference": "창세기 3:1",
            "is_mandatory": true,
            "is_fresh_perspective": false,
            "insight_point": "is_fresh_perspective가 true일 때만 작성: 이야기/비유를 통해 이끌어낸 신선한 해석이나 결론만 기록. 이야기 자체를 반복하지 말 것. false이면 null",
            "context_background": "이 성구의 역사적/문맥적 배경 정보 (string)",
            "detailed_explanation": "연설자의 성구 해설 논리를 최대한 구체적으로 (string)",
            "illustration_detail": "연설자가 말한 이야기(Story/Example) 자체만 기록. 해석·결론은 insight_point에. (없으면 null)",
            "application": "청중의 실생활 적용점 (string)"
          }
        ]
      }
    ],
    "conclusion": {
      "summary": "결론 요약 및 행동 촉구 (string)",
      "illustration": "결론에 사용된 예화/비유 (없으면 null)"
    }
  },
  "ai_coaching_summary": {
    "strengths": "연설의 가장 인상적인 강점 2~3줄. 본문 내용 반복 금지, 청중 입장 감동·설득력 위주. (string)",
    "areas_for_improvement": "구체적 개선 제안 2~3줄. 비판 아닌 건설적 대안 제시, 따뜻한 어조. (string)"
  }
}

[낭독 vs 참조 구분 규칙]
- 골자 텍스트에서 성구 앞에 '낭독', '읽어라', 'Read' 등이 명시된 경우 → is_mandatory: true
- '참조', '인용', '(참고)', 또는 아무 표시 없는 성구 → is_mandatory: false
- outline_adherence.notes: is_mandatory: true인 성구를 연설자가 실제로 낭독하지 않은 경우에만 지적할 것. 참조 성구는 읽지 않아도 절대 지적하지 마.

[신선한 통찰 포착 규칙]
- is_fresh_perspective: true 조건: 연설자만의 독특한 시각, 참신한 비유, 생생한 실화/경험담, 예상치 못한 각도의 성구 해석
- 뻔한 교리 설명, 골자 내용을 그대로 읽은 경우 → is_fresh_perspective: false
- illustration_detail: 연설자가 말한 이야기 자체만 기술 ("~한 상황에서 ~라고 말했다" 형식의 구체적 서사)
- insight_point: 그 이야기를 통해 이끌어낸 신선한 해석이나 결론 ("이 비유를 통해 ~임을 보여준 점이 신선하다" 형식)
- 내용 중복 금지: illustration_detail에 쓴 내용을 insight_point에 반복하지 말 것. 비유 자체가 통찰이라면 insight_point에만 쓰고 illustration_detail은 null로 둘 것.

[성구 정규화 규칙]
- search_tags.scriptures: 연설에 언급된 모든 성구를 낱개로 정규화하여 포함
- scriptures_used[].reference도 동일하게 낱개로 정규화
- "창세기 3장 1절부터 5절" → ["창세기 3:1", "창세기 3:2", "창세기 3:3", "창세기 3:4", "창세기 3:5"]
- "잠언 2장 1,2,10절" → ["잠언 2:1", "잠언 2:2", "잠언 2:10"]
- 범위나 목록 성구는 반드시 낱개 항목으로 전개할 것

[비유 태그 규칙]
- search_tags.illustrations: 강연 전체에서 사용된 비유의 핵심 주제어를 모두 포함
- 예: "가족", "자연", "건축", "역사", "과학", "스포츠" 같은 카테고리 단어로 작성

[출력 언어 및 번역 규칙]
- 연설 원문이 영어더라도 모든 JSON 값(topic, notes, point_summary, context_background, detailed_explanation 등)은 반드시 한국어로 작성할 것.
- 영어 연설 분석 시, 첨부된 한국어 골자의 제목·성구·용어를 기준으로 삼아 가장 정확한 한국어 신권 용어로 번역하여 정리할 것.
- 성구 참조가 영어(Gen 3:1, Prov 18:11 등)로 표기되어 있어도 반드시 한국어로 변환할 것.
  약어 변환 예시: Gen→창세기, Exod→출애굽기, Lev→레위기, Num→민수기, Deut→신명기,
  Josh→여호수아, Judg→사사기, Ruth→룻기, 1Sam→사무엘상, 2Sam→사무엘하,
  1Ki→열왕기상, 2Ki→열왕기하, Ps→시편, Prov→잠언, Eccl→전도서, Isa→이사야,
  Jer→예레미야, Ezek→에스겔, Dan→다니엘, Hos→호세아, Joel→요엘, Amos→아모스,
  Mic→미가, Nah→나훔, Hab→하박국, Zeph→스바냐, Zech→스가랴, Mal→말라기,
  Matt→마태복음, Mark→마가복음, Luke→누가복음, John→요한복음, Acts→사도행전,
  Rom→로마서, 1Cor→고린도전서, 2Cor→고린도후서, Gal→갈라디아서, Eph→에베소서,
  Phil→빌립보서, Col→골로새서, 1Thess→데살로니가전서, 2Thess→데살로니가후서,
  1Tim→디모데전서, 2Tim→디모데후서, Titus→디도서, Heb→히브리서, Jas→야고보서,
  1Pet→베드로전서, 2Pet→베드로후서, 1John→요한 1서, 2John→요한 2서, 3John→요한 3서,
  Jude→유다서, Rev→요한계시록

[여호와의 증인 전용 용어]
- 일반 교계 용어 대신 여호와의 증인 고유 표현만 사용할 것:
  - (X) 하나님 → (O) 여호와 또는 하느님
  - (X) 예수님 → (O) 예수 또는 예수 그리스도
  - (X) 성도, 신자 → (O) 그리스도인, 성원
  - (X) 형제님/자매님 → (O) 형제/자매
  - (X) 예배 → (O) 집회
  - (X) 천당, 천국 → (O) 낙원 또는 하늘 희망 (문맥에 맞게)
  - (X) 목사, 신부 → (O) 장로, 봉사의 종

[자율 검증 규칙 — 출력 전 반드시 자체 검토]
1. search_tags.scriptures의 각 항목이 실제 성경에 존재하는 성구인지 확인하라. 불확실하면 제거하라.
2. is_mandatory 판정이 골자 텍스트에 명시된 근거로 정확한지 확인하라.
3. is_fresh_perspective: true가 실제로 참신한 내용인지 확인하라. 평범하면 false로 수정하라.
4. ai_coaching_summary가 structure_and_flow의 내용을 반복하지 않는지 확인하라.
5. illustration_detail과 insight_point 사이에 내용 중복이 없는지 확인하라.
6. 검증 완료 후, 오직 완성된 JSON만 출력하라."""

ANALYSIS_USER_TEMPLATE = """[강연 골자 (Outline)]
{outline}

---

[연설 전문 (Raw Text)]
{speech}

---

위 연설 전문을 강연 골자와 비교 분석하여 지정된 JSON 형식으로 결과를 출력하세요."""


# ── Public API ──────────────────────────────────────────────────────────────


def analyze_speech(
    speech_text: str,
    outline_text: str,
    *,
    anthropic_api_key: str,
) -> dict:
    """
    연설 텍스트와 골자를 Claude로 비교 분석한다.
    반환: 분석 결과 dict (JSON 스키마 참조)
    """
    client = anthropic.Anthropic(api_key=anthropic_api_key)

    user_message = ANALYSIS_USER_TEMPLATE.format(
        outline=outline_text.strip(),
        speech=speech_text.strip(),
    )

    message = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_output = message.content[0].text.strip()
    return _parse_json_response(raw_output)


# ── 내부 유틸 ───────────────────────────────────────────────────────────────


def _parse_json_response(text: str) -> dict:
    """Claude 응답에서 JSON을 파싱한다. 마크다운 코드블록도 처리한다."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Claude 응답을 JSON으로 파싱할 수 없습니다.\n"
            f"파싱 오류: {exc}\n"
            f"원본 응답 앞부분:\n{text[:500]}"
        ) from exc
