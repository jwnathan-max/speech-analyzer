"""
scripture_parser.py
성구 범위/목록 표현을 개별 성구 배열로 전개하는 유틸리티.

지원 패턴 (로컬 파싱):
  - "창세기 3:1~5"       → ["창세기 3:1", ..., "창세기 3:5"]
  - "창세기 3:1-5"       → 동일
  - "잠언 2:1,2,10"      → ["잠언 2:1", "잠언 2:2", "잠언 2:10"]
  - "잠언 2:1, 2, 10"    → 동일

자연어 표현(예: "창세기 3장 1절부터 5절")은 analyzer.py에서 Claude가 처리 후
이미 표준화된 형태로 넘어오므로, 여기서는 표준화 후 파싱만 담당.
"""

import re
from typing import List


def expand_scripture_ref(reference: str) -> List[str]:
    """
    단일 성구 참조 문자열을 개별 성구 목록으로 전개한다.
    예) "창세기 3:1~5" → ["창세기 3:1", "창세기 3:2", "창세기 3:3", "창세기 3:4", "창세기 3:5"]
    예) "잠언 2:1,2,10"  → ["잠언 2:1", "잠언 2:2", "잠언 2:10"]
    이미 단일 성구(예: "창세기 3:1")면 그대로 반환.
    """
    reference = reference.strip()

    # 패턴: "책 장:절~절" 또는 "책 장:절-절"
    range_match = re.match(
        r"^(.+?)\s+(\d+):(\d+)\s*[~\-–]\s*(\d+)$", reference
    )
    if range_match:
        book = range_match.group(1).strip()
        chapter = range_match.group(2)
        start = int(range_match.group(3))
        end = int(range_match.group(4))
        return [f"{book} {chapter}:{v}" for v in range(start, end + 1)]

    # 패턴: "책 장:절,절,절" (쉼표 구분)
    comma_match = re.match(
        r"^(.+?)\s+(\d+):([\d,\s]+)$", reference
    )
    if comma_match:
        book = comma_match.group(1).strip()
        chapter = comma_match.group(2)
        verses_str = comma_match.group(3)
        verses = [v.strip() for v in verses_str.split(",") if v.strip().isdigit()]
        if len(verses) > 1:
            return [f"{book} {chapter}:{v}" for v in verses]

    # 이미 단일 성구 형태이거나 파싱 불가 → 그대로 반환
    return [reference]


def expand_all(references: List[str]) -> List[str]:
    """
    성구 참조 목록 전체를 개별 성구로 전개하고 중복을 제거한다.
    """
    result = []
    seen = set()
    for ref in references:
        for expanded in expand_scripture_ref(ref):
            if expanded not in seen:
                seen.add(expanded)
                result.append(expanded)
    return result


def parse_search_query_locally(query: str) -> List[str]:
    """
    CLI 검색어에서 성구 패턴을 로컬에서 직접 파싱한다.
    "잠언 1:1~3 찾아줘" 또는 "잠언 2:1,2,10 연설 있어?" 같은 입력 처리.
    파싱 실패 시 빈 리스트 반환(→ analyzer.py의 Claude가 처리).
    """
    # 성구 패턴 탐지: 한글 책이름 + 장:절 (범위/목록 포함)
    patterns = [
        # 범위: 잠언 1:1~3 or 잠언 1:1-3
        r"([가-힣]+)\s+(\d+):(\d+)\s*[~\-–]\s*(\d+)",
        # 쉼표 목록: 잠언 2:1,2,10
        r"([가-힣]+)\s+(\d+):([\d,\s]+)",
        # 단일: 잠언 1:1
        r"([가-힣]+)\s+(\d+):(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            # 매칭된 부분만 추출해 expand_scripture_ref에 전달
            raw = match.group(0).strip()
            # 후행 조사 제거 (을,를,이,가,에서 등)
            raw = re.sub(r"[을를이가에서은는도로]$", "", raw).strip()
            expanded = expand_scripture_ref(raw)
            if expanded:
                return expanded

    return []
