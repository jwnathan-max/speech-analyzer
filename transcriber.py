"""
transcriber.py
Gemini로 강연 음성을 텍스트로 받아쓴다. (app.py 웹 앱과 extractor.py CLI가 함께 사용)

Public API:
  transcribe(audio_bytes, suffix, *, api_key) -> str
"""

import io
import time

from google import genai
from google.genai import types

MODEL = "gemini-3.8-flash"

_MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".aac": "audio/aac",
}

_PROMPT = (
    "이 음성은 공개 강연 녹음입니다. 들리는 말을 처음부터 끝까지 빠짐없이 그대로 받아쓰세요.\n"
    "- 강연에서 사용한 언어 그대로 적을 것 (한국어 강연은 한국어, 영어 강연은 영어). 번역 금지.\n"
    "- 요약·생략·의역·문장 다듬기 금지. 말한 그대로 적을 것.\n"
    "- 성경 구절 낭독은 따옴표로 묶어 그대로 적을 것.\n"
    "- 타임스탬프, 화자 표시, 설명 문구 없이 본문만 출력.\n"
    "- 내용 흐름에 따라 적당히 문단을 나눌 것."
)


def transcribe(audio_bytes: bytes, suffix: str, *, api_key: str) -> str:
    """오디오 바이트를 Gemini에 올려 받아쓰기 결과를 반환한다."""
    if not api_key:
        raise ValueError("GEMINI_API_KEY가 설정되지 않았습니다.")

    mime = _MIME_TYPES.get(suffix.lower(), "audio/mpeg")
    client = genai.Client(api_key=api_key)

    # 한글 파일 이름은 업로드 헤더에서 오류가 나므로 바이트로 올리고 이름은 영문으로 지정
    uploaded = client.files.upload(
        file=io.BytesIO(audio_bytes),
        config={"mime_type": mime, "display_name": f"talk{suffix.lower()}"},
    )
    try:
        # 업로드 직후 처리 중이면 잠시 대기
        for _ in range(60):
            if uploaded.state is None or uploaded.state.name != "PROCESSING":
                break
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)
        if uploaded.state is not None and uploaded.state.name == "FAILED":
            raise RuntimeError("Gemini가 오디오 파일을 처리하지 못했습니다.")

        response = client.models.generate_content(
            model=MODEL,
            contents=[uploaded, _PROMPT],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=65536,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini 받아쓰기 결과가 비어 있습니다.")
        return text
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
