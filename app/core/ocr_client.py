"""Clova OCR 호출 (기획서 8: premium-general, jpg, 5MB 이하).

호출 주소(CLOVA_OCR_URL)는 콘솔에서 만든 premium-general 도메인의 주소를 쓴다.
요청 형식은 General OCR 규격(X-OCR-SECRET 헤더, base64 이미지)을 따른다.
실제 키로 확인하기 전까지 테스트에서는 가짜 클라이언트를 쓴다.
"""
import base64
import json
import time
import urllib.error
import urllib.request
import uuid

from app import config


class OcrError(Exception):
    """OCR 호출이나 응답 처리에 실패했을 때."""


class ClovaOcrClient:
    def __init__(self, url: str, secret: str, timeout: float = 20.0) -> None:
        self.url = url
        self.secret = secret
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "ClovaOcrClient | None":
        if not (config.CLOVA_OCR_URL and config.CLOVA_OCR_SECRET):
            return None
        return cls(config.CLOVA_OCR_URL, config.CLOVA_OCR_SECRET)

    def build_payload(self, image: bytes) -> dict:
        return {
            "version": "V2",
            "requestId": uuid.uuid4().hex,
            "timestamp": int(time.time() * 1000),
            "lang": "ko",
            "resultType": "string",
            "images": [
                {
                    "format": "jpg",
                    "name": "diary",
                    "data": base64.b64encode(image).decode("ascii"),
                }
            ],
        }

    def recognize(self, image: bytes) -> str:
        body = json.dumps(self.build_payload(image)).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json", "X-OCR-SECRET": self.secret},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise OcrError("OCR 서비스 호출에 실패했습니다.") from e
        return parse_text(data)


def parse_text(data: dict) -> str:
    """OCR 응답에서 본문 텍스트를 꺼낸다. 줄바꿈 표시(lineBreak)가 있는 곳에서 줄을 나눈다."""
    try:
        image = data["images"][0]
    except (KeyError, IndexError, TypeError) as e:
        raise OcrError("OCR 응답 형식이 올바르지 않습니다.") from e
    if image.get("inferResult") != "SUCCESS":
        raise OcrError("OCR이 이미지를 처리하지 못했습니다.")

    lines: list[str] = [""]
    for field in image.get("fields", []):
        lines[-1] += field.get("inferText", "")
        if field.get("lineBreak"):
            lines.append("")
    return "\n".join(line.strip() for line in lines if line.strip())
