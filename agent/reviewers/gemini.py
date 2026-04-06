"""
GeminiReviewer — 基于 Google Gemini 多模态 API 的 B1 评审实现。

输入: K 线图 (PNG) + 结构化指标文本
输出: 评分 JSON (scores, total_score, verdict, comment)
"""
import os
from pathlib import Path

from google import genai
from google.genai import types

from .base import BaseReviewer


class GeminiReviewer(BaseReviewer):

    def __init__(self, config: dict, prompt: str):
        super().__init__(config, prompt)

        api_key = os.environ.get("GEMINI_API_KEY", config.get("api_key", ""))
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY 未设置。请设置环境变量或在 config.yaml 中配置 reviewer.api_key"
            )
        self.client = genai.Client(api_key=api_key)
        self.model = config.get("model", "gemini-2.5-flash")

    def review_single(self, code: str, chart_path: Path, context: str) -> dict:
        with open(chart_path, "rb") as f:
            image_data = f.read()

        mime = "image/png" if chart_path.suffix == ".png" else "image/jpeg"
        image_part = types.Part.from_bytes(data=image_data, mime_type=mime)

        user_text = (
            f"## 股票代码: {code}\n\n"
            f"## 结构化指标\n```\n{context}\n```\n\n"
            "请结合上方 K 线图和结构化指标，按照系统提示中的评审框架进行分析，"
            "严格按要求输出 JSON。"
        )

        parts: list[types.Part] = [
            types.Part.from_text(text="【日线K线图】"),
            image_part,
            types.Part.from_text(text=user_text),
        ]

        response = self.client.models.generate_content(
            model=self.model,
            contents=[types.Content(role="user", parts=parts)],
            config=types.GenerateContentConfig(
                system_instruction=self.prompt,
                temperature=0.2,
            ),
        )

        if not response.text:
            raise RuntimeError(f"Gemini 返回空响应 ({code})")

        return self.extract_json(response.text)
