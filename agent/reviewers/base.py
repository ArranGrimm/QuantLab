"""
BaseReviewer — AI 评审抽象基类 (模板方法模式)。

子类只需实现 review_single() 方法即可接入不同 LLM 后端。
基类负责: 遍历候选、断点续跑、速率控制、JSON 解析、结果持久化。
"""
from abc import ABC, abstractmethod
from pathlib import Path
import json
import re
import time


class BaseReviewer(ABC):

    def __init__(self, config: dict, prompt: str):
        self.config = config
        self.prompt = prompt
        self._output_dir = Path(config.get("review_output", "data/review"))
        self._target_date = config.get("target_date", "unknown")

    @abstractmethod
    def review_single(self, code: str, chart_path: Path, context: str) -> dict:
        """
        子类实现: 调用具体 LLM，返回解析后的评审 dict。
        返回的 dict 应至少包含 scores, total_score, verdict, comment 字段。
        """
        ...

    def run(
        self,
        candidates: list[dict],
        charts: dict[str, Path],
        contexts: dict[str, str],
    ) -> list[dict]:
        """
        模板方法: 遍历候选 -> review_single -> 汇总排序。
        支持 skip_existing 断点续跑。
        """
        results: list[dict] = []
        skip_existing = self.config.get("skip_existing", True)
        delay = self.config.get("request_delay", 3)

        for i, item in enumerate(candidates):
            code = item["code"]

            if skip_existing and self._result_exists(code):
                existing = self._load_existing(code)
                if existing:
                    results.append(existing)
                    print(f"  [{i+1}/{len(candidates)}] [SKIP] {code} — 已有结果 ({existing.get('total_score', '?')})")
                continue

            if code not in charts:
                print(f"  [{i+1}/{len(candidates)}] [MISS] {code} — 无图表, 跳过")
                continue

            try:
                result = self.review_single(code, charts[code], contexts.get(code, ""))
                result["code"] = code
                result["name"] = item.get("name", "")
                results.append(result)
                self._save_result(code, result)
                print(f"  [{i+1}/{len(candidates)}] [OK]   {code} — {result.get('verdict', '?')} ({result.get('total_score', '?')})")
            except Exception as e:
                print(f"  [{i+1}/{len(candidates)}] [ERR]  {code} — {e}")

            if i < len(candidates) - 1:
                time.sleep(delay)

        return sorted(results, key=lambda r: r.get("total_score", 0), reverse=True)

    # ---------- JSON 提取 ----------

    @staticmethod
    def extract_json(text: str) -> dict:
        """从 LLM 输出中提取 JSON (支持 markdown 代码块包裹)。"""
        match = re.search(r"```(?:json)?\s*\n(.*?)\n\s*```", text, re.DOTALL)
        raw = match.group(1) if match else text.strip()
        return json.loads(raw)

    # ---------- 持久化 ----------

    def _date_dir(self) -> Path:
        return self._output_dir / self._target_date

    def _result_path(self, code: str) -> Path:
        safe = code.replace(".", "_")
        return self._date_dir() / f"{safe}.json"

    def _result_exists(self, code: str) -> bool:
        return self._result_path(code).exists()

    def _save_result(self, code: str, result: dict):
        path = self._result_path(code)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    def _load_existing(self, code: str) -> dict | None:
        path = self._result_path(code)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
