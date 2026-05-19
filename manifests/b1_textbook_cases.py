from __future__ import annotations

from collections.abc import Iterable

from .b1_expanded_textbook_cases import (
    EXPANDED_TEXTBOOK_CASES,
    EXPANDED_TEXTBOOK_CASES_META,
    EXPANDED_TEXTBOOK_CASES_VERSION,
)

B1_BASE_TEXTBOOK_CASES_VERSION = "manual_textbook_cases_v1"

B1_BASE_TEXTBOOK_CASES: list[dict[str, str]] = [
    {"code": "sh.688799", "date": "2025-05-12", "name": "华纳药厂(标准)"},
    {"code": "sz.300689", "date": "2025-07-18", "name": "澄天伟业(极缩)"},
    {"code": "sh.600601", "date": "2025-07-23", "name": "方正科技(蓄势)"},
    {"code": "sh.688321", "date": "2025-06-20", "name": "微芯生物(双底)"},
    {"code": "sz.002940", "date": "2025-07-11", "name": "昂利康(压轴)"},
    {"code": "sz.301076", "date": "2025-08-01", "name": "新瀚新材(激进)"},
    {"code": "sh.600184", "date": "2025-07-10", "name": "光电股份(回踩)"},
    {"code": "sz.002074", "date": "2025-08-04", "name": "国轩高科(趋势)"},
    {"code": "sh.605378", "date": "2025-08-01", "name": "野马电池(突破)"},
    {"code": "sh.600366", "date": "2025-08-06", "name": "宁波韵升(反包)"},
    {"code": "sz.000547", "date": "2025-11-12", "name": "航天发展(标准)"},
]


def _merge_unique_cases(case_groups: Iterable[Iterable[dict[str, str]]]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str], dict[str, str]] = {}
    for case_group in case_groups:
        for case_item in case_group:
            case_key = (case_item["code"], case_item["date"])
            if case_key in merged:
                continue
            merged[case_key] = {
                "code": case_item["code"],
                "date": case_item["date"],
                "name": case_item["name"],
            }
    return list(merged.values())


B1_TEXTBOOK_CASES = _merge_unique_cases(
    [B1_BASE_TEXTBOOK_CASES, EXPANDED_TEXTBOOK_CASES]
)
B1_TEXTBOOK_CASES_VERSION = (
    f"{B1_BASE_TEXTBOOK_CASES_VERSION}+{EXPANDED_TEXTBOOK_CASES_VERSION}"
)


def list_b1_base_textbook_cases() -> list[dict[str, str]]:
    return list(B1_BASE_TEXTBOOK_CASES)


def list_b1_textbook_cases() -> list[dict[str, str]]:
    return list(B1_TEXTBOOK_CASES)


__all__ = [
    "B1_BASE_TEXTBOOK_CASES",
    "B1_BASE_TEXTBOOK_CASES_VERSION",
    "B1_TEXTBOOK_CASES",
    "B1_TEXTBOOK_CASES_VERSION",
    "EXPANDED_TEXTBOOK_CASES",
    "EXPANDED_TEXTBOOK_CASES_META",
    "EXPANDED_TEXTBOOK_CASES_VERSION",
    "list_b1_base_textbook_cases",
    "list_b1_textbook_cases",
]
