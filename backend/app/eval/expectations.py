from __future__ import annotations

import re
from typing import Any


_IDK_RE = re.compile(r"\b(i don't know|i do not know|not in (the )?context|unknown)\b", re.IGNORECASE)


def _contains_pattern(text: str, pattern: str) -> bool:
    p = str(pattern)
    if p.startswith("re:"):
        return re.search(p[3:], text, flags=re.IGNORECASE) is not None
    return p.lower() in text.lower()


def check_expectations(
    *,
    answer: str,
    expect: dict[str, Any] | None,
    tags: list[str] | None = None,
) -> tuple[float, dict[str, Any]]:
    if not expect:
        return 1.0, {}

    expect_idk = bool(expect.get("expect_idk") or False)
    if tags and "trap" in tags and not expect_idk:
        expect_idk = True

    must_include = expect.get("must_include") or []
    must_not_include = expect.get("must_not_include") or []
    must_include_any = expect.get("must_include_any") or []

    if not isinstance(must_include, list):
        must_include = [must_include]
    if not isinstance(must_not_include, list):
        must_not_include = [must_not_include]
    if not isinstance(must_include_any, list):
        must_include_any = [must_include_any]

    missing: list[str] = []
    forbidden: list[str] = []
    total = 0
    passed = 0

    for p in must_include:
        total += 1
        if isinstance(p, str) and _contains_pattern(answer, p):
            passed += 1
        else:
            missing.append(str(p))

    for group in must_include_any:
        total += 1
        if isinstance(group, str):
            group = [group]
        if not isinstance(group, list):
            missing.append(str(group))
            continue
        if any(isinstance(p, str) and _contains_pattern(answer, p) for p in group):
            passed += 1
        else:
            missing.append(" | ".join(str(p) for p in group))

    for p in must_not_include:
        total += 1
        if isinstance(p, str) and _contains_pattern(answer, p):
            forbidden.append(p)
        else:
            passed += 1

    if expect_idk:
        total += 1
        if _IDK_RE.search(answer or "") is not None:
            passed += 1
        else:
            missing.append("IDK_EXPECTED")

    score = (passed / total) if total > 0 else 1.0
    details: dict[str, Any] = {}
    if missing:
        details["missing"] = missing
    if forbidden:
        details["forbidden"] = forbidden
    return score, details

