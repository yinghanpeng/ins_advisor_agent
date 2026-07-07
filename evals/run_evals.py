"""Local eval runner."""

# 文件说明：
# - 本文件属于项目源码，承担局部工程职责。
# - 修改前请先查看 docs/project-structure.md 中的文件职责说明。
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_core.workflow.contracts import AgentRunRequest, EvalCase  # noqa: E402
from agent_core.workflow.engine import WorkflowEngine  # noqa: E402


def load_dataset(path: Path) -> list[EvalCase]:
    """读取 JSONL 格式的本地评估集，并转换为 EvalCase。"""
    return [
        EvalCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    """运行本地评估集，所有样本通过时返回 0。"""
    dataset = load_dataset(ROOT / "evals" / "dataset.jsonl")
    engine = WorkflowEngine()
    passed = 0
    for case in dataset:
        response = engine.run(AgentRunRequest(input=case.input, metadata={"eval_id": case.id}))
        ok = bool(response.trace_id and response.answer)
        ok = ok and all(term not in response.answer for term in case.must_not_include)
        passed += int(ok)
    print(json.dumps({"total": len(dataset), "passed": passed}, ensure_ascii=False))
    return 0 if passed == len(dataset) else 1


if __name__ == "__main__":
    raise SystemExit(main())
