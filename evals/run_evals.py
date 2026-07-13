"""Local eval runner."""

# 文件说明：
# - 本文件属于项目源码，承担局部工程职责。
# - 修改前请先查看 docs/project-structure.md 中的文件职责说明。
from __future__ import annotations

import json
import sys
from pathlib import Path

# ROOT 指向项目根目录，方便从 evals/ 子目录运行脚本时仍能找到 src 和 dataset。
ROOT = Path(__file__).resolve().parents[1]
# 将 src 加入 sys.path，避免必须先 pip install -e . 才能运行本地评估。
sys.path.insert(0, str(ROOT / "src"))

# Eval runner 只依赖稳定契约和 WorkflowEngine，不直接调用底层节点。
from agent_core.workflow.contracts import AgentRunRequest, EvalCase  # noqa: E402
from agent_core.workflow.engine import WorkflowEngine  # noqa: E402


def load_dataset(path: Path) -> list[EvalCase]:
    """读取 JSONL 格式的本地评估集，并转换为 EvalCase。"""
    # 每一行是一个 EvalCase JSON，对应一条可复现的 Agent 输入和断言规则。
    return [
        # 用 Pydantic 校验评估样本结构，避免脏数据进入 eval。
        EvalCase.model_validate(json.loads(line))
        # splitlines 逐行读取 JSONL。
        for line in path.read_text(encoding="utf-8").splitlines()
        # 跳过空行，方便人工编辑 dataset.jsonl。
        if line.strip()
    ]


def main() -> int:
    """运行本地评估集，所有样本通过时返回 0。"""
    # 加载本地评估集，默认路径为 evals/dataset.jsonl。
    dataset = load_dataset(ROOT / "evals" / "dataset.jsonl")
    # 复用一个 WorkflowEngine，让 eval 能覆盖 session memory 等跨轮能力。
    engine = WorkflowEngine()
    # passed 统计通过样本数。
    passed = 0
    # 逐条执行评估样本。
    for case in dataset:
        # 把 eval_id 写入 metadata，trace 中可以定位具体样本。
        response = engine.run(AgentRunRequest(input=case.input, metadata={"eval_id": case.id}))
        # 最低通过条件：必须产生 trace_id 和 answer。
        ok = bool(response.trace_id and response.answer)
        # must_not_include 用于检查合规禁词，例如保证收益、避税避债。
        ok = ok and all(term not in response.answer for term in case.must_not_include)
        # bool 转 int 后累加通过数量。
        passed += int(ok)
    # 输出机器可读 JSON，方便 CI 或脚本读取。
    print(json.dumps({"total": len(dataset), "passed": passed}, ensure_ascii=False))
    # 全部通过返回 0，否则返回 1 让 CI 失败。
    return 0 if passed == len(dataset) else 1


# 直接运行 python3 evals/run_evals.py 时执行 main；被测试导入时不自动运行。
if __name__ == "__main__":
    # 直接运行评估脚本时把失败状态传给 CI；作为模块导入时只暴露可复用 main。
    raise SystemExit(main())
