"""测试 main.py 这个新手入口是否可直接运行。"""

# 文件说明：
# - 本文件是测试用例，确保用户执行 python3 main.py --message 时能看到完整输出。
# - 这个测试可以防止后续重构时不小心破坏项目最重要的新手入口。

from __future__ import annotations

import subprocess
import sys


def test_main_message_entry_runs_successfully():
    """main.py 单条消息模式应该成功退出，并打印最终回答和状态流转。"""
    result = subprocess.run(
        [sys.executable, "main.py", "--message", "客户喜欢银行理财，我怎么破冰"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "最终回答" in result.stdout
    assert "state_path" in result.stdout
    assert "insurance_advisor_help" in result.stdout
