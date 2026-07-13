"""Sales insight card indexing."""

# 文件说明：
# - 本文件属于 Sales Intelligence Layer，负责销售访谈资产化、检索、合规或评估生成。
# - 原始访谈不能直接进入最终 Prompt，必须先结构化和审查。
from __future__ import annotations

import json
from pathlib import Path

from agent_core.sales_intelligence.schemas import SalesInsightCard


class SalesInsightIndexer:
    """把销售洞察卡片保存为 JSON 文件，并从目录中批量加载。"""

    def __init__(self, directory: str | Path = "data/sales_insight_cards") -> None:
        """初始化卡片目录路径。"""
        # 将字符串或 Path 统一规范为 Path 对象，后续读写使用同一目录语义。
        self.directory = Path(directory)

    def save(self, card: SalesInsightCard) -> Path:
        """把一张 SalesInsightCard 保存为 JSON 文件。"""
        # 递归创建目标目录，使首次离线入库不依赖人工预建路径。
        self.directory.mkdir(parents=True, exist_ok=True)
        # 使用来源和分块 ID 组成稳定文件名，同一卡片再次保存会显式覆盖其版本。
        path = self.directory / f"{card.source_id}__{card.chunk_id}.json"
        # 按 UTF-8 写入 Pydantic 校验后的 JSON，缩进便于离线审计。
        path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
        # 返回实际落盘路径，供调用方记录索引结果或继续生成清单。
        return path

    def load_all(self) -> list[SalesInsightCard]:
        """读取目录下所有 JSON 卡片，并验证为 SalesInsightCard。"""
        # 按发现顺序累计通过 Pydantic 结构校验的卡片对象。
        cards: list[SalesInsightCard] = []
        # 目录尚不存在表示当前没有离线资产，返回空集合而不是创建目录。
        if not self.directory.exists():
            # 返回当前新建空列表，调用方可安全继续执行空检索。
            return cards
        # 逐个加载 JSON 文件；无效文件会抛出解析或校验错误，不能静默进入生成库。
        for path in self.directory.glob("*.json"):
            # 读取 UTF-8 JSON 并经过 SalesInsightCard Schema 校验后加入结果。
            cards.append(SalesInsightCard.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        # 返回全部已验证卡片，后续 Retriever 还会执行风险和准入过滤。
        return cards
