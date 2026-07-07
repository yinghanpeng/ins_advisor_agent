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
        self.directory = Path(directory)

    def save(self, card: SalesInsightCard) -> Path:
        """把一张 SalesInsightCard 保存为 JSON 文件。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{card.source_id}__{card.chunk_id}.json"
        path.write_text(card.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_all(self) -> list[SalesInsightCard]:
        """读取目录下所有 JSON 卡片，并验证为 SalesInsightCard。"""
        cards: list[SalesInsightCard] = []
        if not self.directory.exists():
            return cards
        for path in self.directory.glob("*.json"):
            cards.append(SalesInsightCard.model_validate(json.loads(path.read_text(encoding="utf-8"))))
        return cards
