# 文件说明：
# - 本文件是测试用例，用来验证生产级 Agent 架构中的一个或多个关键能力。
# - 测试既是质量保障，也是给新手看的最小用法示例。
from agent_core.sales_intelligence.anonymizer import anonymize_interview
from agent_core.sales_intelligence.cleaner import clean_transcript
from agent_core.sales_intelligence.extractor import extract_structured_insight
from agent_core.sales_intelligence.ingestion import ingest_raw_interview
from agent_core.sales_intelligence.segmenter import segment_by_scene


def test_interview_pipeline_masks_and_extracts_card():
    raw = ingest_raw_interview("张总电话13800138000，客户想破冰，金额500万。")
    masked, logs = anonymize_interview(raw.text)
    assert "13800138000" not in masked
    assert logs
    cleaned = clean_transcript(masked)
    segments = segment_by_scene(raw.source_id, cleaned)
    card = extract_structured_insight(segments[0], {"interviewee_role": "资深销售"})
    assert card.source_id == raw.source_id
    assert card.suitable_for_rag is True

