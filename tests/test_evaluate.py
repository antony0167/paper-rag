

def test_judge_field_survives_truncated_json():
    from paper_rag.evaluate import judge_field
    assert judge_field('{"score": 4, "reason": "fine"}', "score") == 4
    assert judge_field('```json\n{"verdict": "partial", "reason": "the second claim is not in', "verdict") == "partial"
    assert judge_field('{"score": 5, "reason": "The candidate matches the reference on', "score") == "5"
    assert judge_field("no json here", "score") is None
