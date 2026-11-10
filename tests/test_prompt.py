from ask_toronto.prompt import ABSTAIN, build_messages


def test_system_message_carries_schema_window_and_abstain():
    msgs = build_messages("q?", schema="CREATE TABLE t (x INT);", window="2024 to 2025")
    assert [m["role"] for m in msgs] == ["system", "user"]
    sys_msg = msgs[0]["content"]
    assert "CREATE TABLE t (x INT);" in sys_msg
    assert "2024 to 2025" in sys_msg
    assert ABSTAIN in sys_msg
    assert msgs[1]["content"] == "q?"
