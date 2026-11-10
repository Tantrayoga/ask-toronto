"""Single source of truth for the model prompt.

Training, HF eval, llama.cpp eval and serving all call render_prompt().
A template mismatch between paths would silently shift the logprobs we
measure, so no other module builds prompts by hand.
"""

from __future__ import annotations

ABSTAIN = "ABSTAIN"

SYSTEM_TEMPLATE = """You translate questions about City of Toronto open data into one SQLite SELECT query.
Reply with the SQL only. If the database cannot answer the question, reply with exactly {abstain}.

Data window: {window}

Schema:
{schema}"""

# Placeholder until the real schema is built. Only the smoke tests use it.
SMOKE_SCHEMA = """CREATE TABLE subway_stations (station_id INTEGER PRIMARY KEY, station_name TEXT, line TEXT);
CREATE TABLE subway_delays (delay_id INTEGER PRIMARY KEY, station_id INTEGER REFERENCES subway_stations,
  delay_date TEXT, -- YYYY-MM-DD
  min_delay INTEGER);"""
SMOKE_WINDOW = "2024-01-01 to 2025-12-31"


def build_messages(question: str, schema: str, window: str) -> list[dict]:
    system = SYSTEM_TEMPLATE.format(abstain=ABSTAIN, window=window, schema=schema)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]


def render_prompt(tokenizer, question: str, schema: str, window: str) -> str:
    """Render the generation prompt as a string, ending at the assistant turn.

    We render to text with the HF tokenizer's own chat template, then feed
    that exact string to every runtime. llama.cpp never applies its own
    template, so both runtimes see identical bytes.
    """
    return tokenizer.apply_chat_template(
        build_messages(question, schema, window),
        tokenize=False,
        add_generation_prompt=True,
    )
