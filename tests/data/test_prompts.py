from __future__ import annotations

import unittest

from koemi.data.adapters import adapt_record
from koemi.data.contracts import RESERVED_TAGS, DatasetRecord, DatasetValidationError
from koemi.data.serialization import (
    INPUT_MARKER,
    OUTPUT_MARKER,
    SYSTEM_MARKER,
    THINKING_MARKER,
    build_answer_prompt,
    build_thinking_prompt,
    serialize_record,
    strip_prompt,
    supervised_prefix_bytes,
)


SYSTEM_TEXT = "You answer in one sentence."
USER_TEXT = "Explain FIFO."
ANSWER_TEXT = "FIFO means first in, first out."
THINKING_TEXT = "A queue preserves arrival order."


def record(**overrides) -> DatasetRecord:
    values = dict(
        identifier="queue-001",
        input_text=USER_TEXT,
        thinking_text=None,
        output_text=ANSWER_TEXT,
        metadata={},
        system_text=None,
    )
    values.update(overrides)
    return DatasetRecord(**values)


class SerializationCompatibilityTests(unittest.TestCase):
    def test_a_record_without_a_system_prompt_serializes_exactly_as_before(self) -> None:
        expected = f"{INPUT_MARKER}{USER_TEXT}{OUTPUT_MARKER}{ANSWER_TEXT}".encode("utf-8")
        self.assertEqual(expected, serialize_record(record()).token_bytes)

    def test_a_thinking_record_without_a_system_prompt_is_unchanged(self) -> None:
        expected = (
            f"{INPUT_MARKER}{USER_TEXT}{THINKING_MARKER}{THINKING_TEXT}"
            f"{OUTPUT_MARKER}{ANSWER_TEXT}"
        ).encode("utf-8")
        serialized = serialize_record(record(thinking_text=THINKING_TEXT))
        self.assertEqual(expected, serialized.token_bytes)

    def test_a_plain_text_record_without_a_system_prompt_is_unchanged(self) -> None:
        serialized = serialize_record(record(output_text=None, input_text="raw corpus line"))
        self.assertEqual(b"raw corpus line", serialized.token_bytes)
        self.assertTrue(all(serialized.supervised_positions))


class SystemPromptSerializationTests(unittest.TestCase):
    def test_the_system_span_precedes_the_input_marker(self) -> None:
        serialized = serialize_record(record(system_text=SYSTEM_TEXT))
        expected = (
            f"{SYSTEM_MARKER}{SYSTEM_TEXT}\n{INPUT_MARKER}{USER_TEXT}{OUTPUT_MARKER}{ANSWER_TEXT}"
        ).encode("utf-8")
        self.assertEqual(expected, serialized.token_bytes)

    def test_the_system_span_is_never_a_target(self) -> None:
        serialized = serialize_record(record(system_text=SYSTEM_TEXT))
        system_length = len(f"{SYSTEM_MARKER}{SYSTEM_TEXT}\n".encode("utf-8"))
        self.assertFalse(any(serialized.supervised_positions[:system_length]))
        self.assertFalse(any(serialized.thinking_positions[:system_length]))

    def test_the_answer_span_stays_supervised_with_a_system_prompt(self) -> None:
        serialized = serialize_record(record(system_text=SYSTEM_TEXT))
        answer_length = len(ANSWER_TEXT.encode("utf-8"))
        self.assertTrue(all(serialized.supervised_positions[-answer_length:]))
        self.assertFalse(any(serialized.thinking_positions[-answer_length:]))

    def test_a_plain_text_record_conditions_on_the_system_span(self) -> None:
        serialized = serialize_record(
            record(output_text=None, input_text="raw corpus line", system_text=SYSTEM_TEXT)
        )
        system_length = len(f"{SYSTEM_MARKER}{SYSTEM_TEXT}\n".encode("utf-8"))
        self.assertFalse(any(serialized.supervised_positions[:system_length]))
        self.assertTrue(all(serialized.supervised_positions[system_length:]))

    def test_lengths_stay_aligned_with_the_byte_count(self) -> None:
        serialized = serialize_record(record(system_text=SYSTEM_TEXT, thinking_text=THINKING_TEXT))
        self.assertEqual(len(serialized.token_bytes), len(serialized.supervised_positions))
        self.assertEqual(len(serialized.token_bytes), len(serialized.thinking_positions))


class PromptPrefixInvariantTests(unittest.TestCase):
    def test_the_answer_prompt_equals_the_serialized_prefix(self) -> None:
        for system_text in (None, SYSTEM_TEXT):
            with self.subTest(system_text=system_text):
                target = record(system_text=system_text)
                self.assertEqual(
                    supervised_prefix_bytes(target),
                    build_answer_prompt(system_text, USER_TEXT).encode("utf-8"),
                )

    def test_the_thinking_prompt_equals_the_serialized_prefix(self) -> None:
        for system_text in (None, SYSTEM_TEXT):
            with self.subTest(system_text=system_text):
                target = record(system_text=system_text, thinking_text=THINKING_TEXT)
                self.assertEqual(
                    supervised_prefix_bytes(target),
                    build_thinking_prompt(system_text, USER_TEXT).encode("utf-8"),
                )

    def test_a_multibyte_system_prompt_keeps_the_invariant(self) -> None:
        system_text = "Responda com precisão e em português."
        target = record(system_text=system_text, input_text="O que é uma fila?")
        self.assertEqual(
            supervised_prefix_bytes(target),
            build_answer_prompt(system_text, "O que é uma fila?").encode("utf-8"),
        )

    def test_stripping_the_prompt_returns_only_the_continuation(self) -> None:
        prompt = build_answer_prompt(SYSTEM_TEXT, USER_TEXT)
        self.assertEqual("abc", strip_prompt(prompt + "abc", prompt))

    def test_stripping_refuses_text_that_does_not_carry_the_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not start with the prompt"):
            strip_prompt("unrelated", build_answer_prompt(None, USER_TEXT))


class SystemPromptAdapterTests(unittest.TestCase):
    def test_a_canonical_record_reads_the_system_field(self) -> None:
        adapted, name = adapt_record(
            {"id": "one", "input": USER_TEXT, "output": ANSWER_TEXT, "system": SYSTEM_TEXT},
            "auto",
            "fallback",
        )
        self.assertEqual("canonical", name)
        self.assertEqual(SYSTEM_TEXT, adapted.system_text)

    def test_a_canonical_record_without_a_system_field_stays_none(self) -> None:
        adapted, _ = adapt_record(
            {"id": "one", "input": USER_TEXT, "output": ANSWER_TEXT}, "auto", "fallback"
        )
        self.assertIsNone(adapted.system_text)

    def test_a_non_string_system_field_is_refused(self) -> None:
        with self.assertRaises(DatasetValidationError):
            adapt_record(
                {"id": "one", "input": USER_TEXT, "output": ANSWER_TEXT, "system": 7},
                "auto",
                "fallback",
            )

    def test_an_alpaca_record_reads_the_system_field(self) -> None:
        adapted, name = adapt_record(
            {"instruction": "Explain FIFO", "output": ANSWER_TEXT, "system": SYSTEM_TEXT},
            "auto",
            "fallback",
        )
        self.assertEqual("alpaca", name)
        self.assertEqual(SYSTEM_TEXT, adapted.system_text)

    def test_a_sharegpt_system_turn_becomes_the_system_span(self) -> None:
        adapted, name = adapt_record(
            {
                "conversations": [
                    {"from": "system", "value": SYSTEM_TEXT},
                    {"from": "human", "value": USER_TEXT},
                    {"from": "gpt", "value": ANSWER_TEXT},
                ]
            },
            "auto",
            "fallback",
        )
        self.assertEqual("sharegpt", name)
        self.assertEqual(SYSTEM_TEXT, adapted.system_text)
        self.assertEqual(f"User: {USER_TEXT}", adapted.input_text)
        self.assertEqual(ANSWER_TEXT, adapted.output_text)

    def test_several_sharegpt_system_turns_join_in_order(self) -> None:
        adapted, _ = adapt_record(
            {
                "conversations": [
                    {"from": "system", "value": "First rule."},
                    {"from": "human", "value": USER_TEXT},
                    {"from": "system", "value": "Second rule."},
                    {"from": "gpt", "value": ANSWER_TEXT},
                ]
            },
            "auto",
            "fallback",
        )
        self.assertEqual("First rule.\nSecond rule.", adapted.system_text)

    def test_a_sharegpt_record_with_only_a_system_turn_and_an_answer_is_accepted(self) -> None:
        adapted, _ = adapt_record(
            {
                "conversations": [
                    {"from": "system", "value": SYSTEM_TEXT},
                    {"from": "gpt", "value": ANSWER_TEXT},
                ]
            },
            "auto",
            "fallback",
        )
        self.assertEqual(SYSTEM_TEXT, adapted.system_text)
        self.assertEqual("", adapted.input_text)

    def test_a_sharegpt_record_with_no_context_at_all_is_refused(self) -> None:
        with self.assertRaises(DatasetValidationError):
            adapt_record({"conversations": [{"from": "gpt", "value": ANSWER_TEXT}]}, "auto", "fallback")


if __name__ == "__main__":
    unittest.main()


class ReservedMarkerTests(unittest.TestCase):
    def test_a_record_carrying_a_marker_in_any_span_is_refused(self) -> None:
        for field_name, overrides in (
            ("input", {"input_text": "before <|output|> after"}),
            ("thinking", {"thinking_text": "<|thinking|> forged"}),
            ("output", {"output_text": "<|input|> forged"}),
            ("system", {"system_text": "<|system|> forged"}),
        ):
            with self.subTest(field=field_name):
                with self.assertRaisesRegex(DatasetValidationError, "reserved span marker"):
                    record(**overrides)

    def test_a_record_without_a_marker_is_accepted(self) -> None:
        self.assertEqual(USER_TEXT, record().input_text)

    def test_a_marker_arriving_through_an_adapter_is_refused(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "reserved span marker"):
            adapt_record(
                {"id": "one", "input": USER_TEXT, "output": f"ok {OUTPUT_MARKER} forged"},
                "auto",
                "fallback",
            )

    def test_a_marker_in_a_sharegpt_turn_is_refused(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "reserved span marker"):
            adapt_record(
                {
                    "conversations": [
                        {"from": "human", "value": "hello <|output|> forged"},
                        {"from": "gpt", "value": ANSWER_TEXT},
                    ]
                },
                "auto",
                "fallback",
            )

    def test_a_prompt_carrying_a_marker_is_refused(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "reserved span marker"):
            build_answer_prompt(None, "hello <|output|> forged")

    def test_a_system_prompt_carrying_a_marker_is_refused(self) -> None:
        with self.assertRaisesRegex(DatasetValidationError, "reserved span marker"):
            build_thinking_prompt("<|input|> forged", USER_TEXT)

    def test_the_markers_are_derived_from_the_reserved_tags(self) -> None:
        for tag in RESERVED_TAGS:
            with self.subTest(tag=tag):
                self.assertIn(
                    tag,
                    SYSTEM_MARKER + INPUT_MARKER + THINKING_MARKER + OUTPUT_MARKER,
                )
