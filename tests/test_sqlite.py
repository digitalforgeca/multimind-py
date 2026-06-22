"""Tests for SQLite signal store — mirrors Rust sqlite.rs tests."""

from multimind.types import TrainingSignal
from multimind.signals.sqlite import SqliteSignalStore


def test_round_trip():
    store = SqliteSignalStore.in_memory()

    signal = TrainingSignal(
        model_id="test_model",
        input_text="test content",
        predicted_label="reject",
        corrected_label="store",
        original_confidence=0.52,
    )

    store.record(signal)
    assert store.count_pending("test_model") == 1
    assert store.count_pending("other_model") == 0

    signals = store.export_pending("test_model")
    assert len(signals) == 1
    assert signals[0].predicted_label == "reject"
    assert signals[0].corrected_label == "store"

    store.mark_consumed("test_model")
    assert store.count_pending("test_model") == 0


def test_multiple_models_isolated():
    store = SqliteSignalStore.in_memory()

    for i in range(3):
        store.record(TrainingSignal(
            model_id="model_a",
            input_text=f"input {i}",
            predicted_label="class_1",
            corrected_label="class_2",
            original_confidence=0.6,
        ))

    store.record(TrainingSignal(
        model_id="model_b",
        input_text="another input",
        predicted_label="x",
        corrected_label="y",
        original_confidence=None,
    ))

    assert store.count_pending("model_a") == 3
    assert store.count_pending("model_b") == 1

    # Consuming one model doesn't affect the other
    store.mark_consumed("model_a")
    assert store.count_pending("model_a") == 0
    assert store.count_pending("model_b") == 1
