# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from mechtomo.qwen_refusal import (
    DIRECTION_NAME,
    DirectionBundle,
    PromptRecord,
    QwenRefusalPlant,
    load_directions,
    save_directions,
)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"
    padding_side = "right"
    truncation_side = "right"

    def __init__(self) -> None:
        self._character_ids: dict[str, int] = {}
        self._id_characters: dict[int, str] = {}

    def _character_id(self, character: str) -> int:
        if character not in self._character_ids:
            token_id = len(self._character_ids) + 10
            self._character_ids[character] = token_id
            self._id_characters[token_id] = character
        return self._character_ids[character]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self._character_id(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self._id_characters[int(token_id)] for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        assert [message["role"] for message in messages] == ["system", "user"]
        return [
            1,
            *self.encode(messages[0]["content"]),
            2,
            *self.encode(messages[1]["content"]),
            3,
            4,
        ]


def _unwrap(value):
    if isinstance(value, _FakeTensor):
        return value.values
    if isinstance(value, tuple):
        return tuple(_unwrap(item) for item in value)
    return value


class _FakeTensor:
    def __init__(self, values, torch) -> None:
        self.values = np.asarray(values)
        self._torch = torch

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def device(self):
        return "cpu"

    def __getitem__(self, key):
        return _FakeTensor(self.values[_unwrap(key)], self._torch)

    def __setitem__(self, key, value) -> None:
        self.values[_unwrap(key)] = _unwrap(value)

    def float(self):
        self._torch.float_shapes.append(self.shape)
        return _FakeTensor(self.values.astype(np.float32), self._torch)

    def mean(self):
        return _FakeTensor(self.values.mean(), self._torch)

    def cpu(self):
        return self

    def __float__(self):
        return float(self.values)


class _FakeTorch:
    long = np.int64
    float32 = np.float32
    float16 = np.float16
    bfloat16 = np.float32

    def __init__(self) -> None:
        self.float_shapes: list[tuple[int, ...]] = []

    def device(self, value: str) -> str:
        return value

    def full(self, shape, fill_value, dtype, device):
        return _FakeTensor(np.full(shape, fill_value, dtype=dtype), self)

    def zeros(self, shape, dtype, device):
        return _FakeTensor(np.zeros(shape, dtype=dtype), self)

    def as_tensor(self, values, dtype=None, device=None):
        return _FakeTensor(np.asarray(values, dtype=dtype), self)

    def arange(self, start, stop=None, device=None):
        if stop is None:
            start, stop = 0, start
        return _FakeTensor(np.arange(start, stop), self)

    def inference_mode(self):
        return nullcontext()

    def log_softmax(self, tensor, dim=-1):
        assert dim == -1
        values = tensor.values
        shifted = values - values.max(axis=dim, keepdims=True)
        result = shifted - np.log(np.exp(shifted).sum(axis=dim, keepdims=True))
        return _FakeTensor(result, self)


class _FakeModel:
    def __init__(self, torch: _FakeTorch, vocab_size: int = 128) -> None:
        self.torch = torch
        self.vocab_size = vocab_size
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"].values
        batch_size, sequence_length = input_ids.shape
        kept_rows = min(int(kwargs["logits_to_keep"]), sequence_length)
        logits = np.zeros((batch_size, kept_rows, self.vocab_size), dtype=np.float16)
        first_absolute_row = sequence_length - kept_rows
        for batch_index in range(batch_size):
            for row in range(kept_rows - 1):
                next_token = int(input_ids[batch_index, first_absolute_row + row + 1])
                logits[batch_index, row, next_token] = 20.0
        return SimpleNamespace(logits=_FakeTensor(logits, self.torch))


def _bare_plant(max_length: int = 64) -> QwenRefusalPlant:
    plant = QwenRefusalPlant.__new__(QwenRefusalPlant)
    plant.torch = _FakeTorch()
    plant.device = "cpu"
    plant.tokenizer = _FakeTokenizer()
    plant.system_prompt = "SYS"
    plant.batch_size = 8
    plant.max_length = max_length
    plant._positions = None
    return plant


def test_rejects_unsupported_actuator_position_before_optional_imports():
    with pytest.raises(ValueError, match="last_prompt_token"):
        QwenRefusalPlant(
            model_id="unused",
            revision="unused",
            layers=(),
            refusal_stems=("no",),
            compliance_stems=("yes",),
            system_prompt="system",
            position="all_tokens",
        )


def test_initialization_sets_left_padding_and_longest_stem_budget(monkeypatch):
    tokenizer = _FakeTokenizer()
    loaded_model = SimpleNamespace(
        model=SimpleNamespace(layers=[]),
        eval=lambda: loaded_model,
        to=lambda _device: loaded_model,
    )
    fake_torch = _FakeTorch()
    fake_transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: tokenizer,
        ),
        AutoModelForCausalLM=SimpleNamespace(
            from_pretrained=lambda *_args, **_kwargs: loaded_model,
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    plant = QwenRefusalPlant(
        model_id="fake",
        revision="commit",
        layers=(),
        refusal_stems=("ab", "cdef"),
        compliance_stems=("x",),
        system_prompt="SYS",
        device="cpu",
    )

    assert tokenizer.padding_side == "left"
    assert plant._max_stem_len == 4
    assert plant._logits_to_keep == 5


def test_user_is_truncated_before_templating_and_longest_stem_is_reserved():
    plant = _bare_plant(max_length=15)
    plant._max_stem_len = 4

    prefix = plant._prefix_ids("abcdefghij")

    system_ids = plant.tokenizer.encode("SYS")
    user_ids = plant.tokenizer.encode("abcd")
    assert prefix == [1, *system_ids, 2, *user_ids, 3, 4]
    assert len(prefix) + plant._max_stem_len == plant.max_length


def test_chat_template_accepts_batch_encoding_shape():
    plant = _bare_plant()
    original = plant.tokenizer.apply_chat_template

    def batch_encoding(*args, **kwargs):
        return {"input_ids": [original(*args, **kwargs)]}

    plant.tokenizer.apply_chat_template = batch_encoding
    assert plant._chat_prefix_ids("abc") == original(
        [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "abc"},
        ],
        tokenize=True,
        add_generation_prompt=True,
    )


def test_padding_is_left_aligned():
    plant = _bare_plant()

    input_ids, attention_mask = plant._padded_batch([[7, 8], [9, 10, 11, 12]])

    assert input_ids.values.tolist() == [[0, 0, 7, 8], [9, 10, 11, 12]]
    assert attention_mask.values.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]


def test_continuation_offsets_and_selected_logit_memory_with_fake_model():
    plant = _bare_plant(max_length=32)
    stems = ("ab", "cde")
    plant._stem_token_ids = {
        stem: tuple(plant.tokenizer.encode(stem))
        for stem in stems
    }
    plant._max_stem_len = 3
    plant._logits_to_keep = 4
    plant.model = _FakeModel(plant.torch)
    record = PromptRecord("p", "user", "harmful", "family", "test")

    scores = plant._continuation_scores([record], stems, None, None)

    assert scores.shape == (1, 2)
    assert np.all(scores > -1e-4)
    assert plant.model.calls[0]["logits_to_keep"] == 4
    assert plant.model.calls[0]["use_cache"] is False
    assert plant._positions.values.tolist() == [11, 10]
    assert plant.torch.float_shapes == [(2, 128), (3, 128)]


def test_direction_bundle_records_content_contrast_name(tmp_path):
    bundle = DirectionBundle(
        layers=(1,),
        directions=np.asarray([[0.3, 0.4]]),
        residual_norms=np.asarray([10.0]),
        base_fraction=0.05,
        model_id="fake/model",
        model_revision="commit",
    )
    path = tmp_path / "directions.npz"

    save_directions(path, bundle)
    loaded = load_directions(path)

    assert loaded.direction_name == DIRECTION_NAME
    assert DIRECTION_NAME == "harmful-vs-benign content contrast"
