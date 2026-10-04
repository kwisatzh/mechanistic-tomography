# Experiments designed/concieved by Vijay Erramilli. Code written by Vijay Erramilli and Codex
from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


DIRECTION_NAME = "harmful-vs-benign content contrast"


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str
    label: str
    family: str
    split: str
    source: str = ""


@dataclass(frozen=True)
class DirectionBundle:
    layers: tuple[int, ...]
    directions: np.ndarray
    residual_norms: np.ndarray
    base_fraction: float
    model_id: str
    model_revision: str
    direction_name: str = DIRECTION_NAME

    def validate(self) -> None:
        if self.directions.ndim != 2 or self.directions.shape[0] != len(self.layers):
            raise ValueError("direction shape does not match layers")
        if self.residual_norms.shape != (len(self.layers),):
            raise ValueError("one residual norm is required per layer")
        norms = np.linalg.norm(self.directions, axis=1)
        expected = self.base_fraction * self.residual_norms
        if not np.allclose(norms, expected, rtol=5e-5, atol=1e-7):
            raise ValueError("directions do not have the declared finite edit norm")
        if self.direction_name != DIRECTION_NAME:
            raise ValueError(f"direction must be named {DIRECTION_NAME!r}")


def load_prompt_records(path: str | Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            try:
                record = PromptRecord(
                    prompt_id=str(value["id"]),
                    text=str(value["text"]),
                    label=str(value["label"]),
                    family=str(value["family"]),
                    split=str(value["split"]),
                    source=str(value.get("source", "")),
                )
            except KeyError as exc:
                raise ValueError(f"missing {exc.args[0]!r} on JSONL line {line_number}") from exc
            if record.label not in {"harmful", "benign"}:
                raise ValueError(f"invalid label on JSONL line {line_number}: {record.label}")
            records.append(record)
    ids = [record.prompt_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("prompt IDs must be unique")
    return records


def select_records(
    records: Sequence[PromptRecord],
    split: str,
    label: str | None = None,
) -> list[PromptRecord]:
    return [
        record
        for record in records
        if record.split == split and (label is None or record.label == label)
    ]


def save_directions(path: str | Path, bundle: DirectionBundle) -> None:
    bundle.validate()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        layers=np.asarray(bundle.layers, dtype=int),
        directions=bundle.directions,
        residual_norms=bundle.residual_norms,
        base_fraction=np.asarray([bundle.base_fraction], dtype=float),
        model_id=np.asarray([bundle.model_id]),
        model_revision=np.asarray([bundle.model_revision]),
        direction_name=np.asarray([bundle.direction_name]),
    )


def load_directions(path: str | Path) -> DirectionBundle:
    with np.load(path, allow_pickle=False) as data:
        bundle = DirectionBundle(
            layers=tuple(int(value) for value in data["layers"]),
            directions=np.asarray(data["directions"], dtype=float),
            residual_norms=np.asarray(data["residual_norms"], dtype=float),
            base_fraction=float(data["base_fraction"][0]),
            model_id=str(data["model_id"][0]),
            model_revision=str(data["model_revision"][0]),
            direction_name=(
                str(data["direction_name"][0])
                if "direction_name" in data.files
                else DIRECTION_NAME
            ),
        )
    bundle.validate()
    return bundle


class QwenRefusalPlant:
    """Thin Hugging Face plant for fixed residual edits and stem-bank scoring."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        layers: Sequence[int],
        refusal_stems: Sequence[str],
        compliance_stems: Sequence[str],
        system_prompt: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        batch_size: int = 8,
        max_length: int = 512,
        position: str = "last_prompt_token",
    ) -> None:
        if position != "last_prompt_token":
            raise ValueError("Qwen actuator position must be 'last_prompt_token'")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("Qwen runs require the qwen optional dependencies") from exc

        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.layers = tuple(int(layer) for layer in layers)
        self.refusal_stems = tuple(refusal_stems)
        self.compliance_stems = tuple(compliance_stems)
        self.system_prompt = system_prompt
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.position = position
        if not self.refusal_stems or not self.compliance_stems:
            raise ValueError("both continuation banks must be nonempty")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype: {dtype}")
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self._stem_token_ids: dict[str, tuple[int, ...]] = {}
        for stem in (*self.refusal_stems, *self.compliance_stems):
            token_ids = tuple(
                int(value)
                for value in self.tokenizer.encode(stem, add_special_tokens=False)
            )
            if not token_ids:
                raise ValueError(f"stem tokenized to nothing: {stem!r}")
            self._stem_token_ids[stem] = token_ids
        self._max_stem_len = max(len(value) for value in self._stem_token_ids.values())
        self._logits_to_keep = self._max_stem_len + 1
        if self.max_length <= self._max_stem_len:
            raise ValueError("max_length must leave room for a templated prompt and longest stem")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
        )
        self.model.eval().to(self.device)
        self.blocks = tuple(self.model.model.layers)
        if any(layer < 0 or layer >= len(self.blocks) for layer in self.layers):
            raise ValueError(f"layer selection exceeds model depth {len(self.blocks)}")
        self._positions = None

    def _chat_prefix_ids(self, user_text: str) -> list[int]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], (list, tuple)):
            if len(ids) != 1:
                raise ValueError("chat template returned more than one token sequence")
            ids = ids[0]
        prefix = [int(value) for value in ids]
        if not prefix:
            raise ValueError("chat template produced an empty prompt")
        return prefix

    def _prefix_ids(self, text: str) -> list[int]:
        """Template a prompt after truncating only its user content.

        The system message and chat headers remain intact, and every returned
        prefix leaves room for the longest preregistered continuation stem.
        """

        max_prefix_length = self.max_length - self._max_stem_len
        full_prefix = self._chat_prefix_ids(text)
        if len(full_prefix) <= max_prefix_length:
            return full_prefix

        user_ids = [
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=False)
        ]
        empty_prefix = self._chat_prefix_ids("")
        if len(empty_prefix) > max_prefix_length:
            raise ValueError(
                "max_length cannot preserve the system message and chat headers "
                "while reserving the longest stem"
            )

        truncation_side = getattr(self.tokenizer, "truncation_side", "right")
        if truncation_side not in {"left", "right"}:
            raise ValueError(f"unsupported tokenizer truncation_side: {truncation_side!r}")

        def candidate(keep: int) -> list[int]:
            if keep == 0:
                return empty_prefix
            kept_ids = user_ids[-keep:] if truncation_side == "left" else user_ids[:keep]
            kept_text = self.tokenizer.decode(
                kept_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            return self._chat_prefix_ids(kept_text)

        best = empty_prefix
        low = 1
        high = len(user_ids)
        while low <= high:
            midpoint = (low + high) // 2
            prefix = candidate(midpoint)
            if len(prefix) <= max_prefix_length:
                best = prefix
                low = midpoint + 1
            else:
                high = midpoint - 1
        return best

    def _padded_batch(self, sequences: Sequence[Sequence[int]]):
        torch = self.torch
        length = max(len(sequence) for sequence in sequences)
        ids = torch.full(
            (len(sequences), length),
            int(self.tokenizer.pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        mask = torch.zeros((len(sequences), length), dtype=torch.long, device=self.device)
        for row, sequence in enumerate(sequences):
            if not sequence:
                raise ValueError("cannot pad an empty token sequence")
            values = torch.as_tensor(sequence, dtype=torch.long, device=self.device)
            offset = length - len(sequence)
            ids[row, offset:] = values
            mask[row, offset:] = 1
        return ids, mask

    def _last_prompt_positions(
        self,
        padded_length: int,
        continuation_lengths: Sequence[int],
    ):
        positions = [
            padded_length - int(continuation_length) - 1
            for continuation_length in continuation_lengths
        ]
        if any(position < 0 for position in positions):
            raise ValueError("continuation leaves no prompt token to edit")
        return self.torch.as_tensor(
            positions,
            dtype=self.torch.long,
            device=self.device,
        )

    @contextmanager
    def _capture_hooks(self, captures: dict[int, list[np.ndarray]]) -> Iterator[None]:
        handles = []
        for layer in self.layers:
            def hook(_module, _inputs, output, layer=layer):
                hidden = output[0] if isinstance(output, tuple) else output
                rows = self.torch.arange(hidden.shape[0], device=hidden.device)
                positions = self._positions.to(hidden.device)
                values = hidden[rows, positions].detach().float().cpu().numpy()
                captures[layer].append(values)

            handles.append(self.blocks[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def _edit_hooks(self, bundle: DirectionBundle, action: np.ndarray) -> Iterator[None]:
        bundle.validate()
        action = np.asarray(action, dtype=float)
        if action.shape != (len(bundle.layers),):
            raise ValueError("action length does not match direction bundle")
        if tuple(bundle.layers) != self.layers:
            raise ValueError("plant and direction layers differ")
        handles = []
        for offset, layer in enumerate(self.layers):
            strength = float(action[offset])
            if abs(strength) <= 1e-15:
                continue

            def hook(_module, _inputs, output, offset=offset, strength=strength):
                hidden = output[0] if isinstance(output, tuple) else output
                edited = hidden.clone()
                rows = self.torch.arange(edited.shape[0], device=edited.device)
                positions = self._positions.to(edited.device)
                direction = self.torch.as_tensor(
                    bundle.directions[offset],
                    dtype=edited.dtype,
                    device=edited.device,
                )
                edited[rows, positions] = edited[rows, positions] + strength * direction
                if isinstance(output, tuple):
                    return (edited, *output[1:])
                return edited

            handles.append(self.blocks[layer].register_forward_hook(hook))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    def capture_directions(
        self,
        records: Sequence[PromptRecord],
        base_fraction: float,
    ) -> DirectionBundle:
        if base_fraction <= 0:
            raise ValueError("base_fraction must be positive")
        harmful = [record for record in records if record.label == "harmful"]
        benign = [record for record in records if record.label == "benign"]
        if not harmful or not benign:
            raise ValueError("direction construction requires harmful and benign prompts")
        captures: dict[int, dict[str, list[np.ndarray]]] = {
            layer: {"harmful": [], "benign": []} for layer in self.layers
        }
        for label, selected in (("harmful", harmful), ("benign", benign)):
            for start in range(0, len(selected), self.batch_size):
                batch = selected[start : start + self.batch_size]
                sequences = [self._prefix_ids(record.text) for record in batch]
                input_ids, attention_mask = self._padded_batch(sequences)
                self._positions = self._last_prompt_positions(
                    input_ids.shape[1],
                    [0] * len(batch),
                )
                layer_captures = {layer: [] for layer in self.layers}
                with self.torch.inference_mode(), self._capture_hooks(layer_captures):
                    self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                for layer in self.layers:
                    captures[layer][label].extend(layer_captures[layer])

        directions = []
        residual_norms = []
        for layer in self.layers:
            harmful_values = np.concatenate(captures[layer]["harmful"], axis=0)
            benign_values = np.concatenate(captures[layer]["benign"], axis=0)
            content_contrast = harmful_values.mean(axis=0) - benign_values.mean(axis=0)
            contrast_norm = float(np.linalg.norm(content_contrast))
            if contrast_norm <= 1e-12:
                raise RuntimeError(
                    f"degenerate {DIRECTION_NAME} at layer {layer}"
                )
            combined = np.concatenate([harmful_values, benign_values], axis=0)
            residual_norm = float(np.median(np.linalg.norm(combined, axis=1)))
            directions.append(
                content_contrast / contrast_norm * (base_fraction * residual_norm)
            )
            residual_norms.append(residual_norm)
        bundle = DirectionBundle(
            layers=self.layers,
            directions=np.asarray(directions, dtype=float),
            residual_norms=np.asarray(residual_norms, dtype=float),
            base_fraction=float(base_fraction),
            model_id=self.model_id,
            model_revision=self.revision,
        )
        bundle.validate()
        return bundle

    def _continuation_scores(
        self,
        records: Sequence[PromptRecord],
        stems: Sequence[str],
        bundle: DirectionBundle | None,
        action: np.ndarray | None,
    ) -> np.ndarray:
        examples: list[tuple[int, list[int], int, list[int]]] = []
        for prompt_index, record in enumerate(records):
            prefix = self._prefix_ids(record.text)
            for stem in stems:
                continuation = list(self._stem_token_ids[stem])
                examples.append((prompt_index, prefix + continuation, len(prefix), continuation))
        scores = np.empty((len(records), len(stems)), dtype=float)
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start : start + self.batch_size]
            sequences = [item[1] for item in batch]
            input_ids, attention_mask = self._padded_batch(sequences)
            continuation_lengths = [len(item[3]) for item in batch]
            self._positions = self._last_prompt_positions(
                input_ids.shape[1],
                continuation_lengths,
            )
            context = (
                self._edit_hooks(bundle, np.asarray(action, dtype=float))
                if bundle is not None and action is not None
                else _null_context()
            )
            with self.torch.inference_mode(), context:
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    logits_to_keep=self._logits_to_keep,
                ).logits
            if logits.ndim != 3:
                raise ValueError("model logits must have shape batch x sequence x vocabulary")
            kept_rows = int(logits.shape[1])
            for local_index, (prompt_index, _sequence, _prefix_length, continuation) in enumerate(batch):
                first_row = kept_rows - len(continuation) - 1
                if first_row < 0:
                    raise ValueError(
                        "model returned too few logit rows to score the continuation"
                    )
                prediction_rows = self.torch.arange(
                    first_row,
                    first_row + len(continuation),
                    device=logits.device,
                )
                selected_logits = logits[local_index, prediction_rows, :]
                selected_log_probs = self.torch.log_softmax(
                    selected_logits.float(),
                    dim=-1,
                )
                token_ids = self.torch.as_tensor(
                    continuation,
                    dtype=self.torch.long,
                    device=selected_log_probs.device,
                )
                token_rows = self.torch.arange(
                    len(continuation),
                    device=selected_log_probs.device,
                )
                value = selected_log_probs[token_rows, token_ids].mean()
                stem_index = (start + local_index) % len(stems)
                scores[prompt_index, stem_index] = float(value.cpu())
        return scores

    def refusal_margin(
        self,
        records: Sequence[PromptRecord],
        bundle: DirectionBundle | None = None,
        action: np.ndarray | None = None,
    ) -> np.ndarray:
        refusal = self._continuation_scores(records, self.refusal_stems, bundle, action)
        compliance = self._continuation_scores(records, self.compliance_stems, bundle, action)
        return refusal.mean(axis=1) - compliance.mean(axis=1)


@contextmanager
def _null_context() -> Iterator[None]:
    yield
