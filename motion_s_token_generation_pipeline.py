"""Motion-S text-to-sign motion token generation pipeline.

This module is the source-of-truth implementation for the competition notebook.
It keeps the baseline non-autoregressive path, the length estimator, the
contrastive alignment loss, and a stretch-goal autoregressive path.

The notebook can import this file later so the same code is shared between the
Python source and the ipynb version.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (  # noqa: E402
    AutoTokenizer,
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    get_cosine_schedule_with_warmup,
)

try:
    from scipy import linalg as scipy_linalg  # type: ignore
except Exception:
    scipy_linalg = None


@dataclass
class MotionSConfig:
    """Central configuration for the competition pipeline."""

    data_root: Optional[str] = None
    train_csv: Optional[str] = None
    test_csv: Optional[str] = None
    sample_submission_csv: Optional[str] = None
    motion_features_dir: Optional[str] = None
    output_dir: Optional[str] = None
    text_backbone: str = "google/flan-t5-base"
    alt_text_backbone: str = "openai/clip-vit-base-patch32"
    use_clip_backbone: bool = False
    text_max_length: int = 128
    projection_dim: int = 256
    hidden_dim: int = 256
    max_seq_len: int = 800
    vocab_size: int = 512
    num_rvq_layers: int = 6
    batch_size: int = 32
    num_workers: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    epochs: int = 30
    patience: int = 5
    amp: bool = True
    freeze_text_encoder: bool = True
    use_cached_text_embeddings: bool = True
    contrastive_weight: float = 0.1
    temperature: float = 0.07
    length_bins: int = 128
    length_checkpoint: Optional[str] = None
    seed: int = 42
    model_type: str = "option_a"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        """Resolve local workspace paths when the module is run outside Kaggle."""

        project_root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
        resolved_root = find_motion_s_dataset_root(self.data_root)

        if self.train_csv is None:
            self.train_csv = str(resolved_root / "train.csv") if resolved_root is not None else "/kaggle/input/motion-s/train.csv"
        if self.test_csv is None:
            self.test_csv = str(resolved_root / "test.csv") if resolved_root is not None else "/kaggle/input/motion-s/test.csv"
        if self.sample_submission_csv is None and resolved_root is not None:
            self.sample_submission_csv = str(resolved_root / "sample_submission.csv")
        if self.motion_features_dir is None and resolved_root is not None:
            candidate_motion_dir = resolved_root / "Motion-Features"
            if candidate_motion_dir.exists():
                self.motion_features_dir = str(candidate_motion_dir)

        if self.output_dir is None:
            kaggle_output_root = Path("/kaggle/working")
            if kaggle_output_root.exists():
                self.output_dir = str(kaggle_output_root / "motion_s_outputs")
            else:
                self.output_dir = str(project_root / "motion_s_outputs")


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _iter_dataset_roots(base_path: Path) -> Iterable[Path]:
    """Yield plausible Motion-S dataset roots under a base directory."""

    if (base_path / "train.csv").exists() and (base_path / "test.csv").exists():
        yield base_path

    if not base_path.exists():
        return

    for child in base_path.iterdir():
        if child.is_dir() and (child / "train.csv").exists() and (child / "test.csv").exists():
            yield child


def find_motion_s_dataset_root(preferred_root: Optional[str | Path] = None) -> Optional[Path]:
    """Locate the competition dataset in the workspace or Kaggle input tree."""

    search_bases: List[Path] = []
    if preferred_root is not None:
        search_bases.append(Path(preferred_root))

    env_root = os.environ.get("MOTION_S_DATA_ROOT")
    if env_root:
        search_bases.append(Path(env_root))

    module_root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    search_bases.extend([module_root, Path.cwd(), Path("/kaggle/input")])

    seen: set[str] = set()
    for base_path in search_bases:
        for candidate in _iter_dataset_roots(base_path):
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                return candidate.resolve()
    return None


def build_text_input(sentence: str, gloss: str) -> str:
    """Join sentence and gloss into the single prompt used by the text encoder."""

    sentence = "" if pd.isna(sentence) else str(sentence)
    gloss = "" if pd.isna(gloss) else str(gloss)
    return f"{sentence.strip()} [GLOSS] {gloss.strip()}".strip()


def parse_token_string(token_string: Any) -> List[int]:
    """Parse a space-separated token string into a list of integers."""

    if token_string is None:
        return []
    if isinstance(token_string, float) and np.isnan(token_string):
        return []
    if isinstance(token_string, list):
        return [int(token) for token in token_string]
    text = str(token_string).strip()
    if not text:
        return []
    return [int(token) for token in text.split()]


def clip_length(length: int, min_len: int = 40, max_len: int = 800) -> int:
    """Clip a motion length to the competition constraints."""

    return int(max(min_len, min(max_len, int(length))))


def load_motion_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Load a competition CSV and normalize missing sentence or gloss values."""

    df = pd.read_csv(csv_path)
    for column in ["sentence", "gloss"]:
        if column in df.columns:
            df[column] = df[column].fillna("")
    if "text_input" not in df.columns and {"sentence", "gloss"}.issubset(df.columns):
        df["text_input"] = [build_text_input(sentence, gloss) for sentence, gloss in zip(df["sentence"], df["gloss"])]
    if "length" not in df.columns:
        for token_column in ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]:
            if token_column in df.columns:
                df["length"] = df[token_column].apply(lambda token_string: len(parse_token_string(token_string)))
                break
    return df


def group_aware_train_val_split(
    df: pd.DataFrame,
    group_col: str = "signer_id",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows by signer so validation does not leak signer identity."""

    if group_col not in df.columns:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        split_idx = max(1, int(len(shuffled) * (1.0 - val_fraction)))
        return shuffled.iloc[:split_idx].reset_index(drop=True), shuffled.iloc[split_idx:].reset_index(drop=True)

    groups = pd.Index(df[group_col].fillna("__missing__").unique()).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    num_val_groups = max(1, int(round(len(groups) * val_fraction)))
    val_groups = set(groups[:num_val_groups])
    val_df = df[df[group_col].fillna("__missing__").isin(val_groups)].reset_index(drop=True)
    train_df = df[~df[group_col].fillna("__missing__").isin(val_groups)].reset_index(drop=True)
    return train_df, val_df


def build_length_bin_centers(num_bins: int = 128, min_length: int = 40, max_length: int = 800) -> np.ndarray:
    """Create bin centers that map classifier outputs back to integer lengths."""

    centers = np.linspace(min_length, max_length, num_bins)
    return np.rint(centers).astype(np.int64)


def mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings while ignoring padding positions."""

    mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class HashingTokenizer:
    """Deterministic whitespace tokenizer used when pretrained Hugging Face models are unavailable."""

    def __init__(self, vocab_size: int = 8192, max_length: int = 128) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return 1 + (int.from_bytes(digest[:4], "little") % max(2, self.vocab_size - 1))

    def __call__(
        self,
        texts: Sequence[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of texts into padded integer ids and attention masks."""

        del padding, truncation, return_tensors
        effective_max_length = int(max_length or self.max_length)
        encoded: List[List[int]] = []
        for text in texts:
            tokens = self.pattern.findall(str(text).lower())
            token_ids = [self._token_to_id(token) for token in tokens][:effective_max_length]
            encoded.append(token_ids)

        seq_len = max(1, min(effective_max_length, max((len(item) for item in encoded), default=1)))
        input_ids = torch.zeros((len(encoded), seq_len), dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), seq_len), dtype=torch.long)

        for row_idx, token_ids in enumerate(encoded):
            current_ids = token_ids[:seq_len]
            if current_ids:
                length = len(current_ids)
                input_ids[row_idx, :length] = torch.tensor(current_ids, dtype=torch.long)
                attention_mask[row_idx, :length] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TextEncoderAdapter(nn.Module):
    """Wrap a Hugging Face text encoder and project pooled outputs into a compact space.

    If the requested backbone is not cached locally, fall back to a deterministic
    hashing tokenizer plus a small trainable embedding encoder so the notebook can
    still run fully offline.
    """

    def __init__(
        self,
        backbone_name: str,
        projection_dim: int = 256,
        max_length: int = 128,
        freeze_backbone: bool = True,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.max_length = max_length
        self.projection_dim = projection_dim
        self.is_clip = "clip" in backbone_name.lower()
        self.using_fallback = False

        try:
            if self.is_clip:
                self.tokenizer = CLIPTokenizer.from_pretrained(backbone_name, local_files_only=local_files_only)
                self.encoder = CLIPTextModel.from_pretrained(backbone_name, local_files_only=local_files_only)
                hidden_dim = self.encoder.config.hidden_size
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(backbone_name, local_files_only=local_files_only)
                self.encoder = T5EncoderModel.from_pretrained(backbone_name, local_files_only=local_files_only)
                hidden_dim = self.encoder.config.d_model
        except Exception as exc:
            warnings.warn(
                f"Falling back to a lightweight offline text encoder for {backbone_name!r}: {exc}",
                RuntimeWarning,
            )
            self.using_fallback = True
            self.tokenizer = HashingTokenizer(vocab_size=8192, max_length=max_length)
            hidden_dim = projection_dim
            self.encoder = nn.Embedding(self.tokenizer.vocab_size, hidden_dim, padding_idx=0)

        if freeze_backbone and not self.using_fallback:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a batch of text prompts for the selected backbone."""

        return self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def encode_texts(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Encode texts and return pooled, projected, and optional sequence outputs."""

        device = next(self.parameters()).device
        tokenized = self.tokenize(texts)
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        if self.using_fallback:
            sequence_output = self.encoder(tokenized["input_ids"])
        else:
            outputs = self.encoder(**tokenized)
            sequence_output = outputs.last_hidden_state
        pooled = mean_pool_hidden_states(sequence_output, tokenized["attention_mask"])
        projected = self.adapter(pooled)
        return {
            "pooled": pooled,
            "projected": projected,
            "sequence_output": sequence_output,
            "attention_mask": tokenized["attention_mask"],
        }

    def forward(self, texts: Sequence[str], return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        """Encode text prompts and optionally return token-level hidden states."""

        result = self.encode_texts(texts)
        if return_sequence:
            return result
        return {"pooled": result["pooled"], "projected": result["projected"]}


@torch.no_grad()
def cache_text_embeddings(
    encoder: TextEncoderAdapter,
    texts: Sequence[str],
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Precompute text embeddings so training can stay within the Kaggle budget."""

    encoder = encoder.to(device)
    encoder.eval()
    embeddings: List[torch.Tensor] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Caching text embeddings"):
        batch_texts = texts[start : start + batch_size]
        outputs = encoder.encode_texts(batch_texts)
        embeddings.append(outputs["projected"].detach().cpu())
    return torch.cat(embeddings, dim=0)


class LengthEstimatorMLP(nn.Module):
    """Predict a discrete length bin from a text embedding."""

    def __init__(self, input_dim: int, num_bins: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return logits over length bins."""

        return self.net(embeddings)


class GlossLengthRegressor:
    """Fallback linear regressor using only gloss token count as a feature."""

    def __init__(self) -> None:
        self.coef_: Optional[float] = None
        self.intercept_: float = 0.0

    def fit(self, gloss_token_counts: Sequence[float], lengths: Sequence[float]) -> "GlossLengthRegressor":
        """Fit a least-squares line from gloss length to target motion length."""

        x = np.asarray(gloss_token_counts, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(lengths, dtype=np.float64).reshape(-1, 1)
        design = np.concatenate([x, np.ones_like(x)], axis=1)
        solution, *_ = np.linalg.lstsq(design, y, rcond=None)
        self.coef_ = float(np.asarray(solution[0]).reshape(-1)[0])
        self.intercept_ = float(np.asarray(solution[1]).reshape(-1)[0])
        return self

    def predict(self, gloss_token_counts: Sequence[float]) -> np.ndarray:
        """Predict lengths from gloss token counts and clip to the competition range."""

        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict().")
        x = np.asarray(gloss_token_counts, dtype=np.float64)
        y = x * float(self.coef_) + self.intercept_
        return np.clip(np.rint(y), 40, 800).astype(np.int64)


@torch.no_grad()
def load_length_estimator(
    checkpoint_path: str | Path,
    input_dim: int,
    num_bins: int = 128,
    device: torch.device | str = "cpu",
) -> Tuple[Optional[LengthEstimatorMLP], Optional[np.ndarray]]:
    """Load the provided checkpoint if possible, otherwise return a fallback pair."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None, None

    model = LengthEstimatorMLP(input_dim=input_dim, num_bins=num_bins).to(device)
    payload = torch.load(checkpoint_path, map_location=device)
    state_dict = payload.get("model_state_dict", payload.get("state_dict", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, build_length_bin_centers(num_bins=num_bins)


def fit_gloss_length_regressor(df: pd.DataFrame) -> GlossLengthRegressor:
    """Train the fallback length regressor from gloss token count to sequence length."""

    gloss_lengths = [len(str(gloss).split()) for gloss in df["gloss"]]
    if "length" in df.columns:
        lengths = [int(length) for length in df["length"]]
    else:
        token_column = next((column for column in ["base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"] if column in df.columns), None)
        if token_column is None:
            raise KeyError("Expected either a length column or at least one RVQ token column.")
        lengths = [len(parse_token_string(token_string)) for token_string in df[token_column]]
    return GlossLengthRegressor().fit(gloss_lengths, lengths)


def predict_lengths_for_dataframe(
    df: pd.DataFrame,
    text_embeddings: torch.Tensor,
    length_model: Optional[LengthEstimatorMLP],
    length_bin_centers: Optional[np.ndarray],
    fallback_regressor: Optional[GlossLengthRegressor],
    device: torch.device,
) -> np.ndarray:
    """Predict integer lengths from either the checkpointed model or the fallback regressor."""

    if length_model is not None and length_bin_centers is not None:
        length_model = length_model.to(device)
        length_model.eval()
        logits = length_model(text_embeddings.to(device))
        pred_bins = logits.argmax(dim=-1).detach().cpu().numpy()
        lengths = length_bin_centers[pred_bins]
        return np.asarray([clip_length(length) for length in lengths], dtype=np.int64)

    if fallback_regressor is None:
        raise ValueError("fallback_regressor is required when the checkpointed length model is unavailable")
    gloss_counts = [len(str(gloss).split()) for gloss in df["gloss"]]
    return fallback_regressor.predict(gloss_counts)


def align_token_layers(token_layers: Sequence[Sequence[int]]) -> List[List[int]]:
    """Trim or pad all six token layers to a shared length."""

    lengths = [len(layer) for layer in token_layers]
    if not lengths:
        return [[] for _ in range(6)]
    target_len = min(lengths)
    return [list(layer[:target_len]) for layer in token_layers]


class MotionTokenDataset(Dataset):
    """Return text features, six-layer token targets, and the true sequence length."""

    def __init__(
        self,
        df: pd.DataFrame,
        token_columns: Sequence[str] = ("base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"),
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.token_columns = list(token_columns)
        self.text_embeddings = text_embeddings
        self.has_targets = all(column in self.df.columns for column in self.token_columns)

        if "text_input" not in self.df.columns and {"sentence", "gloss"}.issubset(self.df.columns):
            self.df["text_input"] = [build_text_input(sentence, gloss) for sentence, gloss in zip(self.df["sentence"], self.df["gloss"])]

        if self.has_targets:
            for column in self.token_columns:
                self.df[column] = self.df[column].apply(parse_token_string)

        if "length" not in self.df.columns:
            if self.has_targets:
                self.df["length"] = self.df[self.token_columns[0]].apply(len)
            else:
                self.df["length"] = 0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        sample: Dict[str, Any] = {
            "id": row.get("id", index),
            "sentence": row.get("sentence", ""),
            "gloss": row.get("gloss", ""),
            "text_input": row.get("text_input", build_text_input(row.get("sentence", ""), row.get("gloss", ""))),
            "length": clip_length(row.get("length", 0)),
        }

        if self.text_embeddings is not None:
            sample["text_embedding"] = self.text_embeddings[index].float()

        if self.has_targets:
            token_layers = [row[column] for column in self.token_columns]
            token_layers = align_token_layers(token_layers)
            sample["token_layers"] = torch.tensor(token_layers, dtype=torch.long)
        else:
            sample["token_layers"] = None

        return sample


def collate_motion_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a batch to the longest sequence length while keeping six-layer alignment."""

    if batch[0]["token_layers"] is None:
        raise ValueError("collate_motion_batch requires token targets")

    max_len = max(item["token_layers"].shape[-1] for item in batch)
    padded_tokens = []
    masks = []
    lengths = []
    text_embeddings = []
    text_inputs = []
    sample_ids = []

    for item in batch:
        tokens = item["token_layers"]
        seq_len = tokens.shape[-1]
        pad_width = max_len - seq_len
        if pad_width > 0:
            pad_tensor = torch.full((tokens.shape[0], pad_width), -100, dtype=torch.long)
            tokens = torch.cat([tokens, pad_tensor], dim=-1)
        padded_tokens.append(tokens)
        masks.append(torch.arange(max_len) < seq_len)
        lengths.append(int(item["length"]))
        sample_ids.append(item["id"])
        text_inputs.append(item["text_input"])
        if item.get("text_embedding") is not None:
            text_embeddings.append(item["text_embedding"])

    batch_dict: Dict[str, Any] = {
        "ids": sample_ids,
        "text_inputs": text_inputs,
        "tokens": torch.stack(padded_tokens, dim=0),
        "mask": torch.stack(masks, dim=0),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }
    if text_embeddings:
        batch_dict["text_embeddings"] = torch.stack(text_embeddings, dim=0)
    return batch_dict


@torch.no_grad()
def precompute_text_embeddings_for_df(
    df: pd.DataFrame,
    encoder: TextEncoderAdapter,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Cache embeddings for an entire dataframe before training the token generator."""

    texts = [build_text_input(sentence, gloss) for sentence, gloss in zip(df["sentence"], df["gloss"])]
    return cache_text_embeddings(encoder, texts, batch_size=batch_size, device=device)


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: MotionSConfig,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> Tuple[MotionTokenDataset, MotionTokenDataset]:
    """Prepare train and validation datasets with optional cached text embeddings."""

    train_text_embeddings = None
    val_text_embeddings = None
    if cfg.use_cached_text_embeddings:
        if text_encoder is None:
            raise ValueError("text_encoder is required when use_cached_text_embeddings=True")
        train_text_embeddings = precompute_text_embeddings_for_df(train_df, text_encoder, batch_size=64, device=cfg.device)
        val_text_embeddings = precompute_text_embeddings_for_df(val_df, text_encoder, batch_size=64, device=cfg.device)

    return (
        MotionTokenDataset(train_df, text_embeddings=train_text_embeddings),
        MotionTokenDataset(val_df, text_embeddings=val_text_embeddings),
    )


def build_dataloaders(cfg: MotionSConfig, train_dataset: Dataset, val_dataset: Dataset) -> Tuple[DataLoader, DataLoader]:
    """Construct train and validation dataloaders from the prepared datasets."""

    pin_memory = cfg.device.startswith("cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_motion_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_motion_batch,
    )
    return train_loader, val_loader


def masked_sequence_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average a sequence over valid positions only."""

    mask = mask.unsqueeze(-1).type_as(sequence)
    summed = (sequence * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


class OptionATokenGenerator(nn.Module):
    """Non-autoregressive RVQ token generator conditioned on text and sequence length."""

    def __init__(
        self,
        text_dim: int = 256,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        vocab_size: int = 512,
        max_seq_len: int = 800,
        num_rvq_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_dim = text_dim
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_rvq_layers = num_rvq_layers

        self.text_adapter = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.length_embedding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_heads = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(num_rvq_layers)])
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        text_embedding: torch.Tensor,
        lengths: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate six token distributions for every valid timestep."""

        batch_size = text_embedding.size(0)
        if sequence_mask is not None:
            seq_len = int(sequence_mask.size(1))
        else:
            seq_len = int(lengths.max().item())
        seq_len = min(seq_len, self.max_seq_len)

        positions = torch.arange(seq_len, device=text_embedding.device)
        positional_queries = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        text_context = self.text_adapter(text_embedding)
        length_context = self.length_embedding(lengths.clamp(min=0, max=self.max_seq_len))
        conditioning = (text_context + length_context).unsqueeze(1)
        decoder_input = self.input_dropout(positional_queries + conditioning.expand(batch_size, seq_len, -1))
        hidden = self.decoder(tgt=decoder_input, memory=conditioning)
        logits = torch.stack([head(hidden) for head in self.output_heads], dim=1)

        if sequence_mask is None:
            sequence_mask = torch.arange(seq_len, device=text_embedding.device).unsqueeze(0) < lengths.unsqueeze(1)
        motion_embedding = self.motion_projection(masked_sequence_mean(hidden, sequence_mask))
        return {"logits": logits, "hidden_states": hidden, "motion_embedding": motion_embedding}

    @torch.no_grad()
    def generate(
        self,
        text_embedding: torch.Tensor,
        lengths: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return argmax token ids with shape [batch, 6, seq_len]."""

        outputs = self.forward(text_embedding=text_embedding, lengths=lengths, sequence_mask=sequence_mask)
        logits = outputs["logits"] / max(temperature, 1e-6)
        return logits.argmax(dim=-1)


def resolve_batch_text_embeddings(
    batch: Dict[str, Any],
    text_encoder: Optional[TextEncoderAdapter],
    device: torch.device,
) -> torch.Tensor:
    """Resolve batch text embeddings from cache or from the text encoder on the fly."""

    cached = batch.get("text_embeddings")
    if cached is not None:
        return cached.to(device)
    if text_encoder is None:
        raise ValueError("text_encoder is required when cached text embeddings are unavailable")

    track_grad = any(parameter.requires_grad for parameter in text_encoder.parameters()) and torch.is_grad_enabled()
    if track_grad:
        projected = text_encoder.encode_texts(batch["text_inputs"])["projected"]
    else:
        with torch.no_grad():
            projected = text_encoder.encode_texts(batch["text_inputs"])["projected"]
    return projected.to(device)


def masked_rvq_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, List[float]]:
    """Compute masked cross-entropy over all six RVQ heads and per-layer accuracies."""

    num_layers = logits.size(1)
    vocab_size = logits.size(-1)
    total_loss = logits.new_tensor(0.0)
    accuracies: List[float] = []
    flat_mask = mask.reshape(-1)

    for layer_idx in range(num_layers):
        layer_logits = logits[:, layer_idx].reshape(-1, vocab_size)
        layer_targets = targets[:, layer_idx].reshape(-1)
        valid = flat_mask & (layer_targets != ignore_index)
        if valid.any():
            total_loss = total_loss + F.cross_entropy(layer_logits[valid], layer_targets[valid])
            predictions = layer_logits[valid].argmax(dim=-1)
            accuracies.append((predictions == layer_targets[valid]).float().mean().item())
        else:
            accuracies.append(0.0)

    return total_loss / max(num_layers, 1), accuracies


def info_nce_loss(text_embeddings: torch.Tensor, motion_embeddings: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric CLIP-style contrastive loss that directly optimizes text-motion alignment."""

    text_embeddings = F.normalize(text_embeddings, dim=-1)
    motion_embeddings = F.normalize(motion_embeddings, dim=-1)
    logits = text_embeddings @ motion_embeddings.t() / max(temperature, 1e-6)
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_t2m = F.cross_entropy(logits, labels)
    loss_m2t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_t2m + loss_m2t)


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: MotionSConfig,
    num_training_steps: int,
    extra_parameters: Optional[Iterable[nn.Parameter]] = None,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Create AdamW plus a cosine schedule with warmup."""

    parameters = list(model.parameters())
    if extra_parameters is not None:
        parameters.extend([parameter for parameter in extra_parameters if parameter.requires_grad])
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, num_training_steps),
    )
    return optimizer, scheduler


def train_one_epoch(
    model: OptionATokenGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    cfg: MotionSConfig,
    device: torch.device,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> Dict[str, float]:
    """Run one mixed-precision training epoch and log loss plus layer accuracy."""

    model.train()
    if text_encoder is not None:
        text_encoder.train(any(parameter.requires_grad for parameter in text_encoder.parameters()))

    running_loss = 0.0
    running_contrastive = 0.0
    running_acc = np.zeros(cfg.num_rvq_layers, dtype=np.float64)
    num_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        text_embeddings = resolve_batch_text_embeddings(batch, text_encoder, device)
        lengths = batch["lengths"].to(device)
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)

        with autocast(enabled=cfg.amp and device.type == "cuda"):
            outputs = model(text_embeddings, lengths=lengths, sequence_mask=mask)
            token_loss, accuracies = masked_rvq_cross_entropy(outputs["logits"], tokens, mask)
            contrastive = info_nce_loss(text_embeddings, outputs["motion_embedding"], temperature=cfg.temperature)
            loss = token_loss + cfg.contrastive_weight * contrastive

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += float(token_loss.item())
        running_contrastive += float(contrastive.item())
        running_acc += np.asarray(accuracies)
        num_batches += 1

    denom = max(1, num_batches)
    metrics = {"train_loss": running_loss / denom, "train_contrastive": running_contrastive / denom}
    for idx, value in enumerate(running_acc / denom, start=1):
        metrics[f"train_acc_layer_{idx}"] = float(value)
    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: OptionATokenGenerator,
    loader: DataLoader,
    cfg: MotionSConfig,
    device: torch.device,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> Dict[str, float]:
    """Evaluate token loss, contrastive loss, and per-layer accuracy on validation data."""

    model.eval()
    if text_encoder is not None:
        text_encoder.eval()

    running_loss = 0.0
    running_contrastive = 0.0
    running_acc = np.zeros(cfg.num_rvq_layers, dtype=np.float64)
    num_batches = 0

    for batch in tqdm(loader, desc="val", leave=False):
        text_embeddings = resolve_batch_text_embeddings(batch, text_encoder, device)
        lengths = batch["lengths"].to(device)
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)

        outputs = model(text_embeddings, lengths=lengths, sequence_mask=mask)
        token_loss, accuracies = masked_rvq_cross_entropy(outputs["logits"], tokens, mask)
        contrastive = info_nce_loss(text_embeddings, outputs["motion_embedding"], temperature=cfg.temperature)

        running_loss += float(token_loss.item())
        running_contrastive += float(contrastive.item())
        running_acc += np.asarray(accuracies)
        num_batches += 1

    denom = max(1, num_batches)
    metrics = {"val_loss": running_loss / denom, "val_contrastive": running_contrastive / denom}
    for idx, value in enumerate(running_acc / denom, start=1):
        metrics[f"val_acc_layer_{idx}"] = float(value)
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_val_loss: float,
    cfg: MotionSConfig,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> None:
    """Save the best model state so inference can reload the pipeline quickly."""

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
        "config": asdict(cfg),
    }
    if text_encoder is not None:
        payload["text_encoder_state_dict"] = text_encoder.state_dict()
    torch.save(payload, path)


def build_default_components(
    cfg: MotionSConfig,
    device: Optional[torch.device | str] = None,
) -> Tuple[TextEncoderAdapter, OptionATokenGenerator, Optional[LengthEstimatorMLP], Optional[np.ndarray]]:
    """Build the default text encoder, generator, and length estimator bundle."""

    device = torch.device(device or cfg.device)
    text_backbone = cfg.alt_text_backbone if cfg.use_clip_backbone else cfg.text_backbone
    text_encoder = TextEncoderAdapter(
        backbone_name=text_backbone,
        projection_dim=cfg.projection_dim,
        max_length=cfg.text_max_length,
        freeze_backbone=cfg.freeze_text_encoder,
        local_files_only=True,
    ).to(device)
    generator = OptionATokenGenerator(
        text_dim=cfg.projection_dim,
        d_model=cfg.hidden_dim,
        num_heads=8,
        num_layers=4,
        vocab_size=cfg.vocab_size,
        max_seq_len=cfg.max_seq_len,
        num_rvq_layers=cfg.num_rvq_layers,
    ).to(device)
    length_model, length_bin_centers = (None, None)
    if cfg.length_checkpoint is not None:
        length_model, length_bin_centers = load_length_estimator(
            cfg.length_checkpoint,
            input_dim=cfg.projection_dim,
            num_bins=cfg.length_bins,
            device=device,
        )
    return text_encoder, generator, length_model, length_bin_centers


def fit_model(
    model: OptionATokenGenerator,
    cfg: MotionSConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str | Path,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> pd.DataFrame:
    """Train with early stopping and return a metrics table for analysis."""

    num_training_steps = cfg.epochs * max(1, len(train_loader))
    extra_parameters: Optional[Iterable[nn.Parameter]] = None
    if text_encoder is not None:
        extra_parameters = [parameter for parameter in text_encoder.parameters() if parameter.requires_grad]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, num_training_steps, extra_parameters=extra_parameters)
    scaler = GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))
    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = cfg.patience
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, torch.device(cfg.device), text_encoder=text_encoder)
        val_metrics = evaluate_one_epoch(model, val_loader, cfg, torch.device(cfg.device), text_encoder=text_encoder)
        epoch_metrics = {"epoch": float(epoch + 1), **train_metrics, **val_metrics}
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        current_val = val_metrics["val_loss"] + cfg.contrastive_weight * val_metrics["val_contrastive"]
        if current_val < best_val_loss:
            best_val_loss = current_val
            best_epoch = epoch + 1
            patience_left = cfg.patience
            save_checkpoint(save_path, model, optimizer, scheduler, epoch + 1, best_val_loss, cfg, text_encoder=text_encoder)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}.")
                break

    return pd.DataFrame(history)


def load_model_bundle(
    checkpoint_path: str | Path,
    cfg: MotionSConfig,
    device: torch.device | str,
) -> Tuple[OptionATokenGenerator, Optional[TextEncoderAdapter], Dict[str, Any]]:
    """Reload the generator and, if present, the text encoder from a checkpoint."""

    device = torch.device(device)
    text_encoder, generator, _, _ = build_default_components(cfg, device=device)
    payload = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(payload["model_state_dict"], strict=True)
    if "text_encoder_state_dict" in payload:
        text_encoder.load_state_dict(payload["text_encoder_state_dict"], strict=False)
    return generator, text_encoder, payload


def load_trained_option_a_model(
    checkpoint_path: str | Path,
    cfg: MotionSConfig,
    device: torch.device | str,
) -> OptionATokenGenerator:
    """Reload just the Option A generator for inference."""

    model, _, _ = load_model_bundle(checkpoint_path, cfg, device)
    model.eval()
    return model


def tokens_to_string(tokens: Sequence[int]) -> str:
    """Convert an integer token sequence into the submission string format."""

    clipped = [int(max(0, min(511, int(token)))) for token in tokens]
    return " ".join(str(token) for token in clipped)


@torch.no_grad()
def generate_submission_frame(
    test_df: pd.DataFrame,
    model: OptionATokenGenerator,
    cfg: MotionSConfig,
    text_embeddings: Optional[torch.Tensor] = None,
    text_encoder: Optional[TextEncoderAdapter] = None,
    lengths: Optional[np.ndarray] = None,
    length_model: Optional[LengthEstimatorMLP] = None,
    length_bin_centers: Optional[np.ndarray] = None,
    fallback_regressor: Optional[GlossLengthRegressor] = None,
    device: torch.device | str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Generate the competition submission with synchronized six-layer token lengths."""

    device = torch.device(device)
    model = model.to(device).eval()
    if text_encoder is not None:
        text_encoder = text_encoder.to(device).eval()

    if lengths is None:
        if length_model is None and fallback_regressor is None:
            raise ValueError("Either lengths or a length predictor must be provided")
        if text_embeddings is None and text_encoder is None:
            raise ValueError("Either text_embeddings or text_encoder must be provided")
        if text_embeddings is None:
            texts = [build_text_input(sentence, gloss) for sentence, gloss in zip(test_df["sentence"], test_df["gloss"])]
            text_embeddings = cache_text_embeddings(text_encoder, texts, batch_size=batch_size, device=device)
        lengths = predict_lengths_for_dataframe(test_df, text_embeddings, length_model, length_bin_centers, fallback_regressor, device)

    rows: List[Dict[str, Any]] = []
    for start in tqdm(range(0, len(test_df), batch_size), desc="generate"):
        end = min(len(test_df), start + batch_size)
        if text_embeddings is not None:
            batch_embeddings = text_embeddings[start:end].to(device)
        else:
            batch_texts = [build_text_input(sentence, gloss) for sentence, gloss in zip(test_df.iloc[start:end]["sentence"], test_df.iloc[start:end]["gloss"])]
            batch_embeddings = text_encoder.encode_texts(batch_texts)["projected"]
        batch_lengths = torch.tensor(lengths[start:end], dtype=torch.long, device=device)
        seq_len = int(batch_lengths.max().item())
        batch_mask = torch.arange(seq_len, device=device).unsqueeze(0) < batch_lengths.unsqueeze(1)
        token_ids = model.generate(batch_embeddings, batch_lengths, sequence_mask=batch_mask).cpu().numpy()

        for row_idx, (_, row) in enumerate(test_df.iloc[start:end].iterrows()):
            sample_len = int(batch_lengths[row_idx].item())
            sample_tokens = token_ids[row_idx, :, :sample_len]
            rows.append(
                {
                    "id": row["id"],
                    "base_tokens": tokens_to_string(sample_tokens[0]),
                    "residual_1": tokens_to_string(sample_tokens[1]),
                    "residual_2": tokens_to_string(sample_tokens[2]),
                    "residual_3": tokens_to_string(sample_tokens[3]),
                    "residual_4": tokens_to_string(sample_tokens[4]),
                    "residual_5": tokens_to_string(sample_tokens[5]),
                }
            )
    return pd.DataFrame(rows)


def validate_submission_frame(submission_df: pd.DataFrame) -> None:
    """Check for NaNs, range violations, and length mismatches before saving the CSV."""

    required_columns = ["id", "base_tokens", "residual_1", "residual_2", "residual_3", "residual_4", "residual_5"]
    missing = [column for column in required_columns if column not in submission_df.columns]
    if missing:
        raise ValueError(f"Missing submission columns: {missing}")

    if submission_df[required_columns].isna().any().any():
        raise ValueError("Submission contains NaNs.")

    for _, row in submission_df.iterrows():
        layer_lists = [parse_token_string(row[column]) for column in required_columns[1:]]
        lengths = [len(layer) for layer in layer_lists]
        if len(set(lengths)) != 1:
            raise ValueError(f"Mismatched RVQ layer lengths for id={row['id']}: {lengths}")
        for layer in layer_lists:
            if not layer:
                raise ValueError(f"Empty token row for id={row['id']}")
            if min(layer) < 0 or max(layer) > 511:
                raise ValueError(f"Token range violation for id={row['id']}")


def token_histogram_features(token_layers: torch.Tensor, mask: torch.Tensor, vocab_size: int = 512) -> np.ndarray:
    """Convert a batch of RVQ token sequences into a simple fixed-size feature vector."""

    batch_size, num_layers, _ = token_layers.shape
    features: List[np.ndarray] = []
    mask_np = mask.detach().cpu().numpy().astype(bool)
    token_np = token_layers.detach().cpu().numpy()
    for batch_idx in range(batch_size):
        sample_features: List[np.ndarray] = []
        valid_length = max(1, int(mask_np[batch_idx].sum()))
        for layer_idx in range(num_layers):
            layer_tokens = token_np[batch_idx, layer_idx, :valid_length]
            hist = np.bincount(layer_tokens.astype(np.int64), minlength=vocab_size).astype(np.float64)
            hist = hist / max(hist.sum(), 1.0)
            sample_features.append(hist)
        sample_features.append(np.asarray([valid_length / 800.0], dtype=np.float64))
        features.append(np.concatenate(sample_features, axis=0))
    return np.stack(features, axis=0)


def _matrix_sqrt(mat: np.ndarray) -> np.ndarray:
    """Compute a matrix square root using SciPy when available, otherwise NumPy."""

    if scipy_linalg is not None:
        try:
            return scipy_linalg.sqrtm(mat)
        except Exception:
            pass

    eigvals, eigvecs = np.linalg.eigh((mat + mat.T) / 2.0)
    eigvals = np.clip(eigvals, 0.0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def frechet_distance(mu_1: np.ndarray, sigma_1: np.ndarray, mu_2: np.ndarray, sigma_2: np.ndarray, eps: float = 1e-6) -> float:
    """Compute the Fréchet distance between two Gaussian feature clouds."""

    sigma_1 = np.asarray(sigma_1, dtype=np.float64)
    sigma_2 = np.asarray(sigma_2, dtype=np.float64)
    eye_1 = np.eye(sigma_1.shape[0], dtype=np.float64)
    eye_2 = np.eye(sigma_2.shape[0], dtype=np.float64)
    covmean = _matrix_sqrt((sigma_1 + eps * eye_1) @ (sigma_2 + eps * eye_2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_1 - mu_2
    return float(diff @ diff + np.trace(sigma_1 + sigma_2 - 2.0 * covmean))


def approximate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """Approximate FID with simple Gaussian statistics over token histogram features."""

    real_features = np.atleast_2d(real_features)
    generated_features = np.atleast_2d(generated_features)
    mu_1 = real_features.mean(axis=0)
    mu_2 = generated_features.mean(axis=0)
    sigma_1 = np.cov(real_features, rowvar=False)
    sigma_2 = np.cov(generated_features, rowvar=False)
    return frechet_distance(mu_1, sigma_1, mu_2, sigma_2)


def r_precision_at_k(text_features: np.ndarray, motion_features: np.ndarray, k: int = 3, group_size: int = 32) -> float:
    """Compute a retrieval proxy that matches the text-to-motion retrieval objective."""

    text_features = text_features / np.clip(np.linalg.norm(text_features, axis=1, keepdims=True), 1e-8, None)
    motion_features = motion_features / np.clip(np.linalg.norm(motion_features, axis=1, keepdims=True), 1e-8, None)
    num_samples = text_features.shape[0]
    num_groups = max(1, num_samples // group_size)
    hits = []
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = min(num_samples, start + group_size)
        sims = text_features[start:end] @ motion_features[start:end].T
        targets = np.arange(end - start)
        topk = np.argsort(-sims, axis=1)[:, :k]
        hits.extend([target in topk[row_idx] for row_idx, target in enumerate(targets)])
    return float(np.mean(hits)) if hits else 0.0


def diversity_score(features: np.ndarray, sample_size: int = 1024) -> float:
    """Estimate diversity as the mean pairwise L2 distance between sampled features."""

    if len(features) < 2:
        return 0.0
    rng = np.random.default_rng(42)
    indices = rng.choice(len(features), size=min(sample_size, len(features)), replace=False)
    sample = features[indices]
    distances = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)
    tri = distances[np.triu_indices_from(distances, k=1)]
    return float(tri.mean()) if len(tri) else 0.0


def flatten_interleaved_tokens(token_layers: torch.Tensor) -> torch.Tensor:
    """Convert [batch, 6, length] tokens into one interleaved causal sequence."""

    batch_size, num_layers, seq_len = token_layers.shape
    if num_layers != 6:
        raise ValueError(f"Expected 6 RVQ layers, got {num_layers}")
    return token_layers.permute(0, 2, 1).reshape(batch_size, seq_len * num_layers)


def unflatten_interleaved_tokens(flat_tokens: torch.Tensor, num_layers: int = 6) -> torch.Tensor:
    """Recover [batch, 6, length] tokens from the interleaved autoregressive layout."""

    batch_size, flat_seq_len = flat_tokens.shape
    if flat_seq_len % num_layers != 0:
        raise ValueError("Flat sequence length must be divisible by the number of RVQ layers.")
    seq_len = flat_seq_len // num_layers
    return flat_tokens.reshape(batch_size, seq_len, num_layers).permute(0, 2, 1)


def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Construct a standard upper-triangular autoregressive attention mask."""

    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


def prepare_option_b_batch(token_layers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare teacher-forcing inputs and labels for the interleaved autoregressive model."""

    flat_tokens = flatten_interleaved_tokens(token_layers)
    return flat_tokens[:, :-1], flat_tokens[:, 1:]


class OptionBTokenGenerator(nn.Module):
    """Autoregressive baseline for higher-quality motion token generation."""

    def __init__(
        self,
        text_dim: int = 256,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        vocab_size: int = 512,
        max_flat_seq_len: int = 800 * 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_flat_seq_len = max_flat_seq_len

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_flat_seq_len, d_model)
        self.text_adapter = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_tokens: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass over the flattened token stream."""

        batch_size, seq_len = input_tokens.shape
        positions = torch.arange(seq_len, device=input_tokens.device)
        token_states = self.token_embedding(input_tokens) + self.position_embedding(positions).unsqueeze(0)
        text_memory = self.text_adapter(text_embedding).unsqueeze(1)
        causal_mask = build_causal_mask(seq_len, input_tokens.device)
        hidden = self.decoder(tgt=token_states, memory=text_memory, tgt_mask=causal_mask)
        return self.lm_head(hidden)

    @torch.no_grad()
    def generate(self, text_embedding: torch.Tensor, seq_len: int, start_token: int = 0) -> torch.Tensor:
        """Greedy autoregressive generation for the interleaved RVQ stream."""

        batch_size = text_embedding.size(0)
        generated = torch.full((batch_size, 1), start_token, device=text_embedding.device, dtype=torch.long)
        for _ in range(seq_len - 1):
            logits = self.forward(generated, text_embedding)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated


class DiffusionTokenGeneratorStub(nn.Module):
    """Placeholder for a diffusion model over token logits.

    This path is intentionally left as a stub because it is the least Kaggle-
    friendly option under a strict runtime budget.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Raise a clear error if the diffusion path is accidentally used."""

        raise NotImplementedError("Option C is intentionally omitted from the baseline module.")


def resolve_batch_text_embeddings(
    batch: Dict[str, Any],
    text_encoder: Optional[TextEncoderAdapter],
    device: torch.device,
) -> torch.Tensor:
    """Resolve batch text embeddings from cache or from the text encoder on the fly."""

    cached = batch.get("text_embeddings")
    if cached is not None:
        return cached.to(device)
    if text_encoder is None:
        raise ValueError("text_encoder is required when cached text embeddings are unavailable")

    track_grad = any(parameter.requires_grad for parameter in text_encoder.parameters()) and torch.is_grad_enabled()
    if track_grad:
        projected = text_encoder.encode_texts(batch["text_inputs"])["projected"]
    else:
        with torch.no_grad():
            projected = text_encoder.encode_texts(batch["text_inputs"])["projected"]
    return projected.to(device)


def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: MotionSConfig,
    num_training_steps: int,
    extra_parameters: Optional[Iterable[nn.Parameter]] = None,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Create AdamW plus a cosine schedule with warmup."""

    parameters = list(model.parameters())
    if extra_parameters is not None:
        parameters.extend([parameter for parameter in extra_parameters if parameter.requires_grad])
    optimizer = torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_steps = int(num_training_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, num_training_steps),
    )
    return optimizer, scheduler


def train_one_epoch(
    model: OptionATokenGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: GradScaler,
    cfg: MotionSConfig,
    device: torch.device,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> Dict[str, float]:
    """Run one mixed-precision training epoch and log loss plus layer accuracy."""

    model.train()
    if text_encoder is not None:
        text_encoder.train(any(parameter.requires_grad for parameter in text_encoder.parameters()))

    running_loss = 0.0
    running_contrastive = 0.0
    running_acc = np.zeros(cfg.num_rvq_layers, dtype=np.float64)
    num_batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        text_embeddings = resolve_batch_text_embeddings(batch, text_encoder, device)
        lengths = batch["lengths"].to(device)
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)

        with autocast(enabled=cfg.amp and device.type == "cuda"):
            outputs = model(text_embeddings, lengths=lengths, sequence_mask=mask)
            token_loss, accuracies = masked_rvq_cross_entropy(outputs["logits"], tokens, mask)
            contrastive = info_nce_loss(text_embeddings, outputs["motion_embedding"], temperature=cfg.temperature)
            loss = token_loss + cfg.contrastive_weight * contrastive

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_loss += float(token_loss.item())
        running_contrastive += float(contrastive.item())
        running_acc += np.asarray(accuracies)
        num_batches += 1

    denom = max(1, num_batches)
    metrics = {"train_loss": running_loss / denom, "train_contrastive": running_contrastive / denom}
    for idx, value in enumerate(running_acc / denom, start=1):
        metrics[f"train_acc_layer_{idx}"] = float(value)
    return metrics


@torch.no_grad()
def evaluate_one_epoch(
    model: OptionATokenGenerator,
    loader: DataLoader,
    cfg: MotionSConfig,
    device: torch.device,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> Dict[str, float]:
    """Evaluate token loss, contrastive loss, and per-layer accuracy on validation data."""

    model.eval()
    if text_encoder is not None:
        text_encoder.eval()

    running_loss = 0.0
    running_contrastive = 0.0
    running_acc = np.zeros(cfg.num_rvq_layers, dtype=np.float64)
    num_batches = 0

    for batch in tqdm(loader, desc="val", leave=False):
        text_embeddings = resolve_batch_text_embeddings(batch, text_encoder, device)
        lengths = batch["lengths"].to(device)
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)

        outputs = model(text_embeddings, lengths=lengths, sequence_mask=mask)
        token_loss, accuracies = masked_rvq_cross_entropy(outputs["logits"], tokens, mask)
        contrastive = info_nce_loss(text_embeddings, outputs["motion_embedding"], temperature=cfg.temperature)

        running_loss += float(token_loss.item())
        running_contrastive += float(contrastive.item())
        running_acc += np.asarray(accuracies)
        num_batches += 1

    denom = max(1, num_batches)
    metrics = {"val_loss": running_loss / denom, "val_contrastive": running_contrastive / denom}
    for idx, value in enumerate(running_acc / denom, start=1):
        metrics[f"val_acc_layer_{idx}"] = float(value)
    return metrics


def fit_model(
    model: OptionATokenGenerator,
    cfg: MotionSConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_path: str | Path,
    text_encoder: Optional[TextEncoderAdapter] = None,
) -> pd.DataFrame:
    """Train with early stopping and return a metrics table for analysis."""

    num_training_steps = cfg.epochs * max(1, len(train_loader))
    extra_parameters: Optional[Iterable[nn.Parameter]] = None
    if text_encoder is not None:
        extra_parameters = [parameter for parameter in text_encoder.parameters() if parameter.requires_grad]
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, num_training_steps, extra_parameters=extra_parameters)
    scaler = GradScaler(enabled=cfg.amp and cfg.device.startswith("cuda"))
    best_val_loss = float("inf")
    best_epoch = -1
    patience_left = cfg.patience
    history: List[Dict[str, float]] = []

    for epoch in range(cfg.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, cfg, torch.device(cfg.device), text_encoder=text_encoder)
        val_metrics = evaluate_one_epoch(model, val_loader, cfg, torch.device(cfg.device), text_encoder=text_encoder)
        epoch_metrics = {"epoch": float(epoch + 1), **train_metrics, **val_metrics}
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, indent=2))

        current_val = val_metrics["val_loss"] + cfg.contrastive_weight * val_metrics["val_contrastive"]
        if current_val < best_val_loss:
            best_val_loss = current_val
            best_epoch = epoch + 1
            patience_left = cfg.patience
            save_checkpoint(save_path, model, optimizer, scheduler, epoch + 1, best_val_loss, cfg, text_encoder=text_encoder)
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch + 1}; best epoch was {best_epoch}.")
                break

    return pd.DataFrame(history)


def prepare_submission_lengths(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_embeddings: torch.Tensor,
    length_model: Optional[LengthEstimatorMLP],
    length_bin_centers: Optional[np.ndarray],
    fallback_regressor: Optional[GlossLengthRegressor],
    device: torch.device,
) -> np.ndarray:
    """Predict motion lengths for the test split using the best available length model."""

    if length_model is not None and length_bin_centers is not None:
        return predict_lengths_for_dataframe(test_df, text_embeddings, length_model, length_bin_centers, fallback_regressor, device)
    if fallback_regressor is None:
        fallback_regressor = fit_gloss_length_regressor(train_df)
    return predict_lengths_for_dataframe(test_df, text_embeddings, None, None, fallback_regressor, device)


def run_training_pipeline(cfg: MotionSConfig) -> pd.DataFrame:
    """Run the baseline training pipeline end to end."""

    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    train_df = load_motion_dataframe(cfg.train_csv)
    train_df, val_df = group_aware_train_val_split(train_df, val_fraction=0.1, seed=cfg.seed)
    text_encoder, model, _, _ = build_default_components(cfg, device=device)
    train_dataset, val_dataset = prepare_datasets(train_df, val_df, cfg, text_encoder=text_encoder)
    train_loader, val_loader = build_dataloaders(cfg, train_dataset, val_dataset)
    history = fit_model(model, cfg, train_loader, val_loader, Path(cfg.output_dir) / "option_a_best.pt", text_encoder=text_encoder)
    history.to_csv(Path(cfg.output_dir) / "training_history.csv", index=False)
    return history


@torch.no_grad()
def run_inference_pipeline(cfg: MotionSConfig) -> pd.DataFrame:
    """Run the inference pipeline and save a competition submission CSV."""

    device = torch.device(cfg.device)
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    train_df = load_motion_dataframe(cfg.train_csv)
    test_df = load_motion_dataframe(cfg.test_csv)
    text_encoder, model, length_model, length_bin_centers = build_default_components(cfg, device=device)
    checkpoint_path = Path(cfg.output_dir) / "option_a_best.pt"
    if checkpoint_path.exists():
        model = load_trained_option_a_model(checkpoint_path, cfg, device=device)

    texts = [build_text_input(sentence, gloss) for sentence, gloss in zip(test_df["sentence"], test_df["gloss"])]
    test_text_embeddings = cache_text_embeddings(text_encoder, texts, batch_size=64, device=device)
    fallback_regressor = fit_gloss_length_regressor(train_df)
    lengths = prepare_submission_lengths(
        train_df=train_df,
        test_df=test_df,
        text_embeddings=test_text_embeddings,
        length_model=length_model,
        length_bin_centers=length_bin_centers,
        fallback_regressor=fallback_regressor,
        device=device,
    )
    submission_df = generate_submission_frame(
        test_df=test_df,
        model=model,
        cfg=cfg,
        text_embeddings=test_text_embeddings,
        lengths=lengths,
        device=device,
        batch_size=cfg.batch_size,
    )
    validate_submission_frame(submission_df)
    submission_df.to_csv(Path(cfg.output_dir) / "submission.csv", index=False)
    return submission_df


def prepare_datasets_from_csvs(cfg: MotionSConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train CSV and produce a group-aware train/validation split."""

    train_df = load_motion_dataframe(cfg.train_csv)
    return group_aware_train_val_split(train_df, val_fraction=0.1, seed=cfg.seed)


def main() -> None:
    """Minimal entry point for local smoke tests and notebook imports."""

    cfg = MotionSConfig()
    ensure_dir(cfg.output_dir)
    print(json.dumps(asdict(cfg), indent=2))
    print("This file is intended to be imported into the notebook or used as a module.")
    print("Uncomment run_training_pipeline(cfg) or run_inference_pipeline(cfg) when the data paths are ready.")


if __name__ == "__main__":
    main()