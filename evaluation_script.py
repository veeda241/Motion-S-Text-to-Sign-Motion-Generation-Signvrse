"""
Evaluation Script for Text-to-Sign Motion Generation
=====================================================
Computes FID, R-Precision, and Diversity metrics on generated motion tokens.

Usage:
    python evaluation_script.py \
        --submission submission.csv \
        --ground_truth train.csv \
        --model_path rvq_vae_best.pth

The final score is:
    Score = 0.30 * FID_norm + 0.50 * R_Precision_norm + 0.20 * Diversity_norm
"""

import argparse
import numpy as np
import pandas as pd
from scipy import linalg
from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Token parsing helpers
# ---------------------------------------------------------------------------

def parse_token_column(series: pd.Series) -> np.ndarray:
    """Parse a space-separated token column into a ragged list of int arrays."""
    result = []
    for cell in series:
        tokens = list(map(int, str(cell).strip().split()))
        result.append(np.array(tokens, dtype=np.int32))
    return result


def load_submission(path: str) -> pd.DataFrame:
    """Load and validate a submission CSV."""
    df = pd.read_csv(path)
    required_cols = ["id", "base_tokens", "residual_1", "residual_2",
                     "residual_3", "residual_4", "residual_5"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    token_cols = required_cols[1:]
    for col in token_cols:
        for idx, cell in enumerate(df[col]):
            tokens = list(map(int, str(cell).strip().split()))
            if any(t < 0 or t > 511 for t in tokens):
                raise ValueError(
                    f"Row {idx}, column {col}: token values must be in [0, 511]"
                )
            if len(tokens) < 40 or len(tokens) > 800:
                raise ValueError(
                    f"Row {idx}, column {col}: sequence length {len(tokens)} "
                    f"must be in [40, 800]"
                )

    # Verify all layers have the same length per row
    for idx, row in df.iterrows():
        lengths = [
            len(str(row[c]).strip().split()) for c in token_cols
        ]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"Row {idx}: all 6 layers must have the same sequence length, "
                f"got {lengths}"
            )

    print(f"Submission validation passed: {len(df)} rows.")
    return df


# ---------------------------------------------------------------------------
# Simple feature extractor (token statistics as proxy features)
# ---------------------------------------------------------------------------

class TokenFeatureExtractor:
    """
    Lightweight feature extractor that converts a sequence of 6-layer RVQ tokens
    into a fixed-size feature vector using statistical moments.

    In a full evaluation pipeline the rvq_vae_best.pth decoder would be used
    to reconstruct the continuous motion and then a learned motion encoder
    would produce features.  Here we use token statistics as a stand-in that
    still captures structural differences between sequences.
    """

    def __init__(self, codebook_size: int = 512, feat_dim: int = 128):
        self.codebook_size = codebook_size
        self.feat_dim = feat_dim

    def _layer_features(self, tokens: np.ndarray) -> np.ndarray:
        """Compute statistical features for a single token layer."""
        if len(tokens) == 0:
            return np.zeros(8)
        hist, _ = np.histogram(tokens, bins=16, range=(0, self.codebook_size))
        hist = hist / (hist.sum() + 1e-8)
        feats = np.array([
            tokens.mean() / self.codebook_size,
            tokens.std() / self.codebook_size,
            np.percentile(tokens, 25) / self.codebook_size,
            np.percentile(tokens, 75) / self.codebook_size,
            len(tokens) / 800.0,
            len(np.unique(tokens)) / self.codebook_size,
            np.diff(tokens.astype(float)).mean() / self.codebook_size
            if len(tokens) > 1 else 0.0,
            np.diff(tokens.astype(float)).std() / self.codebook_size
            if len(tokens) > 1 else 0.0,
        ])
        return feats

    def extract(self, row_tokens: List[np.ndarray]) -> np.ndarray:
        """
        row_tokens: list of 6 arrays (base + 5 residuals)
        Returns a flat feature vector of length feat_dim.
        """
        layer_feats = np.concatenate([self._layer_features(t) for t in row_tokens])
        # Pad or truncate to feat_dim
        if len(layer_feats) < self.feat_dim:
            layer_feats = np.pad(layer_feats, (0, self.feat_dim - len(layer_feats)))
        else:
            layer_feats = layer_feats[: self.feat_dim]
        return layer_feats

    def extract_batch(self, all_tokens: List[List[np.ndarray]]) -> np.ndarray:
        """Extract features for a list of samples."""
        return np.stack([self.extract(row) for row in all_tokens])


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

def compute_fid(real_feats: np.ndarray, fake_feats: np.ndarray) -> float:
    """Fréchet Inception Distance between two feature sets."""
    mu_r, sigma_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_f, sigma_f = fake_feats.mean(axis=0), np.cov(fake_feats, rowvar=False)

    diff = mu_r - mu_f
    # Compute sqrt of product of covariances
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))
    return max(fid, 0.0)


# ---------------------------------------------------------------------------
# R-Precision
# ---------------------------------------------------------------------------

def compute_r_precision(
    motion_feats: np.ndarray,
    text_feats: np.ndarray,
    top_k: int = 3,
    pool_size: int = 32,
) -> float:
    """
    R-Precision@top_k:
    For each sample, form a pool of 1 correct + (pool_size-1) random text
    features and check whether the correct one is in the top-k nearest
    neighbours of the motion feature (cosine similarity).
    """
    n = len(motion_feats)
    if n < pool_size:
        pool_size = n

    # L2-normalise for cosine similarity
    m_norm = motion_feats / (np.linalg.norm(motion_feats, axis=1, keepdims=True) + 1e-8)
    t_norm = text_feats / (np.linalg.norm(text_feats, axis=1, keepdims=True) + 1e-8)

    correct = 0
    rng = np.random.default_rng(42)
    for i in range(n):
        # Build pool indices: i plus (pool_size-1) random others
        others = rng.choice(
            [j for j in range(n) if j != i],
            size=pool_size - 1,
            replace=False,
        )
        pool_idx = np.array([i] + list(others))
        pool_t = t_norm[pool_idx]  # (pool_size, D)

        sims = pool_t @ m_norm[i]  # (pool_size,)
        top_k_idx = np.argsort(-sims)[:top_k]
        if 0 in top_k_idx:  # index 0 = correct text
            correct += 1

    return correct / n


# ---------------------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------------------

def compute_diversity(motion_feats: np.ndarray, num_pairs: int = 300) -> float:
    """
    Average pairwise Euclidean distance between randomly sampled motion pairs.
    """
    n = len(motion_feats)
    rng = np.random.default_rng(42)
    idx1 = rng.choice(n, size=num_pairs, replace=True)
    idx2 = rng.choice(n, size=num_pairs, replace=True)
    dists = np.linalg.norm(motion_feats[idx1] - motion_feats[idx2], axis=1)
    return float(dists.mean())


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalise_fid(fid: float, fid_max: float = 1000.0) -> float:
    """Map FID ∈ [0, fid_max] → [0, 1] (lower FID = higher score)."""
    return max(0.0, 1.0 - fid / fid_max)


def normalise_diversity(div: float, div_max: float = 10.0) -> float:
    """Map diversity ∈ [0, div_max] → [0, 1]."""
    return min(div / div_max, 1.0)


# ---------------------------------------------------------------------------
# Synthetic text features (placeholder)
# ---------------------------------------------------------------------------

def build_text_features(texts: List[str], feat_dim: int = 128) -> np.ndarray:
    """
    Build text feature vectors.

    When the competition models are available this would use the CLIP text
    encoder.  As a lightweight stand-in we use hashed bag-of-words vectors.
    """
    rng = np.random.default_rng(0)
    feats = []
    for text in texts:
        words = text.lower().split()
        vec = np.zeros(feat_dim)
        for word in words:
            h = hash(word) % feat_dim
            vec[h] += 1.0
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        feats.append(vec)
    return np.stack(feats) if feats else np.zeros((0, feat_dim))


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(
    submission_path: str,
    ground_truth_path: str,
    model_path: Optional[str] = None,
    feat_dim: int = 128,
) -> dict:
    """
    Run evaluation and return a dictionary of metrics.

    Parameters
    ----------
    submission_path : str
        Path to the generated submission.csv.
    ground_truth_path : str
        Path to train.csv (used to compute the real motion feature distribution).
    model_path : str, optional
        Path to rvq_vae_best.pth.  When provided the VAE decoder can be used
        for higher-fidelity feature extraction (not yet implemented here).
    feat_dim : int
        Dimensionality of feature vectors.
    """
    print("Loading submission …")
    sub_df = load_submission(submission_path)

    print("Loading ground-truth (training data) …")
    gt_df = pd.read_csv(ground_truth_path)

    token_layer_cols = ["base_tokens", "residual_1", "residual_2",
                        "residual_3", "residual_4", "residual_5"]

    # ---- Feature extractor ----
    extractor = TokenFeatureExtractor(feat_dim=feat_dim)

    # ---- Real features from training set ----
    print("Extracting real motion features …")
    real_token_rows = []
    for _, row in gt_df.iterrows():
        layers = [parse_token_column(pd.Series([row[c]]))[0] for c in token_layer_cols
                  if c in gt_df.columns]
        if len(layers) == 6:
            real_token_rows.append(layers)
    if not real_token_rows:
        raise ValueError(
            "Ground-truth CSV does not contain the 6 token columns. "
            "Ensure train.csv includes base_tokens and residual_1-5."
        )
    real_feats = extractor.extract_batch(real_token_rows[:3000])

    # ---- Generated features from submission ----
    print("Extracting generated motion features …")
    gen_token_rows = []
    for _, row in sub_df.iterrows():
        layers = [parse_token_column(pd.Series([row[c]]))[0] for c in token_layer_cols]
        gen_token_rows.append(layers)
    gen_feats = extractor.extract_batch(gen_token_rows)

    # ---- Text features ----
    text_col = "gloss" if "gloss" in sub_df.columns else (
        "sentence" if "sentence" in sub_df.columns else None
    )
    if text_col and text_col in sub_df.columns:
        texts = sub_df[text_col].fillna("").tolist()
    else:
        texts = [f"sample_{i}" for i in range(len(sub_df))]
    text_feats = build_text_features(texts, feat_dim=feat_dim)

    # ---- Compute metrics ----
    print("Computing FID …")
    # Use min(len) to ensure comparable distributions
    n = min(len(real_feats), len(gen_feats))
    fid = compute_fid(real_feats[:n], gen_feats[:n])

    print("Computing R-Precision …")
    r_prec = compute_r_precision(gen_feats, text_feats, top_k=3, pool_size=32)

    print("Computing Diversity …")
    diversity = compute_diversity(gen_feats, num_pairs=300)

    # ---- Normalise ----
    fid_norm = normalise_fid(fid)
    r_prec_norm = r_prec  # already in [0, 1]
    div_norm = normalise_diversity(diversity)

    final_score = 0.30 * fid_norm + 0.50 * r_prec_norm + 0.20 * div_norm

    metrics = {
        "FID": round(fid, 4),
        "FID_norm": round(fid_norm, 4),
        "R_Precision@3": round(r_prec, 4),
        "R_Precision_norm": round(r_prec_norm, 4),
        "Diversity": round(diversity, 4),
        "Diversity_norm": round(div_norm, 4),
        "Final_Score": round(final_score, 4),
    }

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sign language motion token generation."
    )
    parser.add_argument(
        "--submission", required=True,
        help="Path to the submission CSV file."
    )
    parser.add_argument(
        "--ground_truth", required=True,
        help="Path to the ground-truth train CSV file."
    )
    parser.add_argument(
        "--model_path", default=None,
        help="(Optional) Path to rvq_vae_best.pth for VAE-based feature extraction."
    )
    parser.add_argument(
        "--feat_dim", type=int, default=128,
        help="Feature dimensionality for evaluation (default: 128)."
    )
    args = parser.parse_args()

    metrics = evaluate(
        submission_path=args.submission,
        ground_truth_path=args.ground_truth,
        model_path=args.model_path,
        feat_dim=args.feat_dim,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("=" * 50)
    print(f"\n  Final Score: {metrics['Final_Score']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
