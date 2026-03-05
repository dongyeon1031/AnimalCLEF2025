#!/usr/bin/env python3
"""
Local reranker-only experiment pipeline.

This script is intentionally separated from the original fusion pipeline.
It evaluates Top-1 accuracy on random candidate pools where each pool
explicitly includes at least one ground-truth (positive) database sample.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

INVALID_IDENTITY_STRINGS = {"", "nan", "none", "null", "new_individual", "unknown"}


class SimpleImageDataset:
    """Minimal dataset compatible with wildlife_tools extractors and matchers."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        transform=None,
        load_label: bool = True,
        apply_salamander_orientation_fix: bool = True,
        col_path: str = "path",
        col_label: str = "identity",
    ):
        self.metadata = metadata.reset_index(drop=True).copy()
        self.transform = transform
        self.load_label = load_label
        self.apply_salamander_orientation_fix = apply_salamander_orientation_fix
        self.col_path = col_path
        self.col_label = col_label
        self.labels, self.labels_map = pd.factorize(self.metadata[self.col_label].astype(str).values)

    @property
    def labels_string(self) -> np.ndarray:
        return self.metadata[self.col_label].astype(str).values

    def __len__(self) -> int:
        return len(self.metadata)

    def get_image(self, idx: int) -> Image.Image:
        row = self.metadata.iloc[idx]
        img_path = row[self.col_path]
        img = Image.open(img_path).convert("RGB")
        if self.apply_salamander_orientation_fix:
            dataset_name = str(row.get("dataset", ""))
            if dataset_name == "SalamanderID2025":
                orientation = str(row.get("orientation", "top"))
                if orientation == "right":
                    img = img.rotate(-90, expand=True)
                elif orientation == "left":
                    img = img.rotate(90, expand=True)
        return img

    def __getitem__(self, idx: int):
        img = self.get_image(idx)
        if self.transform is not None:
            img = self.transform(img)
        if self.load_label:
            return img, int(self.labels[idx])
        return img

    def get_subset(self, idx: Sequence[int] | Sequence[bool] | np.ndarray | pd.Series):
        if isinstance(idx, pd.Series):
            idx = idx.to_numpy()
        idx_arr = np.asarray(idx)
        if idx_arr.dtype == bool and len(idx_arr) == len(self.metadata):
            subset_df = self.metadata[idx_arr]
        else:
            subset_df = self.metadata.iloc[idx_arr]
        return SimpleImageDataset(
            subset_df,
            transform=self.transform,
            load_label=self.load_label,
            apply_salamander_orientation_fix=self.apply_salamander_orientation_fix,
            col_path=self.col_path,
            col_label=self.col_label,
        )


class PairCountCollector:
    """
    Sparse collector for pairwise local matchers.
    Stores one scalar score per requested pair: number of matches over threshold.
    """

    def __init__(self, confidence_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.data: list[tuple[int, int, float]] = []

    def init_store(self, **kwargs):
        self.data = []

    def add(self, results_list: list[dict]):
        for item in results_list:
            pair_score = float(np.sum(np.asarray(item["scores"]) > self.confidence_threshold))
            self.data.append((int(item["idx0"]), int(item["idx1"]), pair_score))

    def process_results(self):
        return self.data


class BaseLocalMatcher:
    def prepare(self, query_dataset: SimpleImageDataset, db_dataset: SimpleImageDataset):
        raise NotImplementedError

    def score_pairs(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        raise NotImplementedError

    def get_pair_matches(self, q_idx: int, db_idx: int) -> dict[str, Any]:
        raise NotImplementedError

    def get_visual_image(self, side: str, idx: int) -> Image.Image:
        raise NotImplementedError


class AlikedLightGlueLocalMatcher(BaseLocalMatcher):
    def __init__(
        self,
        device: str,
        batch_size: int,
        num_workers: int,
        confidence_threshold: float,
        max_num_keypoints: int,
    ):
        from wildlife_tools.features import AlikedExtractor
        from wildlife_tools.similarity import MatchLightGlue
        import torchvision.transforms as T

        self.transform = T.Compose([T.Resize([512, 512]), T.ToTensor()])
        self.extractor = AlikedExtractor(device=device, max_num_keypoints=max_num_keypoints)
        self.matcher = MatchLightGlue(
            features="aliked",
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            collector=PairCountCollector(confidence_threshold),
            tqdm_silent=False,
        )
        self.query_features = None
        self.db_features = None
        self.query_dataset = None
        self.db_dataset = None

    def prepare(self, query_dataset: SimpleImageDataset, db_dataset: SimpleImageDataset):
        query_copy = query_dataset.get_subset(np.arange(len(query_dataset)))
        db_copy = db_dataset.get_subset(np.arange(len(db_dataset)))
        query_copy.transform = self.transform
        db_copy.transform = self.transform
        self.query_dataset = query_copy
        self.db_dataset = db_copy
        self.query_features = self.extractor(query_copy)
        self.db_features = self.extractor(db_copy)

    def score_pairs(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        if len(pairs) == 0:
            return np.array([], dtype=np.float32)
        if self.query_features is None or self.db_features is None:
            raise RuntimeError("prepare() must be called before score_pairs().")
        pairs_np = np.asarray(pairs, dtype=np.int64)
        rows = self.matcher(self.query_features, self.db_features, pairs=pairs_np)
        score_map = {(q, d): score for q, d, score in rows}
        return np.asarray([score_map.get((int(q), int(d)), 0.0) for q, d in pairs], dtype=np.float32)

    def get_pair_matches(self, q_idx: int, db_idx: int) -> dict[str, Any]:
        from wildlife_tools.similarity import CollectAll

        if self.query_features is None or self.db_features is None:
            raise RuntimeError("prepare() must be called before get_pair_matches().")

        pairs_np = np.asarray([(int(q_idx), int(db_idx))], dtype=np.int64)
        old_collector = self.matcher.collector
        try:
            self.matcher.collector = CollectAll()
            rows = self.matcher(self.query_features, self.db_features, pairs=pairs_np)
        finally:
            self.matcher.collector = old_collector
        if len(rows) == 0:
            return {"idx0": int(q_idx), "idx1": int(db_idx), "kpts0": np.empty((0, 2)), "kpts1": np.empty((0, 2)), "scores": np.empty((0,))}
        return rows[0]

    def get_visual_image(self, side: str, idx: int) -> Image.Image:
        if side not in {"query", "db"}:
            raise ValueError("side must be either 'query' or 'db'.")
        dataset = self.query_dataset if side == "query" else self.db_dataset
        if dataset is None:
            raise RuntimeError("prepare() must be called before get_visual_image().")
        sample = dataset[idx][0]
        return to_rgb_pil(sample)


class LoFTRLocalMatcher(BaseLocalMatcher):
    def __init__(
        self,
        device: str,
        batch_size: int,
        num_workers: int,
        confidence_threshold: float,
        pretrained: str,
    ):
        from wildlife_tools.similarity import MatchLOFTR
        import torchvision.transforms as T

        self.transform = T.Compose([T.Resize([480, 480]), T.Grayscale(num_output_channels=1), T.ToTensor()])
        self.matcher = MatchLOFTR(
            pretrained=pretrained,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            collector=PairCountCollector(confidence_threshold),
            tqdm_silent=False,
        )
        self.query_dataset = None
        self.db_dataset = None

    def prepare(self, query_dataset: SimpleImageDataset, db_dataset: SimpleImageDataset):
        query_copy = query_dataset.get_subset(np.arange(len(query_dataset)))
        db_copy = db_dataset.get_subset(np.arange(len(db_dataset)))
        query_copy.transform = self.transform
        db_copy.transform = self.transform
        self.query_dataset = query_copy
        self.db_dataset = db_copy

    def score_pairs(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        if len(pairs) == 0:
            return np.array([], dtype=np.float32)
        if self.query_dataset is None or self.db_dataset is None:
            raise RuntimeError("prepare() must be called before score_pairs().")
        pairs_np = np.asarray(pairs, dtype=np.int64)
        rows = self.matcher(self.query_dataset, self.db_dataset, pairs=pairs_np)
        score_map = {(q, d): score for q, d, score in rows}
        return np.asarray([score_map.get((int(q), int(d)), 0.0) for q, d in pairs], dtype=np.float32)

    def get_pair_matches(self, q_idx: int, db_idx: int) -> dict[str, Any]:
        from wildlife_tools.similarity import CollectAll

        if self.query_dataset is None or self.db_dataset is None:
            raise RuntimeError("prepare() must be called before get_pair_matches().")

        pairs_np = np.asarray([(int(q_idx), int(db_idx))], dtype=np.int64)
        old_collector = self.matcher.collector
        try:
            self.matcher.collector = CollectAll()
            rows = self.matcher(self.query_dataset, self.db_dataset, pairs=pairs_np)
        finally:
            self.matcher.collector = old_collector
        if len(rows) == 0:
            return {"idx0": int(q_idx), "idx1": int(db_idx), "kpts0": np.empty((0, 2)), "kpts1": np.empty((0, 2)), "scores": np.empty((0,))}
        return rows[0]

    def get_visual_image(self, side: str, idx: int) -> Image.Image:
        if side not in {"query", "db"}:
            raise ValueError("side must be either 'query' or 'db'.")
        dataset = self.query_dataset if side == "query" else self.db_dataset
        if dataset is None:
            raise RuntimeError("prepare() must be called before get_visual_image().")
        sample = dataset[idx][0]
        return to_rgb_pil(sample)


class RoMALocalMatcher(BaseLocalMatcher):
    """RoMA dense matcher wrapper for pair scoring + visualization."""

    def __init__(
        self,
        device: str,
        variant: str,
        cert_threshold: float,
        max_samples: int,
        score_mode: str,
        coarse_res: int,
        upsample_res: int,
    ):
        import torch
        import romatch

        if variant == "outdoor":
            builder = romatch.roma_outdoor
        elif variant == "indoor":
            builder = romatch.roma_indoor
        else:
            raise ValueError(f"Unsupported RoMA variant: {variant}")

        amp_dtype = torch.float16 if device == "cuda" else torch.float32
        self.device = device
        self.cert_threshold = float(cert_threshold)
        self.max_samples = int(max_samples)
        self.score_mode = str(score_mode)
        self.query_dataset = None
        self.db_dataset = None
        self._torch = torch
        self.matcher = builder(
            device=device,
            coarse_res=int(coarse_res),
            upsample_res=int(upsample_res),
            amp_dtype=amp_dtype,
        )
        self._pair_cache: dict[tuple[int, int], dict[str, Any]] = {}

    def prepare(self, query_dataset: SimpleImageDataset, db_dataset: SimpleImageDataset):
        self.query_dataset = query_dataset
        self.db_dataset = db_dataset
        self._pair_cache = {}

    def _compute_pair(self, q_idx: int, db_idx: int) -> dict[str, Any]:
        key = (int(q_idx), int(db_idx))
        cached = self._pair_cache.get(key)
        if cached is not None:
            return cached
        if self.query_dataset is None or self.db_dataset is None:
            raise RuntimeError("prepare() must be called before pair matching.")

        img_q = self.query_dataset.get_image(int(q_idx)).convert("RGB")
        img_d = self.db_dataset.get_image(int(db_idx)).convert("RGB")

        with self._torch.inference_mode():
            warp, cert = self.matcher.match(img_q, img_d, batched=True, device=self.device)
        if warp.ndim == 4:
            warp = warp[0]
        if cert.ndim == 3:
            cert = cert[0]
        cert = cert.float()

        if self.score_mode == "count":
            pair_score = float((cert > self.cert_threshold).sum().item())
        elif self.score_mode == "sum":
            pair_score = float(cert.sum().item())
        elif self.score_mode == "sum_above":
            pair_score = float((cert - self.cert_threshold).clamp(min=0.0).sum().item())
        else:
            raise ValueError(f"Unsupported roma-score-mode: {self.score_mode}")

        kpts0 = np.empty((0, 2), dtype=np.float32)
        kpts1 = np.empty((0, 2), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
        if self.max_samples > 0 and cert.numel() > 0:
            sample_num = int(min(self.max_samples, cert.numel()))
            sampled_matches, sampled_scores = self.matcher.sample(warp, cert, num=sample_num)
            keep_mask = sampled_scores > self.cert_threshold
            sampled_matches = sampled_matches[keep_mask]
            sampled_scores = sampled_scores[keep_mask]
            if len(sampled_matches) > 0:
                kpts0_t, kpts1_t = self.matcher.to_pixel_coordinates(
                    sampled_matches,
                    H_A=img_q.height,
                    W_A=img_q.width,
                    H_B=img_d.height,
                    W_B=img_d.width,
                )
                kpts0 = kpts0_t.detach().cpu().numpy().astype(np.float32)
                kpts1 = kpts1_t.detach().cpu().numpy().astype(np.float32)
                scores = sampled_scores.detach().cpu().numpy().astype(np.float32)

        out = {
            "idx0": int(q_idx),
            "idx1": int(db_idx),
            "kpts0": kpts0,
            "kpts1": kpts1,
            "scores": scores,
            "pair_score": pair_score,
        }
        self._pair_cache[key] = out
        return out

    def score_pairs(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        if len(pairs) == 0:
            return np.array([], dtype=np.float32)
        rows = [self._compute_pair(int(q), int(d)) for q, d in pairs]
        return np.asarray([float(row["pair_score"]) for row in rows], dtype=np.float32)

    def get_pair_matches(self, q_idx: int, db_idx: int) -> dict[str, Any]:
        row = self._compute_pair(int(q_idx), int(db_idx))
        return {
            "idx0": row["idx0"],
            "idx1": row["idx1"],
            "kpts0": row["kpts0"],
            "kpts1": row["kpts1"],
            "scores": row["scores"],
        }

    def get_visual_image(self, side: str, idx: int) -> Image.Image:
        if side not in {"query", "db"}:
            raise ValueError("side must be either 'query' or 'db'.")
        dataset = self.query_dataset if side == "query" else self.db_dataset
        if dataset is None:
            raise RuntimeError("prepare() must be called before get_visual_image().")
        return dataset.get_image(idx).convert("RGB")


class ORBLocalMatcher(BaseLocalMatcher):
    """Weight-free local matcher for offline smoke tests."""

    def __init__(self, n_features: int = 1500, ratio_test: float = 0.75):
        import cv2

        self.cv2 = cv2
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_test = ratio_test
        self.query_dataset = None
        self.db_dataset = None
        self.query_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.db_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def prepare(self, query_dataset: SimpleImageDataset, db_dataset: SimpleImageDataset):
        self.query_dataset = query_dataset
        self.db_dataset = db_dataset
        self.query_cache = {}
        self.db_cache = {}

    def _compute_features(
        self,
        dataset: SimpleImageDataset,
        idx: int,
        cache: dict[int, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if idx in cache:
            return cache[idx]
        img = np.asarray(dataset.get_image(idx).convert("L"))
        keypoints, desc = self.orb.detectAndCompute(img, None)
        if desc is None:
            desc = np.empty((0, 32), dtype=np.uint8)
            points = np.empty((0, 2), dtype=np.float32)
        else:
            points = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
        cache[idx] = (points, desc)
        return cache[idx]

    def _match_pair(self, q_idx: int, db_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.query_dataset is None or self.db_dataset is None:
            raise RuntimeError("prepare() must be called before pair matching.")
        pts_q, desc_q = self._compute_features(self.query_dataset, q_idx, self.query_cache)
        pts_d, desc_d = self._compute_features(self.db_dataset, db_idx, self.db_cache)
        if len(desc_q) == 0 or len(desc_d) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        knn_matches = self.bf.knnMatch(desc_q, desc_d, k=2)
        matched_q = []
        matched_d = []
        scores = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                matched_q.append(pts_q[m.queryIdx])
                matched_d.append(pts_d[m.trainIdx])
                # Convert distance to similarity-like score in [0,1]
                scores.append(float(max(0.0, 1.0 - (m.distance / 256.0))))
        if len(matched_q) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return (
            np.asarray(matched_q, dtype=np.float32),
            np.asarray(matched_d, dtype=np.float32),
            np.asarray(scores, dtype=np.float32),
        )

    def _score_pair(self, q_idx: int, db_idx: int) -> float:
        kpts0, _, _ = self._match_pair(q_idx, db_idx)
        return float(len(kpts0))

    def score_pairs(self, pairs: list[tuple[int, int]]) -> np.ndarray:
        if len(pairs) == 0:
            return np.array([], dtype=np.float32)
        scores = [self._score_pair(int(q), int(d)) for q, d in pairs]
        return np.asarray(scores, dtype=np.float32)

    def get_pair_matches(self, q_idx: int, db_idx: int) -> dict[str, Any]:
        kpts0, kpts1, scores = self._match_pair(int(q_idx), int(db_idx))
        return {
            "idx0": int(q_idx),
            "idx1": int(db_idx),
            "kpts0": kpts0,
            "kpts1": kpts1,
            "scores": scores,
        }

    def get_visual_image(self, side: str, idx: int) -> Image.Image:
        if side not in {"query", "db"}:
            raise ValueError("side must be either 'query' or 'db'.")
        dataset = self.query_dataset if side == "query" else self.db_dataset
        if dataset is None:
            raise RuntimeError("prepare() must be called before get_visual_image().")
        return dataset.get_image(idx).convert("RGB")


def to_rgb_pil(image_like: Any) -> Image.Image:
    if isinstance(image_like, Image.Image):
        return image_like.convert("RGB")

    try:
        import torch

        if torch.is_tensor(image_like):
            tensor = image_like.detach().cpu()
            if tensor.ndim != 3:
                raise ValueError(f"Expected 3D tensor (C,H,W), got shape={tuple(tensor.shape)}")
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            elif tensor.shape[0] > 3:
                tensor = tensor[:3]
            tensor = tensor.float()
            if tensor.max() <= 1.5:
                tensor = tensor.clamp(0.0, 1.0) * 255.0
            tensor = tensor.round().clamp(0, 255).byte()
            array = tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(array).convert("RGB")
    except ImportError:
        pass

    if isinstance(image_like, np.ndarray):
        array = image_like
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array).convert("RGB")

    raise TypeError(f"Unsupported image type for visualization: {type(image_like)}")


def sanitize_for_filename(text: Any) -> str:
    value = str(text)
    safe = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    compact = "".join(safe).strip("_")
    return compact if compact else "unknown"


def get_line_color(i: int) -> tuple[int, int, int]:
    r = (37 * i + 79) % 206 + 50
    g = (97 * i + 31) % 206 + 50
    b = (53 * i + 163) % 206 + 50
    return int(r), int(g), int(b)


def render_pair_matches(
    img0: Image.Image,
    img1: Image.Image,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    scores: np.ndarray,
    max_matches: int,
    title_text: str,
) -> Image.Image:
    img0 = to_rgb_pil(img0)
    img1 = to_rgb_pil(img1)
    w0, h0 = img0.size
    w1, h1 = img1.size
    canvas_h = max(h0, h1) + 36
    canvas_w = w0 + w1
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 18, 18))
    canvas.paste(img0, (0, 36))
    canvas.paste(img1, (w0, 36))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title_text, fill=(235, 235, 235))

    kpts0 = np.asarray(kpts0, dtype=np.float32).reshape(-1, 2)
    kpts1 = np.asarray(kpts1, dtype=np.float32).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    n = min(len(kpts0), len(kpts1))
    if n == 0:
        draw.text((10, 24), "No matches", fill=(255, 120, 120))
        return canvas

    if len(scores) < n:
        padded = np.zeros((n,), dtype=np.float32)
        padded[: len(scores)] = scores
        scores = padded
    else:
        scores = scores[:n]

    order = np.argsort(-scores)[: min(max_matches, n)]
    for i, idx in enumerate(order):
        x0, y0 = float(kpts0[idx][0]), float(kpts0[idx][1] + 36)
        x1, y1 = float(kpts1[idx][0] + w0), float(kpts1[idx][1] + 36)
        color = get_line_color(i)
        draw.line([(x0, y0), (x1, y1)], fill=color, width=1)
        r = 2
        draw.ellipse((x0 - r, y0 - r, x0 + r, y0 + r), outline=color)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), outline=color)

    draw.text((10, 24), f"matches shown: {len(order)} / {n}", fill=(180, 220, 255))
    return canvas


def select_visualization_rows(
    results_df: pd.DataFrame,
    per_dataset: int,
) -> pd.DataFrame:
    selected_parts = []
    for dataset_name, group in results_df.groupby("dataset"):
        group = group.sort_values(["scenario_id"]).reset_index(drop=True)
        good = group[group["top1_hit"] == 1].sort_values(["gt_rank_in_candidates", "scenario_id"]).head(per_dataset)
        bad = group[group["top1_hit"] == 0].sort_values(["gt_rank_in_candidates", "scenario_id"], ascending=[False, True]).head(per_dataset)
        if len(good) > 0:
            good = good.copy()
            good["viz_bucket"] = "good"
            selected_parts.append(good)
        if len(bad) > 0:
            bad = bad.copy()
            bad["viz_bucket"] = "bad"
            selected_parts.append(bad)

        # If one side is empty, fill with whichever exists to keep up to 2*per_dataset samples.
        if len(good) < per_dataset:
            extra = group[group["top1_hit"] == 0].sort_values(["gt_rank_in_candidates", "scenario_id"], ascending=[False, True]).head(per_dataset - len(good))
            if len(extra) > 0:
                extra = extra.copy()
                extra["viz_bucket"] = "bad_fill"
                selected_parts.append(extra)
        if len(bad) < per_dataset:
            extra = group[group["top1_hit"] == 1].sort_values(["gt_rank_in_candidates", "scenario_id"]).head(per_dataset - len(bad))
            if len(extra) > 0:
                extra = extra.copy()
                extra["viz_bucket"] = "good_fill"
                selected_parts.append(extra)

    if len(selected_parts) == 0:
        return pd.DataFrame()

    selected = pd.concat(selected_parts, axis=0, ignore_index=True)
    selected = selected.drop_duplicates(subset=["scenario_id"]).sort_values(["dataset", "viz_bucket", "scenario_id"])
    return selected.reset_index(drop=True)


def generate_match_visualizations(
    results_df: pd.DataFrame,
    query_dataset: SimpleImageDataset,
    db_dataset: SimpleImageDataset,
    matcher: BaseLocalMatcher,
    matcher_name: str,
    output_dir: Path,
    per_dataset: int,
    max_matches: int,
) -> list[Path]:
    if per_dataset <= 0:
        return []

    selected = select_visualization_rows(results_df, per_dataset=per_dataset)
    if len(selected) == 0:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for _, row in selected.iterrows():
        q_idx = int(row["query_idx"])
        db_idx = int(row["pred_db_idx"])
        dataset_name = str(row.get("dataset", "all"))
        bucket = str(row.get("viz_bucket", "sample"))
        scenario_id = int(row["scenario_id"])
        query_image_id = sanitize_for_filename(row.get("query_image_id", f"q{q_idx}"))
        pred_identity = sanitize_for_filename(row.get("pred_identity", f"db{db_idx}"))
        top1_hit = int(row.get("top1_hit", -1))
        gt_rank = int(row.get("gt_rank_in_candidates", -1))

        try:
            matches = matcher.get_pair_matches(q_idx, db_idx)
            img_q = matcher.get_visual_image("query", q_idx)
            img_db = matcher.get_visual_image("db", db_idx)
        except Exception as exc:
            print(f"[Warn] Visualization skipped for scenario={scenario_id}: {exc}")
            continue

        title = (
            f"{matcher_name} | {dataset_name} | {bucket} | "
            f"scenario={scenario_id} hit={top1_hit} gt_rank={gt_rank}"
        )
        canvas = render_pair_matches(
            img0=img_q,
            img1=img_db,
            kpts0=np.asarray(matches.get("kpts0", np.empty((0, 2)))),
            kpts1=np.asarray(matches.get("kpts1", np.empty((0, 2)))),
            scores=np.asarray(matches.get("scores", np.empty((0,)))),
            max_matches=max_matches,
            title_text=title,
        )

        file_name = (
            f"{sanitize_for_filename(dataset_name)}_{bucket}_"
            f"scenario{scenario_id}_q{query_image_id}_pred{pred_identity}.jpg"
        )
        path = output_dir / file_name
        canvas.save(path, quality=95)
        saved_paths.append(path)

    return saved_paths


@dataclass
class Scenario:
    scenario_id: int
    query_idx: int
    trial: int
    dataset_name: str
    query_image_id: str
    query_identity: str
    candidate_db_indices: np.ndarray
    injected_positive_db_idx: int


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def to_absolute_paths(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    df = df.copy()
    if "path" not in df.columns:
        raise ValueError("metadata.csv must contain `path` column.")
    df["path"] = df["path"].astype(str).map(lambda p: p if os.path.isabs(p) else str(root / p))
    return df


def load_real_dataset(root: Path, auto_download: bool, check_paths: bool) -> tuple[SimpleImageDataset, SimpleImageDataset]:
    metadata_path = root / "metadata.csv"
    if auto_download and not metadata_path.exists():
        from wildlife_datasets.datasets import AnimalCLEF2025

        AnimalCLEF2025.get_data(str(root))

    if not metadata_path.exists():
        raise FileNotFoundError(f"`metadata.csv` not found at: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    required_columns = {"split", "identity", "path"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {sorted(missing)}")

    metadata = to_absolute_paths(metadata, root)
    if check_paths:
        missing_rows = metadata[~metadata["path"].map(os.path.exists)]
        if len(missing_rows) > 0:
            raise FileNotFoundError(
                f"{len(missing_rows)} image paths from metadata.csv do not exist. "
                "Use `--no-check-paths` to skip this validation."
            )

    db_df = metadata[metadata["split"] == "database"].reset_index(drop=True)
    query_df = metadata[metadata["split"] == "query"].reset_index(drop=True)
    if len(db_df) == 0 or len(query_df) == 0:
        raise ValueError("No `database` or `query` rows found in metadata.csv")

    db_dataset = SimpleImageDataset(db_df, load_label=True)
    query_dataset = SimpleImageDataset(query_df, load_label=True)
    return query_dataset, db_dataset


def normalize_identity(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.lower() in INVALID_IDENTITY_STRINGS:
        return None
    return text


def normalized_id_array(dataset: SimpleImageDataset) -> np.ndarray:
    if dataset.col_label not in dataset.metadata.columns:
        raise ValueError(f"Dataset metadata is missing `{dataset.col_label}` column.")
    return np.asarray([normalize_identity(v) for v in dataset.metadata[dataset.col_label].values], dtype=object)


def identity_overlap_count(query_dataset: SimpleImageDataset, db_dataset: SimpleImageDataset) -> int:
    query_norm = normalized_id_array(query_dataset)
    db_norm = normalized_id_array(db_dataset)
    query_set = {x for x in query_norm if x is not None}
    db_set = {x for x in db_norm if x is not None}
    return len(query_set & db_set)


def build_db_self_eval_split(
    db_dataset: SimpleImageDataset,
    per_identity_queries: int,
    seed: int,
) -> tuple[SimpleImageDataset, SimpleImageDataset]:
    if per_identity_queries < 1:
        raise ValueError("per_identity_queries must be >= 1")

    df = db_dataset.metadata.reset_index(drop=True)
    if "identity" not in df.columns:
        raise ValueError("Database metadata must include `identity` for db_self_eval mode.")

    rng = np.random.default_rng(seed)
    query_indices = []

    grouped = df.groupby("identity").indices
    for identity, idx_like in grouped.items():
        norm_id = normalize_identity(identity)
        idx = np.asarray(idx_like, dtype=int)
        if norm_id is None or len(idx) < 2:
            continue
        shuffled = idx.copy()
        rng.shuffle(shuffled)
        n_q = min(per_identity_queries, len(shuffled) - 1)
        query_indices.extend(shuffled[:n_q].tolist())

    if len(query_indices) == 0:
        raise RuntimeError(
            "db_self_eval split failed: no identity has at least 2 valid images in database."
        )

    query_indices = np.array(sorted(set(query_indices)), dtype=int)
    db_mask = np.ones(len(df), dtype=bool)
    db_mask[query_indices] = False

    query_dataset = db_dataset.get_subset(query_indices)
    db_dataset_new = db_dataset.get_subset(db_mask)
    return query_dataset, db_dataset_new


def build_pattern_image(identity_idx: int, variant_idx: int, size: int = 192) -> Image.Image:
    base_rng = np.random.default_rng(identity_idx)
    bg_color = tuple(int(v) for v in base_rng.integers(50, 180, size=3))
    img = Image.new("RGB", (size, size), color=bg_color)
    draw = ImageDraw.Draw(img)

    line_rng = np.random.default_rng(identity_idx * 100 + 7)
    for _ in range(14):
        x1 = int(line_rng.integers(0, size))
        y1 = int(line_rng.integers(0, size))
        x2 = int(line_rng.integers(0, size))
        y2 = int(line_rng.integers(0, size))
        color = tuple(int(v) for v in line_rng.integers(100, 255, size=3))
        width = int(line_rng.integers(1, 4))
        draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    rot = ((variant_idx % 9) - 4) * 1.5
    img = img.rotate(rot, resample=Image.Resampling.BILINEAR)

    arr = np.asarray(img).astype(np.int16)
    noise_rng = np.random.default_rng(identity_idx * 1000 + variant_idx)
    noise = noise_rng.integers(-8, 9, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def create_synthetic_datasets(
    out_dir: Path,
    n_identities: int = 8,
    db_per_identity: int = 3,
) -> tuple[SimpleImageDataset, SimpleImageDataset]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for identity_idx in range(n_identities):
        identity = f"id_{identity_idx:03d}"
        for j in range(db_per_identity):
            file_name = f"db_{identity}_{j}.png"
            path = out_dir / file_name
            build_pattern_image(identity_idx, variant_idx=j).save(path)
            rows.append(
                {
                    "image_id": f"db_{identity}_{j}",
                    "identity": identity,
                    "split": "database",
                    "dataset": "SyntheticSet",
                    "path": str(path),
                }
            )

        file_name = f"q_{identity}.png"
        path = out_dir / file_name
        build_pattern_image(identity_idx, variant_idx=99).save(path)
        rows.append(
            {
                "image_id": f"q_{identity}",
                "identity": identity,
                "split": "query",
                "dataset": "SyntheticSet",
                "path": str(path),
            }
        )

    df = pd.DataFrame(rows)
    db_dataset = SimpleImageDataset(df[df["split"] == "database"].reset_index(drop=True), load_label=True)
    query_dataset = SimpleImageDataset(df[df["split"] == "query"].reset_index(drop=True), load_label=True)
    return query_dataset, db_dataset


def parse_dataset_filter(dataset_filter: str | None) -> set[str] | None:
    if dataset_filter is None or dataset_filter.strip() == "":
        return None
    values = [s.strip() for s in dataset_filter.split(",") if s.strip()]
    return set(values) if values else None


def build_scenarios(
    query_dataset: SimpleImageDataset,
    db_dataset: SimpleImageDataset,
    candidate_size: int,
    trials_per_query: int,
    max_queries: int | None,
    dataset_filter: set[str] | None,
    seed: int,
    same_dataset_only: bool,
) -> tuple[list[Scenario], int]:
    if candidate_size < 2:
        raise ValueError("candidate_size must be >= 2")

    query_labels = query_dataset.labels_string.astype(str)
    db_labels = db_dataset.labels_string.astype(str)
    query_labels_norm = normalized_id_array(query_dataset)
    db_labels_norm = normalized_id_array(db_dataset)
    rng = np.random.default_rng(seed)
    query_dataset_names = None
    db_dataset_names = None
    if same_dataset_only:
        if "dataset" not in query_dataset.metadata.columns or "dataset" not in db_dataset.metadata.columns:
            raise ValueError(
                "`same_dataset_only` requires `dataset` column in both query and database metadata."
            )
        query_dataset_names = query_dataset.metadata["dataset"].astype(str).to_numpy()
        db_dataset_names = db_dataset.metadata["dataset"].astype(str).to_numpy()

    query_indices = np.arange(len(query_dataset))
    if dataset_filter is not None and "dataset" in query_dataset.metadata.columns:
        mask = query_dataset.metadata["dataset"].astype(str).isin(dataset_filter).to_numpy()
        query_indices = query_indices[mask]

    if max_queries is not None and len(query_indices) > max_queries:
        query_indices = np.sort(rng.choice(query_indices, size=max_queries, replace=False))

    db_all_indices = np.arange(len(db_dataset))
    if len(db_all_indices) < 2:
        raise ValueError("Database must contain at least 2 images.")

    scenarios: list[Scenario] = []
    skipped_no_positive = 0

    scenario_id = 0
    for q_idx in query_indices:
        q_label = query_labels[q_idx]
        q_norm = query_labels_norm[q_idx]
        if same_dataset_only:
            q_dataset = query_dataset_names[q_idx]
            db_scope = np.where(db_dataset_names == q_dataset)[0]
            if len(db_scope) == 0:
                skipped_no_positive += 1
                continue
        else:
            db_scope = db_all_indices
        effective_candidate_size = min(candidate_size, len(db_scope))
        if effective_candidate_size < 2:
            skipped_no_positive += 1
            continue
        if q_norm is None:
            skipped_no_positive += 1
            continue
        positive_indices = db_scope[db_labels_norm[db_scope] == q_norm]
        if len(positive_indices) == 0:
            skipped_no_positive += 1
            continue
        negative_indices = db_scope[db_labels_norm[db_scope] != q_norm]

        for trial in range(trials_per_query):
            pos_idx = int(rng.choice(positive_indices))
            if len(negative_indices) >= effective_candidate_size - 1:
                sampled_neg = rng.choice(negative_indices, size=effective_candidate_size - 1, replace=False)
                candidate = np.concatenate([[pos_idx], sampled_neg])
            else:
                candidate = np.concatenate([[pos_idx], negative_indices])
                if len(candidate) < effective_candidate_size:
                    remaining = np.setdiff1d(db_scope, candidate, assume_unique=False)
                    extra = rng.choice(remaining, size=effective_candidate_size - len(candidate), replace=False)
                    candidate = np.concatenate([candidate, extra])

            candidate = candidate.astype(int)
            rng.shuffle(candidate)

            meta = query_dataset.metadata.iloc[q_idx]
            scenarios.append(
                Scenario(
                    scenario_id=scenario_id,
                    query_idx=int(q_idx),
                    trial=trial,
                    dataset_name=str(meta.get("dataset", "all")),
                    query_image_id=str(meta.get("image_id", f"q_{q_idx}")),
                    query_identity=str(q_label),
                    candidate_db_indices=candidate,
                    injected_positive_db_idx=pos_idx,
                )
            )
            scenario_id += 1

    return scenarios, skipped_no_positive


def build_matcher(args, device: str) -> BaseLocalMatcher:
    try:
        if args.matcher == "aliked":
            return AlikedLightGlueLocalMatcher(
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                confidence_threshold=args.confidence_threshold,
                max_num_keypoints=args.max_num_keypoints,
            )
        if args.matcher == "loftr":
            return LoFTRLocalMatcher(
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                confidence_threshold=args.confidence_threshold,
                pretrained=args.loftr_pretrained,
            )
        if args.matcher == "roma":
            if args.roma_coarse_res % 14 != 0 or args.roma_upsample_res % 14 != 0:
                raise ValueError(
                    "RoMA resolution must be a multiple of 14. "
                    f"Got coarse={args.roma_coarse_res}, upsample={args.roma_upsample_res}."
                )
            return RoMALocalMatcher(
                device=device,
                variant=args.roma_variant,
                cert_threshold=args.roma_cert_threshold,
                max_samples=args.roma_max_samples,
                score_mode=args.roma_score_mode,
                coarse_res=args.roma_coarse_res,
                upsample_res=args.roma_upsample_res,
            )
        if args.matcher == "orb":
            return ORBLocalMatcher(n_features=args.orb_features, ratio_test=args.orb_ratio_test)
        raise ValueError(f"Unsupported matcher: {args.matcher}")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize local matcher. "
            "If you are offline, ALIKED/LoFTR/RoMA may fail while downloading pretrained checkpoints. "
            "Try `--matcher orb` for smoke tests, or pre-populate $TORCH_HOME/hub/checkpoints."
        ) from exc


def evaluate_scenarios(
    scenarios: list[Scenario],
    db_dataset: SimpleImageDataset,
    pair_score_map: dict[tuple[int, int], float],
) -> pd.DataFrame:
    db_labels = db_dataset.labels_string.astype(str)
    rows = []

    for scenario in scenarios:
        candidate_scores = np.asarray(
            [pair_score_map.get((scenario.query_idx, int(db_idx)), 0.0) for db_idx in scenario.candidate_db_indices],
            dtype=np.float32,
        )
        candidate_scores = np.where(np.isnan(candidate_scores), -np.inf, candidate_scores)
        best_local_idx = int(np.argmax(candidate_scores))
        pred_db_idx = int(scenario.candidate_db_indices[best_local_idx])
        pred_identity = str(db_labels[pred_db_idx])
        hit = int(pred_db_idx == scenario.injected_positive_db_idx)

        gt_pos_local = int(np.where(scenario.candidate_db_indices == scenario.injected_positive_db_idx)[0][0])
        descending = np.argsort(-candidate_scores)
        gt_rank = int(np.where(descending == gt_pos_local)[0][0] + 1)

        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "query_idx": scenario.query_idx,
                "query_image_id": scenario.query_image_id,
                "query_identity": scenario.query_identity,
                "dataset": scenario.dataset_name,
                "trial": scenario.trial,
                "candidate_size": len(scenario.candidate_db_indices),
                "injected_positive_db_idx": scenario.injected_positive_db_idx,
                "pred_db_idx": pred_db_idx,
                "pred_identity": pred_identity,
                "top1_hit": hit,
                "gt_rank_in_candidates": gt_rank,
            }
        )

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame, meta: dict) -> dict:
    summary = dict(meta)
    if len(df) == 0:
        summary["num_scenarios"] = 0
        summary["top1_accuracy"] = None
        summary["per_dataset"] = {}
        return summary

    summary["num_scenarios"] = int(len(df))
    summary["top1_accuracy"] = float(df["top1_hit"].mean())
    summary["avg_gt_rank"] = float(df["gt_rank_in_candidates"].mean())

    per_dataset = (
        df.groupby("dataset")["top1_hit"]
        .agg(top1_accuracy="mean", num_scenarios="count")
        .reset_index()
    )
    summary["per_dataset"] = per_dataset.to_dict(orient="records")
    return summary


def save_outputs(results_df: pd.DataFrame, summary: dict, results_dir: Path, run_prefix: str) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"{run_prefix}_{ts}_detail.csv"
    json_path = results_dir / f"{run_prefix}_{ts}_summary.json"
    results_df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return csv_path, json_path


def print_summary(summary: dict):
    print("\n=== Local Rerank Summary ===")
    print(f"matcher: {summary.get('matcher')}")
    print(f"device: {summary.get('device')}")
    print(f"candidate_size: {summary.get('candidate_size')}")
    print(f"trials_per_query: {summary.get('trials_per_query')}")
    print(f"num_queries_requested: {summary.get('num_queries_requested')}")
    print(f"num_queries_without_positive_in_db: {summary.get('num_queries_without_positive_in_db')}")
    print(f"num_scenarios: {summary.get('num_scenarios')}")
    print(f"top1_accuracy: {summary.get('top1_accuracy')}")
    print(f"avg_gt_rank: {summary.get('avg_gt_rank')}")
    print("per_dataset:")
    for row in summary.get("per_dataset", []):
        print(f"  - {row['dataset']}: top1={row['top1_accuracy']:.4f} ({row['num_scenarios']} scenarios)")


def parse_args():
    parser = argparse.ArgumentParser(description="Local reranker-only experiment for AnimalCLEF2025.")
    parser.add_argument("--root", type=str, default=None, help="AnimalCLEF2025 root containing metadata.csv")
    parser.add_argument("--auto-download", action="store_true", help="Try downloading dataset if root is missing files.")
    parser.add_argument(
        "--matcher",
        type=str,
        default="aliked",
        choices=["aliked", "loftr", "roma", "orb"],
        help="Local matcher backend",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-num-keypoints", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--loftr-pretrained", type=str, default="outdoor", choices=["outdoor", "indoor"])
    parser.add_argument("--roma-variant", type=str, default="outdoor", choices=["outdoor", "indoor"])
    parser.add_argument("--roma-coarse-res", type=int, default=560)
    parser.add_argument("--roma-upsample-res", type=int, default=864)
    parser.add_argument("--roma-cert-threshold", type=float, default=0.5)
    parser.add_argument("--roma-max-samples", type=int, default=1200)
    parser.add_argument("--roma-score-mode", type=str, default="sum_above", choices=["count", "sum", "sum_above"])
    parser.add_argument("--orb-features", type=int, default=1500)
    parser.add_argument("--orb-ratio-test", type=float, default=0.75)

    parser.add_argument("--candidate-size", type=int, default=25)
    parser.add_argument("--trials-per-query", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--dataset-filter", type=str, default=None, help="Comma-separated dataset names filter")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cross-dataset-candidates",
        action="store_true",
        help=(
            "Allow candidate sampling across different datasets/species. "
            "Default behavior is same-dataset-only candidate sampling."
        ),
    )
    parser.add_argument(
        "--query-source",
        type=str,
        default="auto",
        choices=["auto", "metadata_query", "db_self_eval"],
        help=(
            "Source of query set. `metadata_query` uses split=query rows. "
            "`db_self_eval` samples queries from database identities. "
            "`auto` falls back to db_self_eval when query/db identity overlap is zero."
        ),
    )
    parser.add_argument(
        "--db-self-eval-per-id",
        type=int,
        default=1,
        help="When query-source is db_self_eval, number of query images sampled per identity.",
    )

    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Use generated synthetic data for smoke testing (no AnimalCLEF files required).",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default="experiments/local_rerank/synthetic_data",
        help="Directory used to store synthetic images.",
    )
    parser.add_argument("--synthetic-identities", type=int, default=8)
    parser.add_argument("--synthetic-db-per-identity", type=int, default=3)

    parser.add_argument("--results-dir", type=str, default="experiments/local_rerank/results")
    parser.add_argument("--run-prefix", type=str, default="local_rerank")
    parser.add_argument(
        "--visualize-per-dataset",
        type=int,
        default=0,
        help=(
            "If > 0, save match-point visualizations per dataset with "
            "up to N good and N bad samples."
        ),
    )
    parser.add_argument(
        "--visualize-max-matches",
        type=int,
        default=120,
        help="Maximum number of correspondence lines to draw per visualization image.",
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default=None,
        help="Visualization output directory. Default: <results-dir>/visualizations/<run-prefix>",
    )
    parser.add_argument("--no-check-paths", action="store_true", help="Skip checking that metadata image paths exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"[Info] matcher={args.matcher}, device={device}")

    if args.synthetic_smoke:
        query_dataset, db_dataset = create_synthetic_datasets(
            out_dir=Path(args.synthetic_dir),
            n_identities=args.synthetic_identities,
            db_per_identity=args.synthetic_db_per_identity,
        )
        print(
            f"[Info] Synthetic datasets ready: query={len(query_dataset)}, "
            f"database={len(db_dataset)}"
        )
    else:
        if args.root is None:
            raise ValueError("--root is required unless --synthetic-smoke is used.")
        query_dataset, db_dataset = load_real_dataset(
            root=Path(args.root),
            auto_download=args.auto_download,
            check_paths=not args.no_check_paths,
        )
        overlap = identity_overlap_count(query_dataset, db_dataset)
        print(
            f"[Info] Loaded real dataset: query={len(query_dataset)}, "
            f"database={len(db_dataset)}, identity_overlap={overlap}"
        )

        if args.query_source == "db_self_eval":
            try:
                query_dataset, db_dataset = build_db_self_eval_split(
                    db_dataset=db_dataset,
                    per_identity_queries=args.db_self_eval_per_id,
                    seed=args.seed,
                )
            except Exception as exc:
                print(f"[Error] Failed to create db_self_eval split: {exc}")
                return 1
            print(
                f"[Info] query_source=db_self_eval -> query={len(query_dataset)}, "
                f"database={len(db_dataset)}"
            )
        elif args.query_source == "auto" and overlap == 0:
            print(
                "[Warn] Query/DB identity overlap is zero. "
                "Falling back to db_self_eval mode automatically."
            )
            try:
                query_dataset, db_dataset = build_db_self_eval_split(
                    db_dataset=db_dataset,
                    per_identity_queries=args.db_self_eval_per_id,
                    seed=args.seed,
                )
            except Exception as exc:
                print(
                    "[Error] Auto fallback to db_self_eval failed. "
                    f"Reason: {exc}"
                )
                return 1
            print(
                f"[Info] auto->db_self_eval -> query={len(query_dataset)}, "
                f"database={len(db_dataset)}"
            )
        else:
            print(f"[Info] query_source={args.query_source}")

    dataset_filter = parse_dataset_filter(args.dataset_filter)
    scenarios, skipped_no_positive = build_scenarios(
        query_dataset=query_dataset,
        db_dataset=db_dataset,
        candidate_size=args.candidate_size,
        trials_per_query=args.trials_per_query,
        max_queries=args.max_queries,
        dataset_filter=dataset_filter,
        seed=args.seed,
        same_dataset_only=not args.cross_dataset_candidates,
    )
    if len(scenarios) == 0:
        overlap_after = identity_overlap_count(query_dataset, db_dataset)
        print(
            "[Error] No scenarios were generated. "
            "Check labels/query source/filter settings."
        )
        print(
            f"[Error] query={len(query_dataset)}, database={len(db_dataset)}, "
            f"identity_overlap={overlap_after}, dataset_filter={args.dataset_filter}"
        )
        print(
            "[Hint] Try one of: "
            "`--query-source db_self_eval`, "
            "`--db-self-eval-per-id 1`, "
            "or adjust `--dataset-filter`."
        )
        return 1

    print(f"[Info] same_dataset_only={not args.cross_dataset_candidates}")
    print(f"[Info] Generated scenarios: {len(scenarios)}")

    try:
        matcher = build_matcher(args, device=device)
        matcher.prepare(query_dataset, db_dataset)
    except Exception as exc:
        print(f"[Error] {exc}")
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            print(f"[Error] Root cause: {type(cause).__name__}: {cause}")
        return 1

    unique_pairs = sorted({(s.query_idx, int(db_idx)) for s in scenarios for db_idx in s.candidate_db_indices})
    print(f"[Info] Unique local-matching pairs to score: {len(unique_pairs)}")
    try:
        unique_scores = matcher.score_pairs(unique_pairs)
    except Exception as exc:
        print(f"[Error] Pair scoring failed: {exc}")
        return 1
    pair_score_map = {pair: float(score) for pair, score in zip(unique_pairs, unique_scores)}

    results_df = evaluate_scenarios(scenarios, db_dataset=db_dataset, pair_score_map=pair_score_map)
    summary = summarize_results(
        results_df,
        meta={
            "matcher": args.matcher,
            "device": device,
            "candidate_size": args.candidate_size,
            "trials_per_query": args.trials_per_query,
            "num_queries_requested": args.max_queries,
            "num_queries_without_positive_in_db": skipped_no_positive,
            "seed": args.seed,
            "synthetic_smoke": bool(args.synthetic_smoke),
            "query_source": args.query_source,
            "db_self_eval_per_id": args.db_self_eval_per_id,
            "same_dataset_only": bool(not args.cross_dataset_candidates),
            "visualize_per_dataset": args.visualize_per_dataset,
            "visualize_max_matches": args.visualize_max_matches,
            "roma_variant": args.roma_variant if args.matcher == "roma" else None,
            "roma_cert_threshold": args.roma_cert_threshold if args.matcher == "roma" else None,
            "roma_score_mode": args.roma_score_mode if args.matcher == "roma" else None,
            "roma_coarse_res": args.roma_coarse_res if args.matcher == "roma" else None,
            "roma_upsample_res": args.roma_upsample_res if args.matcher == "roma" else None,
        },
    )
    print_summary(summary)

    if args.visualize_per_dataset > 0:
        if args.visualize_dir is None:
            vis_dir = Path(args.results_dir) / "visualizations" / args.run_prefix
        else:
            vis_dir = Path(args.visualize_dir)
        saved_paths = generate_match_visualizations(
            results_df=results_df,
            query_dataset=query_dataset,
            db_dataset=db_dataset,
            matcher=matcher,
            matcher_name=args.matcher,
            output_dir=vis_dir,
            per_dataset=args.visualize_per_dataset,
            max_matches=args.visualize_max_matches,
        )
        summary["num_visualizations"] = int(len(saved_paths))
        summary["visualization_dir"] = str(vis_dir)
        print(f"[Saved] Visualizations: {len(saved_paths)} files -> {vis_dir}")
    else:
        summary["num_visualizations"] = 0

    csv_path, json_path = save_outputs(
        results_df,
        summary,
        results_dir=Path(args.results_dir),
        run_prefix=args.run_prefix,
    )
    print(f"\n[Saved] Detail CSV: {csv_path}")
    print(f"[Saved] Summary JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
