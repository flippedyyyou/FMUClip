"""BLIP reward model helpers.

This module wraps a BLIP feature extractor to provide a lightweight reward
model that can rank candidate captions for a batch of images. The helper is
kept small and dependency-free beyond LAVIS so it can be reused inside
``clip_unlearn_label.py`` without cluttering the main training script.
"""

from typing import Iterable, List, Sequence, Tuple

import torch

try:  # LAVIS has a unified helper for loading BLIP checkpoints
    from lavis.models import load_model_and_preprocess
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "BLIP reward model requires LAVIS. Please install LAVIS before using BLIP rewards.") from exc


class BlipRewardSelector:
    """A minimal BLIP-based reward model for caption selection.

    The selector uses the BLIP feature extractor shipped with LAVIS to compute
    cosine similarities between images and a pool of candidate captions. It is
    intentionally stateless aside from the BLIP backbone so that upstream code
    can feed arbitrary batches of images and candidate sentences and retrieve
    top-k captions together with their similarity scores.
    """

    def __init__(
        self,
        device: torch.device,
        *,
        model_name: str = "blip_feature_extractor",
        model_type: str = "base",
        topk: int = 5,
    ) -> None:
        self.device = device
        self.topk = topk
        # The BLIP feature extractor outputs normalized embeddings suitable for
        # similarity search. We keep processors for potential preprocessing,
        # but the calling code usually provides already-normalized image
        # tensors from the CLIP pipeline.
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            model_name, model_type, device=device, is_eval=True
        )
        self.model.eval()

    @torch.no_grad()
    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # Images from the training loop are already tensors; we just ensure they
        # are on the correct device.
        if images.device != self.device:
            images = images.to(self.device)
        feats = self.model.extract_features({"image": images}, mode="image")
        image_embeds = feats["image_embeds"]
        return image_embeds / image_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def _encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        processed = [self.txt_processors["eval"](txt) for txt in texts]
        feats = self.model.extract_features({"text_input": processed}, mode="text")
        text_embeds = feats["text_embeds"]
        return text_embeds / text_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def clipscore_matrix(self, images: torch.Tensor, candidate_texts: Sequence[str]) -> torch.Tensor:
        """Compute BLIP-based similarity matrix.

        Args:
            images: batched image tensor ``[B, 3, H, W]``.
            candidate_texts: a flat list of candidate captions.

        Returns:
            Tensor ``[B, N]`` where ``N=len(candidate_texts)`` containing cosine
            similarities scaled by BLIP's internal projection.
        """

        if not isinstance(candidate_texts, Iterable):
            raise TypeError("candidate_texts must be an iterable of strings")

        image_feats = self._encode_images(images)  # [B, D]
        text_feats = self._encode_texts(list(candidate_texts))  # [N, D]
        return image_feats @ text_feats.t()  # [B, N]

    @torch.no_grad()
    def select_topk(
        self, images: torch.Tensor, candidate_texts: Sequence[str]
    ) -> Tuple[List[List[str]], torch.Tensor]:
        """Return top-k captions and their scores for each image.

        Returns:
            - a nested list with shape ``[B, K]`` of selected captions.
            - a tensor ``[B, K]`` of similarity scores before softmax.
        """

        score_matrix = self.clipscore_matrix(images, candidate_texts)
        topk_scores, topk_indices = torch.topk(score_matrix, k=min(self.topk, score_matrix.size(1)), dim=-1)

        selected: List[List[str]] = []
        for row in topk_indices.tolist():
            selected.append([candidate_texts[idx] for idx in row])

        return selected, topk_scores
