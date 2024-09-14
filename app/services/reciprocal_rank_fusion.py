from typing import List, Dict
from app.schemas.responses.keyframes import KeyframeWithConfidence
from collections import defaultdict


class ReciporalRankFusionService:
    def __init__(self, k: int = 60):
        self.k = k

    def reciprocal_rank_fusion(
        self,
        clip_keyframes: List[KeyframeWithConfidence],
        object_keyframes: List[KeyframeWithConfidence],
        audio_keyframes: List[KeyframeWithConfidence],
        ocr_keyframes: List[KeyframeWithConfidence],
    ) -> List[KeyframeWithConfidence]:

        def reciprocal_rank(rank: int) -> float:
            return 1 / (self.k + rank)

        # Create dictionaries for faster lookup of confidence scores
        clip_dict = {kf.key: kf.confidence for kf in clip_keyframes}
        object_dict = {kf.key: kf.confidence for kf in object_keyframes}
        audio_dict = {kf.key: kf.confidence for kf in audio_keyframes}
        ocr_dict = {kf.key: kf.confidence for kf in ocr_keyframes}

        # Sort each keyframe list once and store ranks in dictionaries
        def get_rank_dict(keyframes: List[KeyframeWithConfidence]) -> Dict[int, int]:
            sorted_keyframes = sorted(
                keyframes, key=lambda x: x.confidence, reverse=True
            )
            return {kf.key: rank + 1 for rank, kf in enumerate(sorted_keyframes)}

        clip_ranks = get_rank_dict(clip_keyframes)
        object_ranks = get_rank_dict(object_keyframes)
        audio_ranks = get_rank_dict(audio_keyframes)
        ocr_ranks = get_rank_dict(ocr_keyframes)

        # Combine keys from all three lists
        all_keys = (
            set(clip_dict.keys()) | set(object_dict.keys()) | set(audio_dict.keys()) | set(ocr_dict.keys())
        )

        # Calculate reciprocal rank fusion scores
        rrf_score = defaultdict(float)
        for key in all_keys:
            clip_rank = clip_ranks.get(key, len(clip_keyframes) + 1)
            object_rank = object_ranks.get(key, len(object_keyframes) + 1)
            audio_rank = audio_ranks.get(key, len(audio_keyframes) + 1)
            ocr_rank = ocr_ranks.get(key, len(ocr_keyframes) + 1)

            # Sum the reciprocal ranks for each key
            rrf_score[key] = (
                reciprocal_rank(clip_rank)
                + reciprocal_rank(object_rank)
                + reciprocal_rank(audio_rank)
                + reciprocal_rank(ocr_rank)
            )

        # Sort the keys by the computed reciprocal rank fusion score
        sorted_indices = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)

        # Combine the results with corresponding KeyframeWithConfidence objects
        keyframes = clip_keyframes + object_keyframes + audio_keyframes + ocr_keyframes
        all_keyframes = {kf.key: kf for kf in keyframes}

        final_results = []
        for rank, (idx, _) in enumerate(sorted_indices, start=1):
            if idx in all_keyframes:
                keyframe = all_keyframes[idx]
                final_results.append(
                    KeyframeWithConfidence(
                        key=keyframe.key,
                        value=keyframe.value,
                        confidence=rank,  # Rank used as confidence
                        video_id=keyframe.video_id,
                        group_id=keyframe.group_id,
                    )
                )

        return final_results
