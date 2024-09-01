from typing import List, Dict
from app.schemas.responses.keyframes import KeyframeWithConfidence
from collections import defaultdict


class ReciporalRankFusionService:
    def __init__(self, k: int = 60):
        self.k = k

    def normalize_scores(self, scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}

        min_score = min(scores.values())
        max_score = max(scores.values())

        if max_score == min_score:
            return {index: 1.0 for index in scores}

        return {
            index: (score - min_score) / (max_score - min_score)
            for index, score in scores.items()
        }

    def reciprocal_rank_fusion(
        self,
        clip_keyframes: List[KeyframeWithConfidence],
        object_keyframes: List[KeyframeWithConfidence],
    ) -> List[KeyframeWithConfidence]:
        def reciprocal_rank(score: float) -> float:
            return 1 / (self.k + (1 - score))

        # Create dictionaries for faster lookup
        clip_dict = {kf.key: kf.confidence for kf in clip_keyframes}
        object_dict = {kf.key: kf.confidence for kf in object_keyframes}

        print(f"Clip results: {len(clip_dict)}")
        print(f"Object results: {len(object_dict)}")

        # Normalize scores
        clip_scores_norm = self.normalize_scores(clip_dict)
        object_scores_norm = self.normalize_scores(object_dict)

        # Calculate reciprocal ranks
        combined_weights = defaultdict(float)
        for index, score in clip_scores_norm.items():
            combined_weights[index] += reciprocal_rank(score)
        for index, score in object_scores_norm.items():
            combined_weights[index] += reciprocal_rank(score)

        # Sort combined weights
        sorted_indices = sorted(
            combined_weights, key=combined_weights.get, reverse=True
        )
        print(f"Combined results: {len(sorted_indices)}")

        # Create a dictionary to map keyframe keys to KeyframeWithConfidence objects
        all_keyframes = {kf.key: kf for kf in clip_keyframes + object_keyframes}

        # Combine the results with corresponding KeyframeWithConfidence objects
        final_results = []
        for rank, idx in enumerate(sorted_indices, start=1):
            if idx in all_keyframes:
                keyframe = all_keyframes[idx]
                final_results.append(
                    KeyframeWithConfidence(
                        key=keyframe.key,
                        value=keyframe.value,
                        confidence=rank,
                        video_id=keyframe.video_id,
                        group_id=keyframe.group_id,
                    )
                )

        return final_results
