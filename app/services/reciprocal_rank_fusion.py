from typing import Dict, List, Tuple

from app.schemas.responses.keyframes import KeyframeWithConfidence


class ReciporalRankFusionService:
    def __init__(self):
        pass

    def format_keyframes_results(self, keyframes: List[KeyframeWithConfidence]) -> List[Tuple[int, float]]:
        return [(keyframe.key, keyframe.confidence) for keyframe in keyframes]

    def combine_results(
        self, keyframes: List[KeyframeWithConfidence], results: List[Tuple[int, float]]
    ) -> List[KeyframeWithConfidence]:
        # Create a dictionary to map keyframe keys to KeyframeWithConfidence objects
        keyframe_dict = {keyframe.key: keyframe for keyframe in keyframes}

        # Combine the results with corresponding KeyframeWithConfidence objects
        final = []

        for idx, score in results:
            if idx in keyframe_dict:
                final.append(
                    KeyframeWithConfidence(
                        key=keyframe_dict[idx].key,
                        value=keyframe_dict[idx].value,
                        confidence=score,
                        video_id=keyframe_dict[idx].video_id,
                        group_id=keyframe_dict[idx].group_id,
                    )
                )
            
                # Assign ranks based on the sorted order
        for rank, keyframe in enumerate(final, start=1):
            keyframe.confidence = rank  # Replace the confidence score with the rank

        return final

    def normalize_scores(self, scores: List[Tuple[int, float]]) -> Dict[int, float]:
        if not scores:
            return {}

        min_score = min(score for _, score in scores)
        max_score = max(score for _, score in scores)
        if max_score == min_score:
            return {index: 1.0 for index, _ in scores}
        return {
            index: (score - min_score) / (max_score - min_score)
            for index, score in scores
        }

    def reciprocal_rank_fusion(
        self,
        clip_results: List[Tuple[int, float]],
        object_results: List[Tuple[int, float]],
        k: int = 60,
    ) -> List[Tuple[int, float]]:
        print(f"Clip results: {len(clip_results)}")
        print(f"Object results: {len(object_results)}")

        def reciprocal_rank(score: float) -> float:
            return 1 / (k + (1 - score))

        clip_scores_norm = self.normalize_scores(clip_results)
        object_scores_norm = self.normalize_scores(object_results)

        clip_weights = {
            index: reciprocal_rank(score) for index, score in clip_scores_norm.items()
        }
        object_weights = {
            index: reciprocal_rank(score) for index, score in object_scores_norm.items()
        }

        combined_weights = {}
        all_indices = set(clip_weights.keys()) | set(object_weights.keys())

        for index in all_indices:
            combined_weights[index] = clip_weights.get(index, 0) + object_weights.get(
                index, 0
            )

        combined_weights = sorted(
            combined_weights.items(), key=lambda x: x[1], reverse=True
        )
        print(f"Combined results: {len(combined_weights)}")
        return combined_weights
