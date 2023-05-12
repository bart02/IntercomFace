from deepface.commons import distance as dst


def find_distance(source_representation, target_representation, distance_metric="cosine"):
    if distance_metric == "cosine":
        return dst.findCosineDistance(source_representation, target_representation)
    if distance_metric == "euclidean":
        return dst.findEuclideanDistance(source_representation, target_representation)
    if distance_metric == "euclidean_l2":
        return dst.findEuclideanDistance(
            dst.l2_normalize(source_representation),
            dst.l2_normalize(target_representation),
        )
    raise ValueError(f"invalid distance metric passes - {distance_metric}")