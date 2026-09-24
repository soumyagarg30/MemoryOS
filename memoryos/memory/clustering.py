from uuid import UUID


def find_clusters(
    memory_ids: list[UUID], similar_pairs: set[frozenset[UUID]], *, max_size: int = 20
) -> list[list[UUID]]:
    """Deterministic, disjoint clusters whose members are all pairwise similar."""
    if max_size < 3:
        raise ValueError("Clusters must allow at least three memories")
    remaining = sorted(set(memory_ids))
    clusters = []
    while len(remaining) >= 3:
        cluster = [remaining[0]]
        for candidate in remaining[1:]:
            if all(frozenset((candidate, member)) in similar_pairs for member in cluster):
                cluster.append(candidate)
                if len(cluster) == max_size:
                    break
        if len(cluster) >= 3:
            clusters.append(cluster)
            chosen = set(cluster)
            remaining = [candidate for candidate in remaining if candidate not in chosen]
        else:
            remaining.pop(0)
    return clusters
