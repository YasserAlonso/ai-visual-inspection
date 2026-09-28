"""BK-tree radius search and connected components for perceptual duplicate groups."""

from collections import defaultdict
from itertools import combinations

import pandas as pd

PAIR_COLUMNS = ["group_id", "source", "target", "source_split", "target_split", "distance"]


class HammingIndex:
    """BK-tree of distinct 64-bit hashes; worst-case radius queries remain linear."""

    def __init__(self) -> None:
        self.nodes: list[tuple[int, dict[int, int], list[int]]] = []

    def add(self, value: int, identifier: int) -> None:
        if not self.nodes:
            self.nodes.append((value, {}, [identifier]))
            return
        position = 0
        while True:
            existing, children, identifiers = self.nodes[position]
            distance = (value ^ existing).bit_count()
            if distance == 0:
                identifiers.append(identifier)
                return
            if distance not in children:
                children[distance] = len(self.nodes)
                self.nodes.append((value, {}, [identifier]))
                return
            position = children[distance]

    def query(self, value: int, radius: int) -> list[tuple[int, int]]:
        matches = []
        pending = [0] if self.nodes else []
        while pending:
            existing, children, identifiers = self.nodes[pending.pop()]
            distance = (value ^ existing).bit_count()
            if distance <= radius:
                matches.extend((identifier, distance) for identifier in identifiers)
            pending.extend(
                child
                for edge, child in children.items()
                if distance - radius <= edge <= distance + radius
            )
        return matches


def _pair(a: dict, b: dict, group: str, distance: int) -> dict:
    return dict(
        group_id=group,
        source=a["path"],
        target=b["path"],
        source_split=a["split"],
        target_split=b["split"],
        distance=distance,
    )


def exact_duplicates(images: pd.DataFrame) -> pd.DataFrame:
    buckets = defaultdict(list)
    for record in images[images.status == "valid"].to_dict("records"):
        buckets[record["sha256"]].append(record)
    pairs = []
    for records in buckets.values():
        if len(records) > 1:
            group = records[0]["sha256"]
            pairs.extend(_pair(a, b, group, 0) for a, b in combinations(records, 2))
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def near_duplicates(images: pd.DataFrame, threshold: int) -> pd.DataFrame:
    records = images[images.status == "valid"].to_dict("records")
    index = HammingIndex()
    parents = list(range(len(records)))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    edges = []
    for i, record in enumerate(records):
        value = int(record["phash"], 16)
        for j, distance in index.query(value, threshold):
            if record["sha256"] == records[j]["sha256"]:
                continue
            parents[find(i)] = find(j)
            edges.append((j, i, distance))
        index.add(value, i)
    return pd.DataFrame(
        [
            _pair(records[a], records[b], f"near-{find(a):06d}", distance)
            for a, b, distance in edges
        ],
        columns=PAIR_COLUMNS,
    )
