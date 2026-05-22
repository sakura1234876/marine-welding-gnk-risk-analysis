
from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


Node = str
Edge = Tuple[Node, Node]


@dataclass
class PathScore:
    path: List[Node]
    score: float
    avg_weight: float
    min_weight: float


class DiGraph:
    def __init__(self) -> None:
        self.nodes: Set[Node] = set()
        self.out_adj: Dict[Node, Dict[Node, float]] = defaultdict(dict)
        self.in_adj: Dict[Node, Dict[Node, float]] = defaultdict(dict)

    def add_edge(self, source: Node, target: Node, weight: float = 1.0) -> None:
        if not source or not target or source == target:
            return
        self.nodes.add(source)
        self.nodes.add(target)
        self.out_adj[source][target] = self.out_adj[source].get(target, 0.0) + weight
        self.in_adj[target][source] = self.in_adj[target].get(source, 0.0) + weight

    def edge_weight(self, source: Node, target: Node) -> float:
        return self.out_adj.get(source, {}).get(target, 0.0)

    def edges(self) -> Iterable[Tuple[Node, Node, float]]:
        for source, targets in self.out_adj.items():
            for target, weight in targets.items():
                yield source, target, weight

    def weak_neighbors(self, node: Node) -> Set[Node]:
        return set(self.out_adj.get(node, {})) | set(self.in_adj.get(node, {}))


def load_edges(path: Path) -> DiGraph:
    graph = DiGraph()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"source", "target"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Edge CSV must contain at least columns: source,target")
        for row in reader:
            weight_text = (row.get("weight") or "1").strip()
            try:
                weight = float(weight_text)
            except ValueError:
                weight = 1.0
            graph.add_edge(row["source"].strip(), row["target"].strip(), weight)
    return graph


def minmax(values: Dict[Node, float]) -> Dict[Node, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if high == low:
        return {k: 0.0 for k in values}
    return {k: (v - low) / (high - low) for k, v in values.items()}


def weighted_degrees(graph: DiGraph) -> Tuple[Dict[Node, float], Dict[Node, float], Dict[Node, float]]:
    out_degree = {n: sum(graph.out_adj.get(n, {}).values()) for n in graph.nodes}
    in_degree = {n: sum(graph.in_adj.get(n, {}).values()) for n in graph.nodes}
    total_degree = {n: out_degree[n] + in_degree[n] for n in graph.nodes}
    return out_degree, in_degree, total_degree


def weak_clustering(graph: DiGraph) -> Dict[Node, float]:
    """Undirected local clustering over weak neighbors."""
    undirected_edges = {tuple(sorted((s, t))) for s, t, _ in graph.edges()}
    result: Dict[Node, float] = {}
    for node in graph.nodes:
        neighbors = list(graph.weak_neighbors(node))
        k = len(neighbors)
        if k < 2:
            result[node] = 0.0
            continue
        links = 0
        for i in range(k):
            for j in range(i + 1, k):
                if tuple(sorted((neighbors[i], neighbors[j]))) in undirected_edges:
                    links += 1
        result[node] = 2.0 * links / (k * (k - 1))
    return result


def shortest_distances(graph: DiGraph, start: Node) -> Dict[Node, int]:
    distances = {start: 0}
    queue: deque[Node] = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in graph.out_adj.get(node, {}):
            if nxt not in distances:
                distances[nxt] = distances[node] + 1
                queue.append(nxt)
    return distances


def closeness(graph: DiGraph) -> Dict[Node, float]:
    result: Dict[Node, float] = {}
    n = len(graph.nodes)
    for node in graph.nodes:
        distances = shortest_distances(graph, node)
        total = sum(d for target, d in distances.items() if target != node)
        reachable = len(distances) - 1
        if total == 0 or n <= 1:
            result[node] = 0.0
        else:
            result[node] = (reachable / total) * (reachable / (n - 1))
    return result


def betweenness(graph: DiGraph) -> Dict[Node, float]:
    """Brandes betweenness centrality for an unweighted directed graph."""
    cb = {v: 0.0 for v in graph.nodes}
    for source in graph.nodes:
        stack: List[Node] = []
        predecessors: Dict[Node, List[Node]] = {w: [] for w in graph.nodes}
        sigma = dict.fromkeys(graph.nodes, 0.0)
        distance = dict.fromkeys(graph.nodes, -1)
        sigma[source] = 1.0
        distance[source] = 0
        queue: deque[Node] = deque([source])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.out_adj.get(v, {}):
                if distance[w] < 0:
                    queue.append(w)
                    distance[w] = distance[v] + 1
                if distance[w] == distance[v] + 1:
                    sigma[w] += sigma[v]
                    predecessors[w].append(v)

        dependency = dict.fromkeys(graph.nodes, 0.0)
        while stack:
            w = stack.pop()
            for v in predecessors[w]:
                if sigma[w]:
                    dependency[v] += (sigma[v] / sigma[w]) * (1.0 + dependency[w])
            if w != source:
                cb[w] += dependency[w]

    n = len(graph.nodes)
    if n > 2:
        scale = 1.0 / ((n - 1) * (n - 2))
        cb = {k: v * scale for k, v in cb.items()}
    return cb


def centrality_table(graph: DiGraph) -> List[Dict[str, float | str]]:
    out_d, in_d, total_d = weighted_degrees(graph)
    cluster = weak_clustering(graph)
    between = betweenness(graph)
    close = closeness(graph)

    metrics = {
        "out_degree": out_d,
        "in_degree": in_d,
        "total_degree": total_d,
        "clustering": cluster,
        "betweenness": between,
        "closeness": close,
    }
    normalized = {name: minmax(values) for name, values in metrics.items()}

    rows = []
    for node in sorted(graph.nodes):
        avg_importance = sum(normalized[name][node] for name in metrics) / len(metrics)
        rows.append(
            {
                "node": node,
                "out_degree": out_d[node],
                "in_degree": in_d[node],
                "total_degree": total_d[node],
                "clustering": cluster[node],
                "betweenness": between[node],
                "closeness": close[node],
                "avg_importance": avg_importance,
            }
        )
    rows.sort(key=lambda r: float(r["avg_importance"]), reverse=True)
    return rows


def all_simple_paths(
    graph: DiGraph,
    start: Node,
    end: Node,
    path_edges: int,
    key_edges: Set[Edge],
) -> Iterable[List[Node]]:
    """Yield only simple directed paths with exactly path_edges edges."""
    stack: List[Tuple[Node, List[Node]]] = [(start, [start])]
    while stack:
        node, path = stack.pop()
        if len(path) - 1 >= path_edges:
            continue
        for nxt in graph.out_adj.get(node, {}):
            if nxt in path:
                continue
            if key_edges and (node, nxt) not in key_edges:
                continue
            new_path = path + [nxt]
            new_edges = len(new_path) - 1
            if nxt == end and new_edges == path_edges:
                yield new_path
            elif nxt != end and new_edges < path_edges:
                stack.append((nxt, new_path))


def rank_key_paths(
    graph: DiGraph,
    centrality_rows: List[Dict[str, float | str]],
    core_node_count: int,
    key_edge_count: int,
    path_edges: int,
) -> List[PathScore]:
    core_nodes = [str(row["node"]) for row in centrality_rows[:core_node_count]]
    edges_by_weight = sorted(graph.edges(), key=lambda e: e[2], reverse=True)
    key_edges = {(s, t) for s, t, _ in edges_by_weight[:key_edge_count]}

    seen: Set[Tuple[Node, ...]] = set()
    scored: List[PathScore] = []
    for start in core_nodes:
        for end in core_nodes:
            if start == end:
                continue
            for path in all_simple_paths(graph, start, end, path_edges, key_edges):
                path_key = tuple(path)
                if path_key in seen:
                    continue
                seen.add(path_key)
                weights = [graph.edge_weight(path[i], path[i + 1]) for i in range(len(path) - 1)]
                scored.append(
                    PathScore(
                        path=path,
                        score=sum(weights),
                        avg_weight=sum(weights) / len(weights),
                        min_weight=min(weights),
                    )
                )
    scored.sort(key=lambda x: (x.score, x.avg_weight, x.min_weight), reverse=True)
    return scored


def write_centrality(path: Path, rows: List[Dict[str, float | str]]) -> None:
    fields = [
        "node",
        "out_degree",
        "in_degree",
        "total_degree",
        "clustering",
        "betweenness",
        "closeness",
        "avg_importance",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_paths(path: Path, paths: List[PathScore], top_k: int) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rank", "path", "length", "cumulative_weight", "average_weight", "minimum_weight"],
        )
        writer.writeheader()
        for rank, item in enumerate(paths[:top_k], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "path": "->".join(item.path),
                    "length": len(item.path) - 1,
                    "cumulative_weight": item.score,
                    "average_weight": item.avg_weight,
                    "minimum_weight": item.min_weight,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", required=True, type=Path, help="CSV with columns source,target[,weight,relation]")
    parser.add_argument("--out-dir", default=Path("."), type=Path)
    parser.add_argument("--core-node-count", default=10, type=int)
    parser.add_argument("--key-edge-count", default=30, type=int)
    parser.add_argument(
        "--path-edges",
        "--max-edges",
        dest="path_edges",
        default=2,
        type=int,
        help="Exact number of directed edges in each path; use 2 for two-hop paths",
    )
    parser.add_argument("--top-k", default=15, type=int)
    args = parser.parse_args()

    graph = load_edges(args.edges)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    centrality_rows = centrality_table(graph)
    paths = rank_key_paths(
        graph,
        centrality_rows,
        core_node_count=args.core_node_count,
        key_edge_count=args.key_edge_count,
        path_edges=args.path_edges,
    )

    write_centrality(args.out_dir / "baseline_node_centrality.csv", centrality_rows)
    write_paths(args.out_dir / "baseline_key_paths.csv", paths, args.top_k)

    print(f"nodes={len(graph.nodes)} edges={sum(1 for _ in graph.edges())}")
    print(f"wrote {args.out_dir / 'baseline_node_centrality.csv'}")
    print(f"wrote {args.out_dir / 'baseline_key_paths.csv'}")


if __name__ == "__main__":
    main()
