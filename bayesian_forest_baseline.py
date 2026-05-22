from __future__ import annotations

import argparse
import itertools
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


Edge = Tuple[str, str]
Path3 = Tuple[str, str, str]
PathN = Tuple[str, ...]


CATEGORY_MAP = {
    "技能水平": "H1",
    "责任心态度": "H2",
    "操作规范": "H3",
    "监督检查": "MG1",
    "制度流程": "MG2",
    "培训交底": "MG3",
    "材料质量": "MA1",
    "焊材管理": "MA2",
    "存储条件": "MA3",
    "工艺设计": "TE1",
    "工艺执行": "TE2",
    "技术标准": "TE3",
    "天气条件": "EN1",
    "施工环境": "EN2",
    "设备状态": "EQ1",
    "工具使用": "EQ2",
    "性能指标": "EQ3",
}

CODE_TO_NAME = {code: name for name, code in CATEGORY_MAP.items()}
VALID_CODES = set(CATEGORY_MAP.values())


@dataclass(frozen=True)
class ScoredDag:
    edges: frozenset[Edge]
    score: float
    weight: float


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def normalize_factor_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        col: CATEGORY_MAP[col]
        for col in df.columns
        if col in CATEGORY_MAP
    }
    df = df.rename(columns=rename_map)
    factor_cols = [col for col in df.columns if col in VALID_CODES]
    if not factor_cols:
        raise ValueError(
            "No factor columns found. Expected codes like H1/EQ3 or Chinese factor names."
        )
    matrix = df[factor_cols].copy()
    for col in matrix.columns:
        matrix[col] = pd.to_numeric(matrix[col], errors="coerce").fillna(0).astype(int)
        matrix[col] = matrix[col].clip(lower=0, upper=1)
    return matrix


def mutual_information_binary(x: np.ndarray, y: np.ndarray) -> float:
    total = len(x)
    result = 0.0
    for xv in (0, 1):
        for yv in (0, 1):
            joint = np.mean((x == xv) & (y == yv))
            if joint <= 0:
                continue
            px = np.mean(x == xv)
            py = np.mean(y == yv)
            result += joint * math.log(joint / (px * py))
    return float(result)


def build_mi_table(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(matrix.columns, 2):
        mi = mutual_information_binary(matrix[a].to_numpy(), matrix[b].to_numpy())
        rows.append({"source": a, "target": b, "mi": mi})
    return pd.DataFrame(rows)


def prefix(code: str) -> str:
    return "".join(ch for ch in code if ch.isalpha())


def load_local_edges(path: Path | None) -> Counter[Edge]:
    if path is None:
        return Counter()
    df = read_table(path)
    source_candidates = ["source", "from", "from_factor", "起点"]
    target_candidates = ["target", "to", "to_factor", "终点"]
    source_col = next((c for c in source_candidates if c in df.columns), None)
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if source_col is None or target_col is None:
        raise ValueError("Local edge file must contain source/target columns.")

    counts: Counter[Edge] = Counter()
    for _, row in df.iterrows():
        source = CATEGORY_MAP.get(str(row[source_col]).strip(), str(row[source_col]).strip())
        target = CATEGORY_MAP.get(str(row[target_col]).strip(), str(row[target_col]).strip())
        if source in VALID_CODES and target in VALID_CODES and source != target:
            counts[(source, target)] += 1
    return counts


def default_allowed_direction(a: str, b: str) -> Set[Edge]:
    """
    Weak prior: forbid obviously implausible reversals while avoiding a rigid total order.
    Same-category pairs are allowed both ways because the current data do not establish order.
    """
    rank = {"EN": 0, "MA": 0, "EQ": 0, "MG": 0, "TE": 1, "H": 2}
    pa, pb = prefix(a), prefix(b)
    ra, rb = rank.get(pa, 1), rank.get(pb, 1)
    if ra < rb:
        return {(a, b)}
    if rb < ra:
        return {(b, a)}
    return {(a, b), (b, a)}


def sort_rank(code: str) -> int:
    """
    Welding-adapted partial order for the paper's sort() concept.
    Lower ranks are more plausible antecedents; same-rank nodes may compete.
    """
    return {"EN": 0, "MA": 0, "EQ": 0, "MG": 0, "TE": 1, "H": 2}.get(prefix(code), 1)


def dependency_level(mi: float, mmi: float, pd_alpha: float, ppd_alpha: float) -> str:
    if mmi <= 0:
        return "NONE"
    if mi > pd_alpha * mmi:
        return "PD"
    if mi > ppd_alpha * mmi:
        return "PPD"
    return "NONE"


def build_dependency_tables(
    matrix: pd.DataFrame,
    mi_df: pd.DataFrame,
    local_edges: Counter[Edge],
    pd_alpha: float,
    ppd_alpha: float,
) -> Tuple[pd.DataFrame, Set[Edge], Set[Edge]]:
    max_mi = defaultdict(float)
    for _, row in mi_df.iterrows():
        max_mi[row["source"]] = max(max_mi[row["source"]], row["mi"])
        max_mi[row["target"]] = max(max_mi[row["target"]], row["mi"])

    rows = []
    pd_edges: Set[Edge] = set()
    ppd_edges: Set[Edge] = set()
    for _, row in mi_df.iterrows():
        a, b, mi = row["source"], row["target"], row["mi"]
        allowed = default_allowed_direction(a, b)
        if local_edges[(a, b)] > local_edges[(b, a)]:
            allowed = {(a, b)}
        elif local_edges[(b, a)] > local_edges[(a, b)]:
            allowed = {(b, a)}

        for source, target in allowed:
            level = dependency_level(
                mi=mi,
                mmi=max_mi[source],
                pd_alpha=pd_alpha,
                ppd_alpha=ppd_alpha,
            )
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "mi": mi,
                    "mmi_source": max_mi[source],
                    "dependency_level": level,
                }
            )
            if level == "PD":
                pd_edges.add((source, target))
                ppd_edges.add((source, target))
            elif level == "PPD":
                ppd_edges.add((source, target))
    return pd.DataFrame(rows), pd_edges, ppd_edges


def parent_config_counts(matrix: pd.DataFrame, child: str, parents: Sequence[str]) -> Dict[Tuple[int, ...], Counter[int]]:
    counts: Dict[Tuple[int, ...], Counter[int]] = defaultdict(Counter)
    if not parents:
        for value in matrix[child]:
            counts[()][int(value)] += 1
        return counts

    cols = list(parents) + [child]
    for row in matrix[cols].itertuples(index=False, name=None):
        parent_values = tuple(int(v) for v in row[:-1])
        child_value = int(row[-1])
        counts[parent_values][child_value] += 1
    return counts


def local_bdeu_score(matrix: pd.DataFrame, child: str, parents: Sequence[str], equivalent_sample_size: float = 1.0) -> float:
    counts = parent_config_counts(matrix, child, parents)
    q = max(1, 2 ** len(parents))
    r = 2
    alpha_ij = equivalent_sample_size / q
    alpha_ijk = equivalent_sample_size / (q * r)
    score = 0.0
    for config in itertools.product((0, 1), repeat=len(parents)) if parents else [()]:
        child_counts = counts.get(config, Counter())
        nij = sum(child_counts.values())
        score += math.lgamma(alpha_ij) - math.lgamma(alpha_ij + nij)
        for value in (0, 1):
            score += math.lgamma(alpha_ijk + child_counts.get(value, 0)) - math.lgamma(alpha_ijk)
    return score


def dag_score(matrix: pd.DataFrame, nodes: Sequence[str], edges: Set[Edge]) -> float:
    parents = defaultdict(list)
    for source, target in edges:
        parents[target].append(source)
    return sum(local_bdeu_score(matrix, node, tuple(sorted(parents[node]))) for node in nodes)


def is_acyclic(nodes: Sequence[str], edges: Set[Edge]) -> bool:
    indegree = {node: 0 for node in nodes}
    graph = defaultdict(list)
    for source, target in edges:
        graph[source].append(target)
        indegree[target] += 1
    queue = [node for node in nodes if indegree[node] == 0]
    visited = 0
    while queue:
        node = queue.pop()
        visited += 1
        for nxt in graph[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return visited == len(nodes)


def simplified_k2(matrix: pd.DataFrame, pd_edges: Set[Edge]) -> Set[Edge]:
    nodes = list(matrix.columns)
    edges: Set[Edge] = set()
    for child in sorted(nodes, key=lambda node: (sort_rank(node), node)):
        parent_candidates = {
            source
            for source, target in pd_edges
            if target == child and sort_rank(source) <= sort_rank(child)
        }
        parents: Set[str] = set()
        current_score = local_bdeu_score(matrix, child, tuple(sorted(parents)))
        improved = True
        while improved:
            improved = False
            best_parent = None
            best_score = current_score
            for parent in sorted(parent_candidates - parents):
                trial_parents = tuple(sorted(parents | {parent}))
                score = local_bdeu_score(matrix, child, trial_parents)
                if score > best_score + 1e-9:
                    best_score = score
                    best_parent = parent
            if best_parent is not None:
                parents.add(best_parent)
                edges.add((best_parent, child))
                current_score = best_score
                improved = True
    return edges


def weak_components(nodes: Sequence[str], edges: Set[Edge]) -> List[Set[str]]:
    adjacency = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen = set()
    components = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        component = set()
        seen.add(node)
        while stack:
            current = stack.pop()
            component.add(current)
            for nxt in adjacency[current]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        components.append(component)
    return components


def is_bridge(nodes: Sequence[str], edges: Set[Edge], edge: Edge) -> bool:
    before = len(weak_components(nodes, edges))
    after = len(weak_components(nodes, edges - {edge}))
    return after > before


def strongly_connected_components(nodes: Sequence[str], edges: Set[Edge]) -> List[Set[str]]:
    adjacency = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)

    index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components: List[Set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in adjacency[node]:
            if nxt not in indices:
                visit(nxt)
                lowlinks[node] = min(lowlinks[node], lowlinks[nxt])
            elif nxt in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[nxt])
        if lowlinks[node] == indices[node]:
            component = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            visit(node)
    return components


def remove_cycles_with_scc(
    matrix: pd.DataFrame,
    nodes: Sequence[str],
    edges: Set[Edge],
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    rng: random.Random,
) -> Set[Edge]:
    edges = set(edges)
    while not is_acyclic(nodes, edges):
        cyclic_components = [
            component
            for component in strongly_connected_components(nodes, edges)
            if len(component) > 1
        ]
        if not cyclic_components:
            break
        for component in cyclic_components:
            internal_edges = [
                edge for edge in edges if edge[0] in component and edge[1] in component
            ]
            if len(internal_edges) <= 8:
                best_edges = None
                best_score = float("-inf")
                for keep_mask in itertools.product((0, 1), repeat=len(internal_edges)):
                    trial_internal = {
                        edge for edge, keep in zip(internal_edges, keep_mask) if keep
                    }
                    trial = (edges - set(internal_edges)) | trial_internal
                    if is_acyclic(nodes, trial):
                        score = dag_score(matrix, nodes, trial)
                        if score > best_score:
                            best_score = score
                            best_edges = trial
                if best_edges is not None:
                    edges = best_edges
                    continue
            # Larger SCC: preferentially remove weak dependency edges first.
            def deletion_priority(edge: Edge) -> Tuple[int, float]:
                if edge not in ppd_edges:
                    level = 0
                elif edge not in pd_edges:
                    level = 1
                else:
                    level = 2
                return (level, rng.random())

            for edge in sorted(internal_edges, key=deletion_priority):
                edges.remove(edge)
                if is_acyclic(nodes, edges):
                    break
    return edges


def add_edges_for_connectivity(
    nodes: Sequence[str],
    edges: Set[Edge],
    ppd_edges: Set[Edge],
    rng: random.Random,
) -> Set[Edge]:
    edges = set(edges)
    while len(weak_components(nodes, edges)) > 1:
        components = weak_components(nodes, edges)
        current = components[0]
        candidate_edges = [
            edge
            for edge in ppd_edges
            if (edge[0] in current and edge[1] not in current)
            or (edge[1] in current and edge[0] not in current)
        ]
        if not candidate_edges:
            # Fallback: connect components with the direction allowed by the weak prior.
            left = sorted(current)[0]
            right = sorted(components[1])[0]
            candidate_edges = sorted(default_allowed_direction(left, right))
        edge = rng.choice(sorted(candidate_edges))
        edges.add(edge)
    return edges


def mutate_structure(
    nodes: Sequence[str],
    base_edges: Set[Edge],
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    rng: random.Random,
    add_prob: float,
    delete_prob: float,
) -> Set[Edge]:
    edges = set(base_edges)
    for edge in list(edges):
        if rng.random() < delete_prob and not is_bridge(nodes, edges, edge):
            edges.remove(edge)
    for edge in sorted(ppd_edges - edges):
        probability = add_prob if edge in pd_edges else add_prob / 2
        if rng.random() < probability:
            edges.add(edge)
    edges = add_edges_for_connectivity(nodes, edges, ppd_edges, rng)
    return edges


def finalize_structure(
    matrix: pd.DataFrame,
    nodes: Sequence[str],
    edges: Set[Edge],
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    rng: random.Random,
) -> Set[Edge]:
    edges = add_edges_for_connectivity(nodes, edges, ppd_edges, rng)
    edges = remove_cycles_with_scc(matrix, nodes, edges, pd_edges, ppd_edges, rng)
    # Removing a cyclic edge may split the weak graph. Reconnect components after
    # acyclic repair; connecting distinct weak components cannot create a cycle.
    edges = add_edges_for_connectivity(nodes, edges, ppd_edges, rng)
    return edges


def softmax_weights(scores: Sequence[float]) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    shifted = scores - scores.max()
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum()


def build_bayesian_forest(
    matrix: pd.DataFrame,
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    n_dags: int,
    seed: int,
    add_prob: float,
    delete_prob: float,
) -> List[ScoredDag]:
    nodes = list(matrix.columns)
    base_edges = simplified_k2(matrix, pd_edges)
    rng = random.Random(seed)
    base_edges = finalize_structure(matrix, nodes, base_edges, pd_edges, ppd_edges, rng)
    edge_sets = [frozenset(base_edges)]
    while len(edge_sets) < n_dags:
        mutated = frozenset(
            finalize_structure(
                matrix,
                nodes,
                mutate_structure(nodes, base_edges, pd_edges, ppd_edges, rng, add_prob, delete_prob),
                pd_edges,
                ppd_edges,
                rng,
            )
        )
        edge_sets.append(mutated)

    scores = [dag_score(matrix, nodes, set(edges)) for edges in edge_sets]
    weights = softmax_weights(scores)
    return [
        ScoredDag(edges=edges, score=score, weight=float(weight))
        for edges, score, weight in zip(edge_sets, scores, weights)
    ]


def enumerate_paths(nodes: Sequence[str], edges: Set[Edge], factor_count: int) -> Iterable[PathN]:
    adjacency = defaultdict(list)
    for source, target in edges:
        adjacency[source].append(target)

    def dfs(path: Tuple[str, ...]) -> Iterable[PathN]:
        if len(path) == factor_count:
            yield path
            return
        for nxt in adjacency[path[-1]]:
            if nxt not in path:
                yield from dfs(path + (nxt,))

    for node in nodes:
        yield from dfs((node,))


def logmeanexp(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    max_value = values.max()
    return float(max_value + math.log(np.exp(values - max_value).mean()))


def force_path_into_dag(
    nodes: Sequence[str],
    base_edges: Set[Edge],
    ppd_edges: Set[Edge],
    path: PathN,
    rng: random.Random,
    add_prob: float,
    delete_prob: float,
) -> Set[Edge]:
    required = set(zip(path[:-1], path[1:]))
    edges = set(base_edges) | required
    mutable_base = set(edges)
    for edge in list(mutable_base - required):
        if rng.random() < delete_prob:
            mutable_base.remove(edge)
    for edge in sorted(ppd_edges - mutable_base - required):
        if rng.random() < add_prob:
            mutable_base.add(edge)
    return mutable_base


def fit_length_adjustment(
    matrix: pd.DataFrame,
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    forest: Sequence[ScoredDag],
    seed: int,
    add_prob: float,
    delete_prob: float,
    min_length: int,
    max_length: int,
    samples_per_length: int,
) -> Tuple[Dict[int, float], pd.DataFrame]:
    nodes = list(matrix.columns)
    base_edges = simplified_k2(matrix, pd_edges)
    rng = random.Random(seed + 991)
    rows = []
    reference = abs(logmeanexp([dag.score for dag in forest]))
    all_paths = {
        length: list(
            {
                path
                for dag in forest
                for path in enumerate_paths(nodes, set(dag.edges), length)
            }
        )
        for length in range(min_length, max_length + 1)
    }
    for length, paths in all_paths.items():
        if not paths:
            continue
        sampled = [rng.choice(paths) for _ in range(samples_per_length)]
        scores = []
        for idx, path in enumerate(sampled):
            structure = force_path_into_dag(
                nodes,
                base_edges,
                ppd_edges,
                path,
                random.Random(seed + length * 1000 + idx),
                add_prob,
                delete_prob,
            )
            structure = finalize_structure(matrix, nodes, structure, pd_edges, ppd_edges, rng)
            scores.append(abs(dag_score(matrix, nodes, structure)))
        rows.append({"factor_count": length, "avg_abs_log_prob": float(np.mean(scores))})

    fit_df = pd.DataFrame(rows)
    if fit_df.empty or len(fit_df) == 1:
        adjustments = {length: 1.0 for length in range(min_length, max_length + 1)}
        return adjustments, fit_df
    slope, intercept = np.polyfit(fit_df["factor_count"], fit_df["avg_abs_log_prob"], 1)
    target_value = slope * 3 + intercept
    adjustments = {}
    for length in range(min_length, max_length + 1):
        predicted = slope * length + intercept
        adjustments[length] = float(predicted / target_value) if target_value else 1.0
    fit_df["fitted_abs_log_prob"] = slope * fit_df["factor_count"] + intercept
    fit_df["f_n"] = fit_df["fitted_abs_log_prob"] / target_value if target_value else 1.0
    fit_df["reference_abs_log_prob"] = reference
    return adjustments, fit_df


def calculate_rcs(
    matrix: pd.DataFrame,
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    forest: Sequence[ScoredDag],
    path: PathN,
    n_samples: int,
    seed: int,
    add_prob: float,
    delete_prob: float,
    length_adjustment: float = 1.0,
) -> float:
    """
    Approximate the paper's RCS for same-length paths:
    RCS = 1 - log(P(Bs(RC))) / (log(P(Bs)) * f(n)).

    Here structure log-probability is represented by the BDeu score. Because this
    baseline compares only three-factor paths, f(n) defaults to 1.0.
    """
    nodes = list(matrix.columns)
    base_edges = simplified_k2(matrix, pd_edges)
    rng = random.Random(seed)
    constrained_scores = []
    attempts = 0
    while len(constrained_scores) < n_samples and attempts < n_samples * 20:
        attempts += 1
        edges = force_path_into_dag(
            nodes=nodes,
            base_edges=base_edges,
            ppd_edges=ppd_edges,
            path=path,
            rng=rng,
            add_prob=add_prob,
            delete_prob=delete_prob,
        )
        if edges:
            edges = finalize_structure(matrix, nodes, edges, pd_edges, ppd_edges, rng)
            constrained_scores.append(dag_score(matrix, nodes, edges))
    if not constrained_scores:
        return float("-inf")

    reference_log_prob = logmeanexp([dag.score for dag in forest])
    constrained_log_prob = logmeanexp(constrained_scores)
    if reference_log_prob == 0:
        return float("-inf")
    return 1.0 - (constrained_log_prob / (reference_log_prob * length_adjustment))


def rank_paths(
    matrix: pd.DataFrame,
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    forest: Sequence[ScoredDag],
    nodes: Sequence[str],
    factor_count: int,
    length_adjustments: Dict[int, float],
    rcs_samples: int,
    seed: int,
    add_prob: float,
    delete_prob: float,
) -> pd.DataFrame:
    support = defaultdict(float)
    occurrence = Counter()
    for dag in forest:
        paths = set(enumerate_paths(nodes, set(dag.edges), factor_count))
        for path in paths:
            support[path] += dag.weight
            occurrence[path] += 1

    rows = []
    for path, weighted_support in support.items():
        rcs = calculate_rcs(
            matrix=matrix,
            pd_edges=pd_edges,
            ppd_edges=ppd_edges,
            forest=forest,
            path=path,
            n_samples=rcs_samples,
            seed=seed + sum(ord(ch) for ch in "".join(path)),
            add_prob=add_prob,
            delete_prob=delete_prob,
            length_adjustment=length_adjustments.get(factor_count, 1.0),
        )
        rows.append(
            {
                "path": "->".join(path),
                "path_name": "->".join(CODE_TO_NAME[p] for p in path),
                "rcs": rcs,
                "weighted_support": weighted_support,
                "dag_occurrence": occurrence[path],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["rank", "path", "path_name", "weighted_support", "dag_occurrence"])
    result = pd.DataFrame(rows).sort_values(
        ["rcs", "weighted_support", "dag_occurrence", "path"],
        ascending=[False, False, False, True],
    )
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


def write_outputs(
    out_dir: Path,
    matrix: pd.DataFrame,
    mi_df: pd.DataFrame,
    dependency_df: pd.DataFrame,
    pd_edges: Set[Edge],
    ppd_edges: Set[Edge],
    forest: Sequence[ScoredDag],
    ranked_paths: pd.DataFrame,
    length_fit_df: pd.DataFrame,
    top_k: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_dir / "bf_factor_matrix_used.csv", index=False, encoding="utf-8-sig")
    mi_df.to_csv(out_dir / "bf_mutual_information.csv", index=False, encoding="utf-8-sig")
    dependency_df.to_csv(out_dir / "bf_dependency_levels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(sorted(pd_edges), columns=["source", "target"]).to_csv(
        out_dir / "bf_pd_edges.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(sorted(ppd_edges), columns=["source", "target"]).to_csv(
        out_dir / "bf_ppd_edges.csv", index=False, encoding="utf-8-sig"
    )
    length_fit_df.to_csv(out_dir / "bf_length_adjustment_fit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "dag_id": idx + 1,
                "score": dag.score,
                "weight": dag.weight,
                "edges": ";".join(f"{s}->{t}" for s, t in sorted(dag.edges)),
            }
            for idx, dag in enumerate(forest)
        ]
    ).to_csv(out_dir / "bf_dags.csv", index=False, encoding="utf-8-sig")
    ranked_paths.to_csv(out_dir / "bf_all_three_factor_paths.csv", index=False, encoding="utf-8-sig")
    ranked_paths.head(top_k).to_csv(
        out_dir / "bf_top_three_factor_paths.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian Forest baseline for three-factor welding risk path discovery."
    )
    parser.add_argument("--factor-matrix", required=True, type=Path)
    parser.add_argument("--local-edges", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("bayesian_forest_baseline"))
    parser.add_argument("--pd-alpha", type=float, default=0.10)
    parser.add_argument("--ppd-alpha", type=float, default=0.05)
    parser.add_argument("--n-dags", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--add-prob", type=float, default=0.08)
    parser.add_argument("--delete-prob", type=float, default=0.12)
    parser.add_argument("--rcs-samples", type=int, default=60)
    parser.add_argument("--factor-count", type=int, default=3)
    parser.add_argument("--fit-min-length", type=int, default=2)
    parser.add_argument("--fit-max-length", type=int, default=5)
    parser.add_argument("--fit-samples-per-length", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()

    raw_df = read_table(args.factor_matrix)
    matrix = normalize_factor_columns(raw_df)
    local_edges = load_local_edges(args.local_edges)
    mi_df = build_mi_table(matrix)
    dependency_df, pd_edges, ppd_edges = build_dependency_tables(
        matrix=matrix,
        mi_df=mi_df,
        local_edges=local_edges,
        pd_alpha=args.pd_alpha,
        ppd_alpha=args.ppd_alpha,
    )
    forest = build_bayesian_forest(
        matrix=matrix,
        pd_edges=pd_edges,
        ppd_edges=ppd_edges,
        n_dags=args.n_dags,
        seed=args.seed,
        add_prob=args.add_prob,
        delete_prob=args.delete_prob,
    )
    length_adjustments, length_fit_df = fit_length_adjustment(
        matrix=matrix,
        pd_edges=pd_edges,
        ppd_edges=ppd_edges,
        forest=forest,
        seed=args.seed,
        add_prob=args.add_prob,
        delete_prob=args.delete_prob,
        min_length=args.fit_min_length,
        max_length=args.fit_max_length,
        samples_per_length=args.fit_samples_per_length,
    )
    ranked_paths = rank_paths(
        matrix=matrix,
        pd_edges=pd_edges,
        ppd_edges=ppd_edges,
        forest=forest,
        nodes=list(matrix.columns),
        factor_count=args.factor_count,
        length_adjustments=length_adjustments,
        rcs_samples=args.rcs_samples,
        seed=args.seed,
        add_prob=args.add_prob,
        delete_prob=args.delete_prob,
    )
    write_outputs(
        args.out_dir,
        matrix,
        mi_df,
        dependency_df,
        pd_edges,
        ppd_edges,
        forest,
        ranked_paths,
        length_fit_df,
        args.top_k,
    )

    print(f"factors={len(matrix.columns)} samples={len(matrix)}")
    print(f"pd_edges={len(pd_edges)} ppd_edges={len(ppd_edges)} dags={len(forest)} paths={len(ranked_paths)}")
    print(f"wrote {args.out_dir / 'bf_top_three_factor_paths.csv'}")


if __name__ == "__main__":
    main()
