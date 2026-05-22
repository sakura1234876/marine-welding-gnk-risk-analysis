import argparse
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


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

VALID_CODES = set(CATEGORY_MAP.values())


def read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type: {path}")


def load_node_mapping_cache(path):
    if not path:
        return {}

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Node mapping cache not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_code(value, node_mapping=None):
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text in VALID_CODES:
        return text

    if node_mapping and text in node_mapping:
        mapped = node_mapping[text]
        if isinstance(mapped, dict):
            mapped_code = mapped.get("secondary_code") or mapped.get("code")
        else:
            mapped_code = mapped

        if mapped_code in VALID_CODES:
            return mapped_code

    return CATEGORY_MAP.get(text, "")


def split_factor_codes(value):
    text = str(value).strip()
    for sep in [";", "，", "、", " "]:
        text = text.replace(sep, ",")

    codes = []
    for item in text.split(","):
        code = normalize_code(item)
        if code:
            codes.append(code)

    return codes


def make_factor_key(codes):
    return ";".join(sorted(codes))


def load_mi_candidates(mi_path, min_factor_num=2, max_factor_num=3):
    df = read_table(mi_path)

    if "Factors" not in df.columns or "T_Value" not in df.columns:
        raise ValueError(
            "The MI file must contain columns named 'Factors' and 'T_Value'."
        )

    rows = []
    skipped_rows = []

    for idx, row in df.iterrows():
        codes = split_factor_codes(row["Factors"])
        status = "Success"

        if not (min_factor_num <= len(codes) <= max_factor_num):
            status = f"Skipped_factor_number_{len(codes)}"

        if status == "Success":
            rows.append(
                {
                    "row_index": idx,
                    "type": f"{len(codes)}-factor",
                    "factors_raw": row["Factors"],
                    "factor_codes": ";".join(codes),
                    "factor_key": make_factor_key(codes),
                    "T_Value": float(row["T_Value"]),
                }
            )
        else:
            skipped_rows.append(
                {
                    "row_index": idx,
                    "Factors": row["Factors"],
                    "T_Value": row["T_Value"],
                    "parsed_codes": ";".join(codes),
                    "status": status,
                }
            )

    candidate_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if candidate_df.empty:
        raise ValueError("No valid MI candidates were parsed from the MI file.")

    candidate_df = (
        candidate_df.sort_values("T_Value", ascending=False)
        .drop_duplicates(subset=["factor_key"], keep="first")
        .reset_index(drop=True)
    )

    return candidate_df, skipped_df


def pick_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_relation_edges(relation_path, node_mapping=None):
    df = read_table(relation_path)

    source_col = pick_column(
        df,
        ["source_code", "startnode", "source", "Source", "start_code", "source_node"],
    )
    target_col = pick_column(
        df,
        ["target_code", "endnode", "target", "Target", "end_code", "target_node"],
    )
    count_col = pick_column(df, ["count", "Count", "weight", "Weight", "频次"])

    if not source_col or not target_col:
        raise ValueError(
            "The relation file must contain source/target columns, such as "
            "'source_code,target_code,count' or 'source_node,target_node,count'."
        )

    if count_col is None:
        df["count"] = 1
        count_col = "count"

    rows = []
    skipped_rows = []

    for idx, row in df.iterrows():
        source_code = normalize_code(row[source_col], node_mapping=node_mapping)
        target_code = normalize_code(row[target_col], node_mapping=node_mapping)

        if not source_code or not target_code:
            skipped_rows.append(
                {
                    "row_index": idx,
                    "source_raw": row[source_col],
                    "target_raw": row[target_col],
                    "count": row[count_col],
                    "status": "Skipped_unmapped_node",
                }
            )
            continue

        rows.append(
            {
                "source_code": source_code,
                "target_code": target_code,
                "count": int(row[count_col]),
            }
        )

    edge_df = pd.DataFrame(rows)
    skipped_df = pd.DataFrame(skipped_rows)

    if edge_df.empty:
        raise ValueError(
            "No valid relation edges were parsed. Please provide coded relation "
            "edges or relation nodes that match the built-in factor names."
        )

    edge_df = (
        edge_df.groupby(["source_code", "target_code"], as_index=False)["count"]
        .sum()
        .reset_index(drop=True)
    )

    return edge_df, skipped_df


def build_direct_transition_prob(edge_df):
    freq_dict = defaultdict(lambda: defaultdict(int))

    for _, row in edge_df.iterrows():
        freq_dict[row["source_code"]][row["target_code"]] += int(row["count"])

    direct_prob = {}
    for source, targets in freq_dict.items():
        total = sum(targets.values())
        direct_prob[source] = {
            target: count / total
            for target, count in targets.items()
            if total > 0
        }

    return direct_prob


def calculate_ordered_path_probability(direct_prob, ordered_path):
    prob = 1.0

    for i in range(len(ordered_path) - 1):
        current_node = ordered_path[i]
        next_node = ordered_path[i + 1]
        edge_prob = direct_prob.get(current_node, {}).get(next_node, 0.0)

        if edge_prob <= 0:
            return 0.0

        prob *= edge_prob

    return float(prob)


def enumerate_directed_paths_for_factor_set(direct_prob, factor_codes):
    path_rows = []

    for perm in itertools.permutations(factor_codes):
        prob = calculate_ordered_path_probability(direct_prob, perm)

        if prob > 0:
            path_rows.append(
                {
                    "path": "->".join(perm),
                    "path_probability": float(prob),
                }
            )

    return sorted(
        path_rows,
        key=lambda item: item["path_probability"],
        reverse=True,
    )


def aggregate_topk_average_path_probability(path_rows, k):
    if not path_rows:
        return {
            "transition_probability": 0.0,
            "effective_k": 0,
            "max_path": "",
            "max_path_probability": 0.0,
            "topk_paths": "",
            "topk_path_probabilities": "",
        }

    selected = path_rows[: min(int(k), len(path_rows))]
    probs = [item["path_probability"] for item in selected]

    return {
        "transition_probability": float(np.mean(probs)),
        "effective_k": len(selected),
        "max_path": path_rows[0]["path"],
        "max_path_probability": float(path_rows[0]["path_probability"]),
        "topk_paths": " | ".join(item["path"] for item in selected),
        "topk_path_probabilities": ";".join(
            f"{item['path_probability']:.10g}" for item in selected
        ),
    }


def compute_transition_probabilities(edge_df, mi_candidate_df, k_values):
    direct_prob = build_direct_transition_prob(edge_df)
    rows = []

    for _, row in mi_candidate_df.iterrows():
        factor_codes = str(row["factor_codes"]).split(";")
        path_rows = enumerate_directed_paths_for_factor_set(
            direct_prob=direct_prob,
            factor_codes=factor_codes,
        )

        for k in k_values:
            agg = aggregate_topk_average_path_probability(path_rows, k)
            rows.append(
                {
                    "aggregation_rule": "topk_average",
                    "aggregation_k": int(k),
                    "type": row["type"],
                    "factors_raw": row["factors_raw"],
                    "factor_codes": row["factor_codes"],
                    "factor_key": row["factor_key"],
                    "T_Value": float(row["T_Value"]),
                    "candidate_path_count": len(path_rows),
                    "effective_k": agg["effective_k"],
                    "max_path": agg["max_path"],
                    "max_path_probability": agg["max_path_probability"],
                    "topk_paths": agg["topk_paths"],
                    "topk_path_probabilities": agg["topk_path_probabilities"],
                    "transition_probability": agg["transition_probability"],
                }
            )

    return pd.DataFrame(rows)


def compute_gnk_ranking(transition_df, alpha, beta):
    df = transition_df.copy()
    df["combined_coupling_strength"] = (
        alpha * df["T_Value"].astype(float)
        + beta * df["transition_probability"].astype(float)
    )
    df = df.sort_values(
        "combined_coupling_strength",
        ascending=False,
    ).reset_index(drop=True)
    df["ranking"] = np.arange(1, len(df) + 1)
    return df


def topn_list(df, col, top_n):
    return [
        value
        for value in df.head(top_n)[col].astype(str).tolist()
        if value and value.lower() != "nan"
    ]


def jaccard(list_a, list_b):
    set_a = set(list_a)
    set_b = set(list_b)

    if not (set_a | set_b):
        return 0.0

    return len(set_a & set_b) / len(set_a | set_b)


def rank_correlation_on_baseline_topn(baseline_df, comparison_df, top_n):
    base_top = baseline_df.head(top_n).copy()
    comparison_scores = dict(
        zip(
            comparison_df["factor_key"],
            comparison_df["combined_coupling_strength"],
        )
    )

    x = []
    y = []
    for _, row in base_top.iterrows():
        key = row["factor_key"]
        x.append(float(row["combined_coupling_strength"]))
        y.append(float(comparison_scores.get(key, 0.0)))

    spearman = spearmanr(x, y).correlation
    kendall = kendalltau(x, y).correlation

    if np.isnan(spearman):
        spearman = 0.0
    if np.isnan(kendall):
        kendall = 0.0

    return float(spearman), float(kendall)


def compare_against_k1(gnk_by_k, k_values, top_n):
    baseline_df = gnk_by_k[1]
    baseline_top = topn_list(baseline_df, "factor_key", top_n)
    rows = []

    for k in k_values:
        current_df = gnk_by_k[int(k)]
        current_top = topn_list(current_df, "factor_key", top_n)
        spearman, kendall = rank_correlation_on_baseline_topn(
            baseline_df=baseline_df,
            comparison_df=current_df,
            top_n=top_n,
        )

        rows.append(
            {
                "baseline_k": 1,
                "comparison_k": int(k),
                "top_n": int(top_n),
                "top10_jaccard": jaccard(baseline_top, current_top),
                "top10_overlap_count": len(set(baseline_top) & set(current_top)),
                "spearman_on_k1_top10": spearman,
                "kendall_on_k1_top10": kendall,
                "baseline_top10_mean_C": baseline_df.head(top_n)[
                    "combined_coupling_strength"
                ].mean(),
                "comparison_top10_mean_C": current_df.head(top_n)[
                    "combined_coupling_strength"
                ].mean(),
                "delta_top10_mean_C": (
                    current_df.head(top_n)["combined_coupling_strength"].mean()
                    - baseline_df.head(top_n)["combined_coupling_strength"].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def parse_k_values(text):
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def write_outputs(
    output_dir,
    output_prefix,
    transition_df,
    ranking_df,
    stability_df,
    gnk_by_k,
    skipped_mi_df,
    skipped_relation_df,
    top_n,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    transition_path = output_dir / f"{output_prefix}_transition_probability_all_k.csv"
    ranking_path = output_dir / f"{output_prefix}_gnk_ranking_all_k.csv"
    stability_path = output_dir / f"{output_prefix}_top10_stability_summary.csv"
    excel_path = output_dir / f"{output_prefix}_results.xlsx"

    transition_df.to_csv(transition_path, index=False, encoding="utf-8-sig")
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")
    stability_df.to_csv(stability_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(excel_path) as writer:
        transition_df.to_excel(writer, sheet_name="transition_all_k", index=False)
        ranking_df.to_excel(writer, sheet_name="gnk_ranking_all_k", index=False)
        stability_df.to_excel(writer, sheet_name="top10_stability", index=False)

        for k, df in gnk_by_k.items():
            df.head(top_n).to_excel(
                writer,
                sheet_name=f"top{top_n}_k{k}",
                index=False,
            )

        if not skipped_mi_df.empty:
            skipped_mi_df.to_excel(writer, sheet_name="skipped_mi", index=False)
        if not skipped_relation_df.empty:
            skipped_relation_df.to_excel(
                writer,
                sheet_name="skipped_relations",
                index=False,
            )

    return transition_path, ranking_path, stability_path, excel_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Top-k average path aggregation robustness analysis. "
            "k=1 is used as the max-path baseline."
        )
    )
    parser.add_argument("--mi-path", required=True, help="MI result file.")
    parser.add_argument(
        "--relation-path",
        required=True,
        help=(
            "Relation edge file. Recommended columns: "
            "source_code,target_code,count. Chinese factor names are also supported."
        ),
    )
    parser.add_argument(
        "--node-mapping-cache",
        default="",
        help=(
            "Optional JSON cache from the notebook. It should map raw node names "
            "to secondary_code values such as H1, MG1, TE2."
        ),
    )
    parser.add_argument("--output-dir", default="topk_path_aggregation_results")
    parser.add_argument("--output-prefix", default="gnk_topk_path_aggregation")
    parser.add_argument("--k-values", default="1,2,3,5")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--min-factor-num", type=int, default=2)
    parser.add_argument("--max-factor-num", type=int, default=3)
    args = parser.parse_args()

    k_values = parse_k_values(args.k_values)
    if 1 not in k_values:
        raise ValueError("--k-values must include 1 as the max-path baseline.")

    mi_candidate_df, skipped_mi_df = load_mi_candidates(
        mi_path=args.mi_path,
        min_factor_num=args.min_factor_num,
        max_factor_num=args.max_factor_num,
    )
    node_mapping = load_node_mapping_cache(args.node_mapping_cache)
    edge_df, skipped_relation_df = load_relation_edges(
        args.relation_path,
        node_mapping=node_mapping,
    )

    transition_df = compute_transition_probabilities(
        edge_df=edge_df,
        mi_candidate_df=mi_candidate_df,
        k_values=k_values,
    )

    gnk_by_k = {}
    ranking_frames = []
    for k in k_values:
        transition_k_df = (
            transition_df[transition_df["aggregation_k"] == int(k)]
            .copy()
            .reset_index(drop=True)
        )
        ranking_k_df = compute_gnk_ranking(
            transition_k_df,
            alpha=args.alpha,
            beta=args.beta,
        )
        ranking_k_df["aggregation_k"] = int(k)
        gnk_by_k[int(k)] = ranking_k_df
        ranking_frames.append(ranking_k_df)

    ranking_df = pd.concat(ranking_frames, ignore_index=True)
    stability_df = compare_against_k1(
        gnk_by_k=gnk_by_k,
        k_values=k_values,
        top_n=args.top_n,
    )

    output_paths = write_outputs(
        output_dir=Path(args.output_dir),
        output_prefix=args.output_prefix,
        transition_df=transition_df,
        ranking_df=ranking_df,
        stability_df=stability_df,
        gnk_by_k=gnk_by_k,
        skipped_mi_df=skipped_mi_df,
        skipped_relation_df=skipped_relation_df,
        top_n=args.top_n,
    )

    print("\nTop-k average path aggregation stability summary:")
    print(stability_df.to_string(index=False))
    print("\nOutput files:")
    for path in output_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
