#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export real classified causal edges from Neo4j for the complex-network baseline.

Input:
    Neo4j relationships such as (a)-[:LEADS_TO]->(b)

Output:
    real_edges.csv
        source,target,weight,relation

    real_edges_raw.csv
        source,target,weight,relation

    real_edges_details.csv
        raw_source,raw_target,classified_source,classified_target,raw_count

Usage:
    python export_neo4j_edges_for_baseline.py
    python complex_network_baseline.py --edges real_edges.csv --out-dir baseline_real
    python complex_network_baseline.py --edges real_edges_raw.csv --out-dir baseline_raw --core-node-count 20 --key-edge-count 60 --max-edges 4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from neo4j import GraphDatabase


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

VALID_SUBCATEGORIES = set(CATEGORY_MAP)

# These are defect/problem phenomena rather than causal factors. They should not
# enter the causal-factor complex network as H/MG/TE/... nodes.
DEFECT_KEYWORDS = [
    "飞溅",
    "焊接成型差",
    "焊缝成型差",
    "裂纹",
    "气孔",
    "夹渣",
    "咬边",
    "未熔合",
    "未焊透",
    "焊瘤",
    "变形",
    "焊接变形",
    "过度火工",
    "焊脚不足",
    "焊缝尺寸",
    "缺陷",
    "返修",
]

RULES = [
    ("操作规范", ["违反工艺", "违规", "未按图纸", "未按工艺", "操作不当", "焊接顺序", "施工不规范"]),
    ("责任心态度", ["责任心不足", "责任心不强", "质量意识", "盲目施工", "工作态度", "惯性思维", "操作人员不足"]),
    ("技能水平", ["技能", "培训效果不佳", "新员工", "无证", "看图纸能力", "能力不足"]),
    ("监督检查", ["检查不到位", "未检查", "没有按要求检查", "管理不到位", "监管", "巡检", "互检", "未及时发现"]),
    ("制度流程", ["制度", "流程", "转序", "派工", "未落实要求", "管理流程"]),
    ("培训交底", ["培训", "交底", "要求未传达", "岗前培训", "宣贯"]),
    ("设备状态", ["设备异常", "设备故障", "功能损坏", "维护不足", "电缆破损", "零部件缺失"]),
    ("工具使用", ["工具", "设备使用不当", "防护措施", "焊机故障", "工装"]),
    ("性能指标", ["电流", "电压", "行走速度", "加热温度", "性能指标", "参数波动"]),
    ("材料质量", ["材料不合格", "原材料", "钢板", "硬度超标", "材料质量"]),
    ("焊材管理", ["焊材", "焊丝", "焊条", "返锈", "返潮", "焊丝类型"]),
    ("存储条件", ["存储", "保温桶", "临时存放", "保管"]),
    ("工艺设计", ["设计缺陷", "工艺不完善", "坡口问题", "未制定", "新工艺应用不足"]),
    ("工艺执行", ["参数", "方法错误", "顺序混乱", "预处理", "焊接速度", "坡口内", "没有及时处理", "油脂", "水渍", "氧化皮", "垃圾物"]),
    ("技术标准", ["标准不明确", "规格不符", "间隙超差", "标准不清"]),
    ("天气条件", ["天气", "气温", "湿度", "雨水", "梅雨", "低温"]),
    ("施工环境", ["空间狭小", "施工难度", "环境条件", "风力", "光线", "通风", "施工环境"]),
]

CLASSIFICATION_RULES_PROMPT = """
你是一位专业的船舶焊接质量检测师。
请将以下提供的“问题因素”文本内容，根据其描述的核心问题，严格归类到以下6个大类别下的具体二级子因素中。每个问题因素只能归入一个最匹配的二级子因素，不得归入六大类名称本身。

六大类别及其对应的二级子因素定义如下：

1. 人员因素：
   - 技能水平：技能不足、培训效果不佳、新员工较多、不具备看图纸能力、无证上岗等
   - 责任心态度：责任心不足、质量意识淡薄、工作态度不认真、盲目施工、惯性思维等
   - 操作规范：违规操作、未按图纸施工、未按工艺要求、操作不当、违反焊接顺序等

2. 设备因素：
   - 设备状态：设备异常、功能损坏、零部件缺失、维护不足、绝缘电缆破损等
   - 工具使用：设备使用不当、工具不合格、防护措施不足、焊机故障等
   - 性能指标：电流电压异常、行走速度过大、加热温度不够等

3. 材料因素：
   - 材料质量：材料不合格、原材料不满足要求、压紧条硬度超标、钢板越级代用等
   - 焊材管理：焊材问题、焊丝返锈、焊条返潮、焊丝类型使用错误等
   - 存储条件：焊材临时存储不当、保温桶温度不足、焊材返潮返锈等

4. 工艺因素：
   - 工艺设计：设计缺陷、工艺不完善、坡口问题、未制定针对性工艺等
   - 工艺执行：参数不当、方法错误、顺序混乱、预处理不足、焊接速度过快等
   - 技术标准：标准不明确、规格不符、焊脚不足、坡口间隙超差等

5. 环境因素：
   - 天气条件：天气影响、气温过低、湿度过大、雨水天气、梅雨季节等
   - 施工环境：空间狭小、施工难度大、环境条件差、风力过大、光线不足等

6. 管理因素：
   - 监督检查：检查不到位、互检缺失、监管缺位、未及时发现问题、巡检不到位等
   - 制度流程：制度不完善、未落实要求、流程不规范、转序控制不严、派工不当等
   - 培训交底：培训不足、交底不清、要求未传达、标准不明确、未进行岗前培训等

归类规则：
- 只能返回列出的二级子因素名称：技能水平、责任心态度、操作规范、设备状态、工具使用、性能指标、材料质量、焊材管理、存储条件、工艺设计、工艺执行、技术标准、天气条件、施工环境、监督检查、制度流程、培训交底。
- 严禁返回一级大类名称。
- 每个问题因素仅归入一个最匹配的二级子因素。
- 若问题描述无法明确对应任一二级子因素，请归入“未分类”。
- 返回结果必须为严格 JSON，不要包含解释、注释或额外文本。

请严格按照以下 JSON 格式返回：
{"二级因素": ["子因素1"]}
"""


def get_all_relations(uri: str, user: str, password: str, rel_type: str) -> List[Tuple[str, str, int]]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = f"""
        MATCH (a)-[r:{rel_type}]->(b)
        RETURN a.name AS source, b.name AS target, count(r) AS count
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return [(record["source"], record["target"], int(record["count"])) for record in result]
    finally:
        driver.close()


def get_raw_network_relations(uri: str, user: str, password: str) -> List[Tuple[str, str, int, str]]:
    """Export a raw-factor-level propagation network.

    LEADS_TO is kept as factor -> factor.
    CAUSED_BY is reversed because the graph schema stores defect -> factor,
    while propagation analysis needs factor -> defect.
    BELONGS_TO and OCCURS_IN are intentionally excluded.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
        MATCH (a)-[r:LEADS_TO]->(b)
        RETURN a.name AS source, b.name AS target, count(r) AS count, 'LEADS_TO' AS relation

        UNION ALL

        MATCH (d)-[r:CAUSED_BY]->(f)
        RETURN f.name AS source, d.name AS target, count(r) AS count, 'CAUSES_DEFECT' AS relation
    """
    try:
        with driver.session() as session:
            result = session.run(query)
            return [
                (record["source"], record["target"], int(record["count"]), record["relation"])
                for record in result
            ]
    finally:
        driver.close()


def classify_by_rules(text: str) -> Optional[str]:
    if not text:
        return ""

    if any(keyword in text for keyword in DEFECT_KEYWORDS):
        return ""

    for category, keywords in RULES:
        if any(keyword in text for keyword in keywords):
            return category

    # None means "not handled by rules, try Ollama".
    return None


def extract_json_object(text: str) -> str:
    """Handle occasional wrappers such as <think>...</think> or extra text."""
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return match.group(0)


def classify_factor_with_ollama(
    factor: str,
    api_url: str,
    model: str,
    cache: Dict[str, str],
    sleep_seconds: float,
    connect_timeout: int,
    read_timeout: int,
) -> str:
    if factor in cache:
        return cache[factor]

    rule_result = classify_by_rules(factor)
    if rule_result is not None:
        cache[factor] = rule_result
        return rule_result

    prompt = f"{CLASSIFICATION_RULES_PROMPT}\n\n问题因素：{factor}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "num_predict": 80,
        },
    }

    response = requests.post(api_url, json=payload, timeout=(connect_timeout, read_timeout))
    response.raise_for_status()
    content = response.json().get("response", "").strip()
    parsed = json.loads(extract_json_object(content))
    categories = parsed.get("二级因素", [])

    selected = ""
    for category in categories:
        category = str(category).strip()
        if category in VALID_SUBCATEGORIES:
            selected = category
            break

    cache[factor] = selected
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return selected


def load_cache(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_cache(path: Path, cache: Dict[str, str]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def build_classified_edges(
    relation_data: Iterable[Tuple[str, str, int]],
    api_url: str,
    model: str,
    cache: Dict[str, str],
    sleep_seconds: float,
    use_codes: bool,
    skip_self_loops: bool,
    connect_timeout: int,
    read_timeout: int,
) -> Tuple[Dict[Tuple[str, str], int], List[Tuple[str, str, str, str, int]]]:
    edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
    details: List[Tuple[str, str, str, str, int]] = []

    for source, target, count in relation_data:
        try:
            source_category = classify_factor_with_ollama(
                source, api_url, model, cache, sleep_seconds, connect_timeout, read_timeout
            )
            target_category = classify_factor_with_ollama(
                target, api_url, model, cache, sleep_seconds, connect_timeout, read_timeout
            )
        except Exception as exc:
            print(f"跳过分类失败关系：{source} -> {target}，错误：{exc}")
            continue

        if not source_category or not target_category:
            print(f"跳过无法分类关系：{source} -> {target}")
            continue

        source_node = CATEGORY_MAP[source_category] if use_codes else source_category
        target_node = CATEGORY_MAP[target_category] if use_codes else target_category

        if skip_self_loops and source_node == target_node:
            print(f"跳过自环关系：{source} ({source_node}) -> {target} ({target_node})")
            continue

        edge_weights[(source_node, target_node)] += count
        details.append((source, target, source_node, target_node, count))
        print(f"{source} ({source_node}) -> {target} ({target_node}) | Count: {count}")

    return edge_weights, details


def write_edges(path: Path, edge_weights: Dict[Tuple[str, str], int], relation: str) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "weight", "relation"])
        writer.writeheader()
        for (source, target), weight in sorted(edge_weights.items(), key=lambda item: item[1], reverse=True):
            writer.writerow(
                {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "relation": relation,
                }
            )


def write_raw_edges(path: Path, rows: Iterable[Tuple[str, str, int, str]]) -> None:
    edge_weights: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for source, target, count, relation in rows:
        if not source or not target or source == target:
            continue
        edge_weights[(source, target, relation)] += count

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "target", "weight", "relation"])
        writer.writeheader()
        for (source, target, relation), weight in sorted(edge_weights.items(), key=lambda item: item[1], reverse=True):
            writer.writerow(
                {
                    "source": source,
                    "target": target,
                    "weight": weight,
                    "relation": relation,
                }
            )


def write_details(path: Path, details: List[Tuple[str, str, str, str, int]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["raw_source", "raw_target", "classified_source", "classified_target", "raw_count"],
        )
        writer.writeheader()
        for raw_source, raw_target, source, target, count in details:
            writer.writerow(
                {
                    "raw_source": raw_source,
                    "raw_target": raw_target,
                    "classified_source": source,
                    "classified_target": target,
                    "raw_count": count,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="12345678")
    parser.add_argument("--rel-type", default="LEADS_TO")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--model", default="qwen3:30b-a3b-Instruct")
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--connect-timeout", type=int, default=10)
    parser.add_argument("--read-timeout", type=int, default=300)
    parser.add_argument("--cache", type=Path, default=Path("factor_classification_cache.json"))
    parser.add_argument("--out", type=Path, default=Path("real_edges.csv"))
    parser.add_argument("--raw-out", type=Path, default=Path("real_edges_raw.csv"))
    parser.add_argument("--details-out", type=Path, default=Path("real_edges_details.csv"))
    parser.add_argument("--use-names", action="store_true", help="Use Chinese subcategory names instead of H1/MG1 codes")
    parser.add_argument("--keep-self-loops", action="store_true", help="Keep edges such as MG1 -> MG1")
    parser.add_argument("--no-raw-export", action="store_true", help="Do not export raw-factor-level network edges")
    args = parser.parse_args()

    print("正在从 Neo4j 获取关系数据...")
    relation_data = get_all_relations(args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.rel_type)
    print(f"获取到 {len(relation_data)} 条原始关系记录。")

    if not args.no_raw_export:
        raw_relation_data = get_raw_network_relations(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        write_raw_edges(args.raw_out, raw_relation_data)
        print(f"已输出原始因素层复杂网络边表：{args.raw_out}")

    cache = load_cache(args.cache)
    edge_weights, details = build_classified_edges(
        relation_data=relation_data,
        api_url=args.ollama_url,
        model=args.model,
        cache=cache,
        sleep_seconds=args.sleep_seconds,
        use_codes=not args.use_names,
        skip_self_loops=not args.keep_self_loops,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
    )

    write_edges(args.out, edge_weights, args.rel_type)
    write_details(args.details_out, details)
    save_cache(args.cache, cache)

    print(f"分类聚合后得到 {len(edge_weights)} 条二级因素有向边。")
    print(f"已输出复杂网络基线边表：{args.out}")
    print(f"已输出原始关系映射明细：{args.details_out}")
    print(f"已保存分类缓存：{args.cache}")


if __name__ == "__main__":
    main()
