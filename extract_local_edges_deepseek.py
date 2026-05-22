from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm


CATEGORY_MAP = {
    "技能水平": "H1",
    "责任心态度": "H2",
    "操作规范": "H3",
    "设备状态": "EQ1",
    "工具使用": "EQ2",
    "性能指标": "EQ3",
    "材料质量": "MA1",
    "焊材管理": "MA2",
    "存储条件": "MA3",
    "工艺设计": "TE1",
    "工艺执行": "TE2",
    "技术标准": "TE3",
    "天气条件": "EN1",
    "施工环境": "EN2",
    "监督检查": "MG1",
    "制度流程": "MG2",
    "培训交底": "MG3",
}

VALID_FACTORS = set(CATEGORY_MAP)
VALID_RELATIONS = {"导致", "促进", "未能及时发现", "并列关联"}


EXTRACTION_PROMPT = """
你是一位专业的船舶焊接质量检测师。

请阅读输入的“问题原因分析”文本，完成以下任务：
1. 识别文本中明确出现的二级因素；
2. 仅在文本明确表达了方向性时，抽取局部有向关系；
3. 区分真正的因果传递和“未能及时发现/未能拦截”的管理失效关系；
4. 不要把纯并列叙述、句子先后顺序，机械地当成因果顺序。

二级因素只能从以下列表中选择：
技能水平、责任心态度、操作规范、
设备状态、工具使用、性能指标、
材料质量、焊材管理、存储条件、
工艺设计、工艺执行、技术标准、
天气条件、施工环境、
监督检查、制度流程、培训交底

关系类型只能从以下列表中选择：
- 导致：前一因素明确引发后一因素；
- 促进：前一因素使后一因素更容易发生，但文本未达到强因果；
- 未能及时发现：管理/检查因素未能拦截后续问题；
- 并列关联：文本只说明共同出现，不具有明确方向。

抽取规则：
- 只有“导致、造成、引起、进而、使得、从而、未及时发现”等明确语义时，才输出有向关系；
- 对“监督检查不到位、未及时发现问题”这类表述，优先使用“未能及时发现”，不要默认把它串接成主因果链；
- 若文本中只有因素，但没有可靠方向，可返回空的“关系”列表；
- 不要输出一级大类名称；
- 返回结果必须是严格 JSON，不要有任何解释。

返回格式：
{
  "二级因素": ["责任心态度", "操作规范", "监督检查", "技术标准"],
  "关系": [
    {"起点": "责任心态度", "终点": "操作规范", "关系类型": "导致"},
    {"起点": "操作规范", "终点": "技术标准", "关系类型": "导致"},
    {"起点": "监督检查", "终点": "技术标准", "关系类型": "未能及时发现"}
  ],
  "结果问题": "焊角不符合图纸要求"
}
"""


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def extract_json_text(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()
    match = re.search(r"\{.*\}", content, flags=re.S)
    return match.group(0) if match else content


def normalize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    factors = result.get("二级因素", [])
    if isinstance(factors, str):
        factors = [factors]
    factors = [
        factor.strip()
        for factor in factors
        if isinstance(factor, str) and factor.strip() in VALID_FACTORS
    ]
    factors = list(dict.fromkeys(factors))

    normalized_edges: List[Dict[str, str]] = []
    for edge in result.get("关系", []) or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("起点", "")).strip()
        target = str(edge.get("终点", "")).strip()
        relation = str(edge.get("关系类型", "")).strip()
        if (
            source in VALID_FACTORS
            and target in VALID_FACTORS
            and source != target
            and relation in VALID_RELATIONS
        ):
            normalized_edges.append(
                {"起点": source, "终点": target, "关系类型": relation}
            )

    return {
        "二级因素": factors,
        "关系": normalized_edges,
        "结果问题": str(result.get("结果问题", "")).strip(),
    }


def extract_structure(
    client: Any,
    text: str,
    model: str,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, Any]:
    if pd.isna(text) or not str(text).strip():
        return {"二级因素": [], "关系": [], "结果问题": ""}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": str(text).strip()},
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
            content = response.choices[0].message.content.strip()
            return normalize_result(json.loads(extract_json_text(content)))
        except Exception:
            if attempt == max_retries - 1:
                return {"二级因素": [], "关系": [], "结果问题": ""}
            time.sleep(sleep_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract local causal edges with DeepSeek.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--text-column", default="issue_text")
    parser.add_argument("--case-id-column", default=None)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--progress", type=Path, default=Path("local_edges_progress.csv"))
    parser.add_argument("--structured-output", type=Path, default=Path("structured_cases_with_local_relations.xlsx"))
    parser.add_argument("--edges-output", type=Path, default=Path("local_edges.csv"))
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The 'openai' package is required to call DeepSeek. "
            "Install it in the Python environment used to run this script."
        ) from exc

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    df = pd.read_csv(args.progress) if args.progress.exists() else read_table(args.input)
    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}")
    for col in ["局部关系", "结果问题", "结构化抽取"]:
        if col not in df.columns:
            df[col] = None

    for idx in tqdm(df.index, total=len(df)):
        if pd.notna(df.at[idx, "结构化抽取"]) and str(df.at[idx, "结构化抽取"]).strip():
            continue
        result = extract_structure(
            client=client,
            text=df.at[idx, args.text_column],
            model=args.model,
            max_retries=args.max_retries,
            sleep_seconds=args.sleep_seconds,
        )
        df.at[idx, "局部关系"] = json.dumps(result["关系"], ensure_ascii=False)
        df.at[idx, "结果问题"] = result["结果问题"]
        df.at[idx, "结构化抽取"] = json.dumps(result, ensure_ascii=False)
        df.to_csv(args.progress, index=False, encoding="utf-8-sig")

    edge_rows = []
    for idx, row in df.iterrows():
        case_id = row[args.case_id_column] if args.case_id_column else idx
        relations = json.loads(row["局部关系"]) if pd.notna(row["局部关系"]) else []
        for edge in relations:
            edge_rows.append(
                {
                    "case_id": case_id,
                    "source": edge["起点"],
                    "source_code": CATEGORY_MAP[edge["起点"]],
                    "target": edge["终点"],
                    "target_code": CATEGORY_MAP[edge["终点"]],
                    "relation_type": edge["关系类型"],
                }
            )

    edge_df = pd.DataFrame(edge_rows)
    edge_df.to_csv(args.edges_output, index=False, encoding="utf-8-sig")
    df.to_excel(args.structured_output, index=False)
    print(f"wrote {args.edges_output}")
    print(f"wrote {args.structured_output}")


if __name__ == "__main__":
    main()
