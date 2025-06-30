# coding: utf-8
"""
此脚本实现从 Excel、PDF、PPTX、DOCX、Markdown 文件中提取结构化内容，
并构造用于大模型微调的 SFT 和 DPO 数据集：
- SFT: {'input': 原始内容, 'output': 提取的格式化内容}
- DPO: {'prompt': 原始内容, 'chosen': 正确提取, 'rejected': 错误提取}
"""

import os
import json
from typing import List, Dict
from core.utils.parsers import parse_pdf, parse_excel, parse_pptx, parse_docx, parse_markdown
from core.utils.augment import generate_rejected_outputs

DATA_DIR = "dataset/raw_data/PPT"
SFT_OUTPUT_PATH = "dataset/sft_dataset.jsonl"
SFT_OUTPUT_JSON_PATH = "dataset/sft_dataset.json"
DPO_OUTPUT_PATH = "dataset/dpo_dataset.jsonl"

# 工具函数：保存 JSONL 文件
def write_jsonl(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def write_json(data: List[Dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)

# 主处理逻辑
sft_data, dpo_data = [], []

for fname in os.listdir(DATA_DIR)[:2]:
    fpath = os.path.join(DATA_DIR, fname)
    ext = fname.lower().split(".")[-1]
    raw_text, parsed_output = None, None

    try:
        if ext == 'pdf':
            raw_text, parsed_output = parse_pdf(fpath)
        elif ext == 'xlsx':
            raw_text, parsed_output = parse_excel(fpath)
        elif ext == 'pptx' or ext == 'ppt':
            raw_text, parsed_output = parse_pptx(fpath)
        elif ext == 'docx' or ext == 'doc':
            raw_text, parsed_output = parse_docx(fpath)
        elif ext == 'md':
            raw_text, parsed_output = parse_markdown(fpath)

        if raw_text and parsed_output:
            # 构造 SFT 样本
            sft_data.append({
                "input": raw_text.strip(),
                "preview": parsed_output
            })

            # 构造 DPO 样本（基于 SFT 正确输出 + 生成若干个负样本）
            # rejected_list = generate_rejected_outputs(parsed_output)
            # for rejected in rejected_list:
            #     dpo_data.append({
            #         "prompt": raw_text.strip(),
            #         "chosen": json.dumps(parsed_output, ensure_ascii=False),
            #         "rejected": json.dumps(rejected, ensure_ascii=False)
            #     })

    except Exception as e:
        raise e
        print(f"⚠️ 处理失败: {fname}: {e}")

# 保存结果
write_jsonl(sft_data, SFT_OUTPUT_PATH)
write_json(sft_data, SFT_OUTPUT_JSON_PATH)  # 也可以保存为 JSON 格式
# write_jsonl(dpo_data, DPO_OUTPUT_PATH)
print(f"✅ SFT 样本数: {len(sft_data)} 条")
