# utils/augment.py
from typing import Dict, List
import random
import copy
import json

# 基于 ground truth 结构化输出，生成多种类型的负样本

def generate_rejected_outputs(parsed: Dict) -> List[Dict]:
    samples = []

    def drop_fields(obj: Dict) -> Dict:
        result = copy.deepcopy(obj)
        keys = list(result.keys())
        if "type" in keys:
            keys.remove("type")
        if len(keys) > 1:
            drop_n = max(1, len(keys) // 3)
            for k in random.sample(keys, drop_n):
                result.pop(k, None)
        return result

    def shuffle_values(obj: Dict) -> Dict:
        result = copy.deepcopy(obj)
        keys = [k for k in result.keys() if k != "type"]
        values = [result[k] for k in keys]
        random.shuffle(values)
        for k, v in zip(keys, values):
            result[k] = v
        return result

    def flatten_content(obj: Dict) -> Dict:
        return {"type": obj.get("type", "unknown"), "content": json.dumps(obj, ensure_ascii=False)}

    def insert_noise(obj: Dict) -> Dict:
        result = copy.deepcopy(obj)
        result["noise"] = "这是一个无关字段"
        return result

    def rename_keys(obj: Dict) -> Dict:
        result = {}
        for k, v in obj.items():
            if k == "type":
                result[k] = v
            else:
                result[k + "_wrong"] = v
        return result

    def wipe_values(obj: Dict) -> Dict:
        result = copy.deepcopy(obj)
        for k in result:
            if isinstance(result[k], list):
                result[k] = []
            elif isinstance(result[k], dict):
                result[k] = {}
        return result

    def missing_section(obj: Dict) -> Dict:
        result = copy.deepcopy(obj)
        if "chapters" in result:
            if isinstance(result["chapters"], list) and len(result["chapters"]) > 1:
                result["chapters"] = result["chapters"][:-1]  # 删除一个章节
        elif "sections" in result:
            keys = list(result["sections"].keys())
            if len(keys) > 1:
                result["sections"].pop(keys[-1])
        return result

    def disorder_sections(obj: Dict) -> Dict:
        result = copy.deepcopy(obj)
        if "chapters" in result and isinstance(result["chapters"], list):
            random.shuffle(result["chapters"])
        elif "sections" in result and isinstance(result["sections"], dict):
            items = list(result["sections"].items())
            random.shuffle(items)
            result["sections"] = dict(items)
        return result

    def wrong_type_assignment(obj: Dict) -> Dict:
        result = copy.deepcopy(obj)
        if "chapters" in result:
            for ch in result["chapters"]:
                if "title" in ch:
                    ch["bullets"] = [ch["title"]] + ch.get("bullets", [])
                    ch["title"] = "- 模块A"
        return result

    def vague_output(obj: Dict) -> Dict:
        return {
            "type": obj.get("type", "unknown"),
            "summary": "本文件主要介绍了该项目的发展方向及背景信息，涵盖若干内容。"
        }

    def broken_structure(obj: Dict) -> Dict:
        return {
            "sections": "abstract: 本文提出了一种方法 introduction: 方法介绍如下",
            "note": "字段嵌套格式已损坏"
        }

    def irrelevant_repeat(obj: Dict) -> Dict:
        return {
            "type": obj.get("type", "unknown"),
            "content": "以下是原文内容：\n\n" + json.dumps(obj, ensure_ascii=False)
        }

    reject_fns = [
        drop_fields,
        shuffle_values,
        flatten_content,
        insert_noise,
        rename_keys,
        wipe_values,
        missing_section,
        disorder_sections,
        wrong_type_assignment,
        vague_output,
        broken_structure,
        irrelevant_repeat
    ]

    for fn in reject_fns:
        try:
            variant = fn(parsed)
            samples.append(variant)
        except Exception:
            continue

    return samples