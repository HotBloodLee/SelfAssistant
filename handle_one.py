from pprint import pprint
import os
from typing import Tuple, Dict

import fitz
import pdfplumber
import pandas as pd
import pptx
import docx
import re

import re
from typing import Tuple, Dict, List, Any

from tqdm import tqdm


def parse_pdf(path: str) -> Tuple[str, Dict]:
    # 使用PyMuPDF提取结构化元素
    doc = fitz.open(path)
    structured_data = {
        "type": "pdf",
        "pages": len(doc),
        "elements": []
    }

    # 提取文档元数据
    metadata = doc.metadata
    structured_data["metadata"] = {
        "title": metadata.get("title", ""),
        "author": metadata.get("author", ""),
        "subject": metadata.get("subject", ""),
        "keywords": metadata.get("keywords", ""),
        "creation_date": metadata.get("creationDate", ""),
        "modification_date": metadata.get("modDate", "")
    }

    full_text = ""

    with pdfplumber.open(path) as pdf:
        for page_num in tqdm(range(len(doc))):
            page_ = pdf.pages[page_num]

            # 提取当前页的结构化元素
            page_elements = extract_page_elements(doc, page_num, page_)
            structured_data["elements"].extend(page_elements)

    return full_text, structured_data


def extract_page_elements(doc: fitz.Document, page_num: int, page_) -> List[Dict[str, Any]]:
    """提取单页的结构化元素"""
    page = doc.load_page(page_num)
    elements = []

    # 提取文本块（标题和段落）
    blocks = page.get_text("dict")["blocks"]


    # 提取表格（使用pdfplumber更准确）
    page_obj = page_
    tables = page_obj.extract_tables()
    for i, table in enumerate(tables):
        # 转换为二维列表
        table_data = []
        for row in table:
            cleaned_row = [cell.replace("\n", " ") if cell else "" for cell in row]
            table_data.append(cleaned_row)

        elements.append({
            "type": "table",
            "data": table_data,
            "page": page_num + 1,
            "table_index": i
        })

    return elements

raw_text, parsed_output = parse_pdf("dataset/raw_data/Other/W02024072315392024年7月23日贵州省普通高校招生信息表（普通类本科批—物理组合）.pdf")


tables = parsed_output["elements"]

infos = list()

for table in tables:
    data = table["data"]
    for row in data:
        if row[0] == '序号':
            continue
        infos.append({
            "序号": row[0],
            "院校代码": row[1],
            "院校名称": row[2],
            "专业代码": row[3],
            "专业名称": row[4],
            "招考类型": row[5],
            "投档人数": row[6],
            "投档最低分": row[7],
            "投档最低位次": row[8],
            "页码": table["page"],
        })

# save to json file
import json
with open("dataset/raw_data/Other/W02024072315392024年7月23日贵州省普通高校招生信息表（普通类本科批—物理组合）.json", "w", encoding="utf-8") as f:
    json.dump(infos, f, ensure_ascii=False, indent=4)