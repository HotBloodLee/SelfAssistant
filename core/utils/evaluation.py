
# file: core/utils/evaluation.py
import json
from typing import Dict, Any, Sequence, Optional

from sacrebleu import corpus_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score as nltk_meteor_score


def evaluate_json(pred_str: str, required_fields: Sequence[str]) -> Dict[str, Any]:
    result = {
        "json_valid": 0,
        "field_coverage": 0.0,
        "missing_fields": [],
        "parsed": None
    }
    try:
        obj = json.loads(pred_str)
        result["json_valid"] = 1
        result["parsed"] = obj
    except Exception:
        result["missing_fields"] = list(required_fields)
        return result

    def get_by_path(d: Any, path: str):
        cur = d
        for seg in path.split("."):
            if isinstance(cur, dict) and seg in cur:
                cur = cur[seg]
            else:
                return None
        return cur

    missing = []
    present = 0
    for f in required_fields:
        val = get_by_path(obj, f)
        if val is None or (val == "" or val == [] or val == {}):
            missing.append(f)
        else:
            present += 1
    total = len(required_fields) if required_fields else 1
    result["field_coverage"] = present / total
    result["missing_fields"] = missing
    return result


def _zh_tokenize(text: str):
    """
    极简中文分词:
    1. 优先使用 jieba (若已安装)
    2. 否则按字切分
    """
    text = text.strip()
    if not text:
        return []
    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except Exception:
        return list(text)


def evaluate_text_metrics(
        candidates: Sequence[str],
        references: Sequence[Sequence[str]],
        meteor_lang: str = "zh"
) -> Dict[str, Any]:
    assert len(candidates) == len(references), "candidates 与 references 数量不一致"

    # 保留原始文本
    orig_candidates = list(candidates)
    orig_references = [list(refs) for refs in references]

    # 对中文文本进行预处理：分词后再用空格连接
    def prepare_for_metrics(text):
        return " ".join(_zh_tokenize(text))

    if meteor_lang == "zh":
        # 生成处理后的文本（不修改原始文本）
        proc_cands = [prepare_for_metrics(c) for c in orig_candidates]
        proc_refs = [[prepare_for_metrics(r) for r in refs] for refs in orig_references]
    else:
        # 非中文时，直接使用原始文本
        proc_cands = list(orig_candidates)
        proc_refs = [list(refs) for refs in orig_references]

    # BLEU
    refs_by_ref_index = list(zip(*proc_refs))  # 转置
    bleu = corpus_bleu(
        proc_cands,
        refs_by_ref_index,
        tokenize='none',  # 已经分词，不再分
        smooth_method='exp'  # 平滑处理避免零分
    ).score

    # ROUGE
    rouge = Rouge()
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for cand, refs in zip(candidates, references):
        best_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        cand_tok = "".join(_zh_tokenize(cand))  # 对中文，直接连接更合适

        for ref in refs:
            ref_tok = "".join(_zh_tokenize(ref))
            try:
                scores = rouge.get_scores(cand_tok, ref_tok)[0]
                best_scores["rouge1"] = max(best_scores["rouge1"], scores["rouge-1"]["f"])
                best_scores["rouge2"] = max(best_scores["rouge2"], scores["rouge-2"]["f"])
                best_scores["rougeL"] = max(best_scores["rougeL"], scores["rouge-l"]["f"])
            except Exception:
                continue

        rouge_scores["rouge1"].append(best_scores["rouge1"])
        rouge_scores["rouge2"].append(best_scores["rouge2"])
        rouge_scores["rougeL"].append(best_scores["rougeL"])

    rouge_avg = {k: sum(v) / len(v) if v else 0.0 for k, v in rouge_scores.items()}

    # METEOR（使用预分词的token列表）
    meteor_vals = []
    for cand, refs in zip(orig_candidates, orig_references):
        try:
            cand_tok = _zh_tokenize(cand)
            refs_tok = [_zh_tokenize(r) for r in refs]
            meteor_vals.append(nltk_meteor_score(refs_tok, cand_tok))
        except Exception:
            meteor_vals.append(0.0)

    meteor_avg = sum(meteor_vals) / len(meteor_vals) if meteor_vals else 0.0

    return {
        "bleu": bleu,
        "rouge1": rouge_avg["rouge1"],
        "rouge2": rouge_avg["rouge2"],
        "rougeL": rouge_avg["rougeL"],
        "meteor": meteor_avg,
    }


def evaluate_all(
    json_pred: Optional[str],
    required_fields: Sequence[str],
    text_candidates: Optional[Sequence[str]],
    text_references: Optional[Sequence[Sequence[str]]]
) -> Dict[str, Any]:
    out = {}
    if json_pred is not None:
        out["json_metrics"] = evaluate_json(json_pred, required_fields)
    if text_candidates is not None and text_references is not None:
        out["text_metrics"] = evaluate_text_metrics(text_candidates, text_references)
    return out

if __name__ == "__main__":
    pred_json = '{"title": "示例", "meta": {"id": 123}, "content": "正文"}'
    json_req = ["title", "meta.id", "content", "meta.author"]
    print("JSON:", evaluate_json(pred_json, json_req))

    cands = ["今天天气很好", "今天天气很好"]
    refs = [
        ["今天天气很好", "今天天气不错"],
        ["今天天气不错"]
    ]
    print("TEXT:", evaluate_text_metrics(cands, refs), "en")