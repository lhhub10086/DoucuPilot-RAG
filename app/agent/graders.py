import re

import jieba

from app.core.schemas import RetrievedChunk
from app.indexer.bm25_index import tokenize

GENERIC_TOKENS = {
    "文档",
    "这篇",
    "这个",
    "主要",
    "什么",
    "有没有",
    "是否",
    "提到",
    "内容",
    "主要内容",
    "核心",
    "方法",
    "核心方法",
    "摘要",
    "讲",
    "了",
    "的",
    "吗",
}


def classify_question_type(question: str) -> str:
    if any(keyword in question for keyword in ["主要讲了什么", "总结", "概括"]):
        return "summary"
    if any(keyword in question for keyword in ["区别", "对比", "比较"]):
        return "compare"
    if any(keyword in question for keyword in ["方法", "算法", "模型", "框架"]):
        return "method"
    return "fact"


def content_tokens(question: str) -> set[str]:
    raw_tokens = [token.strip() for token in jieba.lcut(question) if token.strip()]
    tokens = set()
    for token in raw_tokens:
        if token in GENERIC_TOKENS:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]", token):
            continue
        tokens.add(token)
    return tokens


def grade_retrieval(question: str, documents: list[RetrievedChunk]) -> tuple[bool, str | None]:
    if not documents:
        return False, "low_retrieval_confidence"

    top_score = max(document.score for document in documents)
    query_tokens = content_tokens(question)
    joined_documents = " ".join(document.text.lower() for document in documents[:5])

    if query_tokens and not any(token.lower() in joined_documents for token in query_tokens):
        return False, "low_retrieval_confidence"
    if top_score < 0.005:
        return False, "low_retrieval_confidence"
    return True, None


def grade_answer_support(answer: str, citations: list[dict]) -> tuple[bool, str]:
    if not citations:
        return False, "unsupported"
    if "当前文档中未找到充分依据" in answer:
        return False, "unsupported"
    if answer.strip():
        return True, "supported"
    return False, "unsupported"


def rewrite_by_rule(question: str, question_type: str | None) -> str:
    file_tokens = [token for token in tokenize(question) if "." in token]
    prefix = " ".join(file_tokens)
    if question_type == "summary":
        rewritten = "文档 主要内容 核心方法 摘要"
    elif question_type == "method":
        rewritten = "核心方法 算法 框架 模型 实验"
    elif question_type == "compare":
        rewritten = "区别 对比 优势 不足 方法比较"
    else:
        rewritten = question
    return f"{prefix} {rewritten}".strip()
