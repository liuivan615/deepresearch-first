import ast
import logging
import os
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI
from requests import RequestException

from mcp.server.fastmcp import FastMCP
from prompts import *  # noqa: F401,F403 - reuse existing prompt constants

mcp = FastMCP("search")

# =============================
# 配置 & 日志
# =============================
OPENAI_BASE = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "aaa")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek/deepseek-chat:free")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8088")
SEARXNG_TIMEOUT = float(os.getenv("SEARXNG_TIMEOUT", "12"))
SEARXNG_LANGUAGE = os.getenv("SEARXNG_LANGUAGE", "zh-CN")
SEARXNG_RESULT_LIMIT = int(os.getenv("SEARXNG_RESULT_LIMIT", "8"))

logger = logging.getLogger("search_mcp")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler("test.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

client = OpenAI(base_url=OPENAI_BASE, api_key=OPENAI_KEY)


# =============================
# LLM 辅助函数
# =============================

def parse_python_list(text: str) -> List[str]:
    if not text:
        return []
    cleaned = text.strip()
    if "```" in cleaned:
        cleaned = cleaned.split("```", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = ast.literal_eval(cleaned)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception as exc:  # noqa: BLE001 - logging for visibility
        logger.warning("无法解析列表：%s", exc)
    return []


def generate_query(query: str) -> List[str]:
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic.\n"
        "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful and precise research assistant."},
            {"role": "user", "content": f"User Query: {query}\n\n{prompt}"},
        ],
    )
    return parse_python_list(response.choices[0].message.content)


def if_useful(query: str, page_text: str) -> str:
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, determine "
        "if the webpage contains information relevant and useful for addressing the query.\n"
        "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
            {
                "role": "user",
                "content": (
                    f"User Query: {query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"
                ),
            },
        ],
    )
    answer = (response.choices[0].message.content or "").strip()
    if answer in {"Yes", "No"}:
        return answer
    if "Yes" in answer:
        return "Yes"
    if "No" in answer:
        return "No"
    return "No"


def extract_relevant_context(query: str, search_query: str, page_text: str) -> str:
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query.\n"
        "Return only the relevant context as plain text without commentary."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
            {
                "role": "user",
                "content": (
                    f"User Query: {query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n"
                    f"{page_text[:20000]}\n\n{prompt}"
                ),
            },
        ],
    )
    return (response.choices[0].message.content or "").strip()


def get_new_search_queries(user_query: str, previous_search_queries: List[str], all_contexts: List[str]):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed.\n"
        "If further research is needed, provide up to four new search queries as a Python list. "
        "If you believe no further research is needed, respond with exactly []."
    )
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
            {
                "role": "user",
                "content": (
                    f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\n"
                    f"Extracted Relevant Contexts:\n{context_combined}\n\n{prompt}"
                ),
            },
        ],
    )
    return parse_python_list(response.choices[0].message.content)


# =============================
# 搜索相关工具
# =============================

def build_searxng_endpoint() -> str:
    base = SEARXNG_URL.rstrip("/")
    return f"{base}/search"


def web_search(query: str, top_k: int = 3, categories: str = "general") -> List[Dict[str, Any]]:
    endpoint = build_searxng_endpoint()
    params = {
        "format": "json",
        "q": query,
        "language": SEARXNG_LANGUAGE,
        "time_range": "",
        "safesearch": 0,
        "categories": categories,
    }
    try:
        response = requests.get(endpoint, params=params, timeout=SEARXNG_TIMEOUT)
        response.raise_for_status()
        results = response.json().get("results", [])
    except RequestException as exc:
        logger.error("搜索引擎请求失败：%s", exc)
        return []
    except ValueError as exc:
        logger.error("解析搜索结果失败：%s", exc)
        return []

    items: List[Dict[str, Any]] = []
    for result in results:
        if categories == "images":
            img_src = result.get("img_src") or result.get("thumbnail")
            if not img_src:
                continue
            items.append(
                {
                    "img_src": img_src,
                    "title": result.get("title") or "",
                    "url": result.get("url"),
                    "source": result.get("source"),
                }
            )
        else:
            url = result.get("url")
            if not url:
                continue
            items.append(
                {
                    "url": url,
                    "title": result.get("title") or url,
                    "snippet": result.get("content", ""),
                }
            )
        if len(items) >= min(top_k, SEARXNG_RESULT_LIMIT):
            break
    return items


def fetch_webpage_text(url: str) -> str:
    jina_proxy = "https://r.jina.ai/"
    try:
        resp = requests.get(f"{jina_proxy}{url}", timeout=50)
        if resp.status_code == 200:
            return resp.text
        logger.info("Jina fetch error for %s: %s", url, resp.status_code)
    except RequestException as exc:
        logger.error("请求网页内容失败：%s", exc)
    return ""


def process_link(result: Dict[str, Any], user_query: str) -> Optional[Dict[str, str]]:
    url = result.get("url")
    if not url:
        return None
    search_query = result.get("search_query") or result.get("query") or user_query
    logger.info("Fetching content from: %s", url)
    page_text = fetch_webpage_text(url)
    if not page_text:
        return None
    usefulness = if_useful(user_query, page_text)
    logger.info("Page usefulness for %s: %s", url, usefulness)
    if usefulness != "Yes":
        return None
    context = extract_relevant_context(user_query, search_query, page_text)
    if not context:
        return None
    trimmed = context[:2000]
    logger.info("Extracted context from %s (first 200 chars): %s", url, trimmed[:200])
    return {
        "url": url,
        "title": result.get("title") or url,
        "query": search_query,
        "context": trimmed,
    }


def get_images_description(image_url: str) -> str:
    completion = client.chat.completions.create(
        model="qwen/qwen2.5-vl-32b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "使用一句话描述图片的内容"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )
    return completion.choices[0].message.content or ""


# =============================
# MCP 工具实现
# =============================

@mcp.tool()
def search(query: str) -> str:
    """互联网搜索"""
    iteration_limit = 3
    aggregated_contexts: List[Dict[str, str]] = []
    all_search_queries: List[str] = []

    new_search_queries = generate_query(query)
    if not new_search_queries:
        new_search_queries = [query]
    all_search_queries.extend(new_search_queries)
    if query not in all_search_queries:
        all_search_queries.append(query)

    iteration = 0
    while iteration < iteration_limit and new_search_queries:
        logger.info("=== Iteration %d ===", iteration + 1)
        iteration_contexts: List[Dict[str, str]] = []
        unique_links: Dict[str, Dict[str, Any]] = {}

        for sq in new_search_queries:
            results = web_search(sq, top_k=3, categories="general")
            for item in results:
                url = item.get("url")
                if not url or url in unique_links:
                    continue
                unique_links[url] = {**item, "search_query": sq}

        logger.info("Collected %d unique links in this round", len(unique_links))

        for meta in unique_links.values():
            processed = process_link(meta, query)
            if processed:
                iteration_contexts.append(processed)

        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
        else:
            logger.info("No useful contexts were found in this iteration.")

        context_texts = [ctx["context"] for ctx in aggregated_contexts]
        new_search_queries = get_new_search_queries(query, all_search_queries, context_texts)
        if not new_search_queries:
            logger.info("LLM indicated no further search is required.")
            break
        all_search_queries.extend(new_search_queries)
        iteration += 1

    if not aggregated_contexts:
        return "未能检索到与问题高度相关的公开资料。"

    blocks = []
    for ctx in aggregated_contexts:
        snippet = ctx["context"].strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        block = (
            f"### {ctx['title']}\n"
            f"- 链接：{ctx['url']}\n"
            f"- 检索词：{ctx['query']}\n\n"
            f"{snippet}"
        )
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def get_images(query: str) -> Dict[str, str]:
    """获取图片链接和描述"""
    logger.info("Searching for images for query: %s", query)
    results = web_search(query, top_k=4, categories="images")
    output: Dict[str, str] = {}
    for item in results:
        img_src = item.get("img_src")
        if not img_src:
            continue
        description = get_images_description(img_src)
        output[img_src] = description
    return output


if __name__ == "__main__":
    mcp.run()
