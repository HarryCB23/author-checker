import streamlit as st
import requests
import pandas as pd
import re
import time

# --- API Credentials ---
try:
    API_USERNAME = st.secrets["API_USERNAME"]
    API_PASSWORD = st.secrets["API_PASSWORD"]
except KeyError as e:
    st.error(f"Missing API secret: {e}. Please configure secrets in Streamlit Cloud dashboard.")
    st.stop()

DATAFORSEO_ORGANIC_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"

EXCLUDED_GENERIC_DOMAINS_REGEX = [
    r"wikipedia\.org", r"linkedin\.com", r"twitter\.com", r"x\.com",
    r"facebook\.com", r"instagram\.com", r"youtube\.com", r"pinterest\.com",
    r"tiktok\.com", r"medium\.com", r"quora\.com", r"reddit\.com"
]

@st.cache_data(ttl=3600)
def make_dataforseo_call(payload: dict) -> dict:
    try:
        if isinstance(payload, dict) and "limit" not in payload:
            payload["limit"] = 20
        elif isinstance(payload, list):
            for p in payload:
                if "limit" not in p:
                    p["limit"] = 20

        if not isinstance(payload, list):
            payload = [payload]

        response = requests.post(DATAFORSEO_ORGANIC_URL, auth=(API_USERNAME, API_PASSWORD), json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as req_err:
        return {"error": f"DataForSEO API request error: {req_err}"}
    except Exception as e:
        return {"error": f"An unexpected error occurred during DataForSEO call: {e}"}

# --- Knowledge Panel Detection ---
def check_knowledge_panel(author: str) -> tuple[bool, str]:
    search_query = f'"{author}"'
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)

    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                for result_block in task["result"]:
                    if "items" in result_block:
                        for item in result_block["items"]:
                            if item.get("type") == "knowledge_graph":
                                kp_title = item.get("title", "").lower()
                                if kp_title == author.lower() or author.lower() in item.get("description", "").lower():
                                    return True, "Knowledge Panel found"
                                if "items" in item:
                                    for sub_item in item["items"]:
                                        if author.lower() in sub_item.get("text", "").lower() or author.lower() in sub_item.get("title", "").lower():
                                            return True, "Knowledge Panel found (sub-item)"
                    if result_block.get("items"):
                        for nested_item in result_block["items"]:
                            if nested_item.get("type") == "knowledge_graph":
                                kp_title = nested_item.get("title", "").lower()
                                if kp_title == author.lower() or author.lower() in nested_item.get("description", "").lower():
                                    return True, "Knowledge Panel found (nested item)"
                                if "items" in nested_item:
                                    for sub_item_nested in nested_item["items"]:
                                        if author.lower() in sub_item_nested.get("text", "").lower() or author.lower() in sub_item_nested.get("title", "").lower():
                                            return True, "Knowledge Panel found (nested item sub-item)"
    return False, "No Knowledge Panel found"

# --- Topical SERP Analysis ---
def analyze_topical_serp(author: str, topic: str) -> tuple[int, float, str, list[str], list[str], list[str]]:
    author_only_query = f'"{author}"'
    payloads = [{"keyword": author_only_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}]

    if topic:
        author_topic_query_str = f'"{author}" AND "{topic}"'
        topic_query_str = f'"{topic}"'
        payloads.append({"keyword": author_topic_query_str, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"})
        payloads.append({"keyword": topic_query_str, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"})

    batch_data_response = make_dataforseo_call(payloads)

    author_topic_data_task = {}
    topic_only_data_task = {}
    author_only_data_task = {}

    if batch_data_response and "tasks" in batch_data_response:
        for task in batch_data_response["tasks"]:
            keyword_in_task = task.get("keyword")
            if keyword_in_task == f'"{author}" AND "{topic}"':
                author_topic_data_task = task
            elif keyword_in_task == f'"{topic}"':
                topic_only_data_task = task
            elif keyword_in_task == f'"{author}"':
                author_only_data_task = task

    author_topic_results_count = 0
    total_topic_results_count = 0
    ai_overview_summary = "N/A"
    topical_authority_ratio = 0.0
    top_stories_mentions = []
    topical_associated_domains = set()
    perspectives_domains = set()

    if "result" in author_only_data_task and author_only_data_task["result"]:
        for result_block in author_only_data_task["result"]:
            if result_block.get("type") == "perspectives" and result_block.get("items"):
                for perspective_item in result_block["items"]:
                    if perspective_item.get("domain"):
                        perspectives_domains.add(perspective_item["domain"])
            if result_block.get("items"):
                for nested_item in result_block["items"]:
                    if nested_item.get("type") == "perspectives" and nested_item.get("items"):
                        for perspective_item in nested_item["items"]:
                            if perspective_item.get("domain"):
                                perspectives_domains.add(perspective_item["domain"])

    if not topic:
        return 0, 0.0, "N/A", [], [], sorted(list(perspectives_domains))

    if "result" in author_topic_data_task and author_topic_data_task["result"]:
        for result_item in author_topic_data_task["result"]:
            if result_item.get("type") == "organic" and result_item.get("serp"):
                author_topic_results_count = result_item["serp"].get("results_count", 0)
                break

    if "result" in topic_only_data_task and topic_only_data_task["result"]:
        for result_item in topic_only_data_task["result"]:
            if result_item.get("type") == "organic" and result_item.get("serp"):
                total_topic_results_count = result_item["serp"].get("results_count", 0)
                break

    if total_topic_results_count > 0:
        topical_authority_ratio = (author_topic_results_count / total_topic_results_count) * 100

    if "result" in author_topic_data_task and author_topic_data_task["result"]:
        for result_item in author_topic_data_task["result"]:
            if result_item.get("type") == "ai_overview" and result_item.get("ai_overview"):
                ai_content = result_item["ai_overview"]
                if ai_content.get("summary"):
                    ai_overview_summary = ai_content["summary"].strip()
                elif ai_content.get("items"):
                    ai_parts = [item_part.get("text", "") for item_part in ai_content["items"] if item_part.get("text")]
                    ai_overview_summary = " ".join(ai_parts).strip() if ai_parts else "N/A"
                elif ai_content.get("asynchronous_ai_overview"):
                    ai_overview_summary = "AI Overview present (content loading dynamically)."
                if ai_overview_summary != "N/A" and len(ai_overview_summary) > 500:
                    ai_overview_summary = ai_overview_summary[:500] + "..."
                break

            if result_item.get("type") == "top_stories" and result_item.get("items"):
                for news_item in result_item["items"]:
                    if author.lower() in news_item.get("title", "").lower() or author.lower() in news_item.get("description", "").lower():
                        top_stories_mentions.append(news_item.get("domain"))

            if result_item.get("items"):
                for item in result_item["items"]:
                    if item.get("type") == "organic" and "domain" in item:
                        domain = item["domain"]
                        if (author.lower() in item.get("title", "").lower() or author.lower() in item.get("description", "").lower()):
                            if not any(re.search(pattern, domain) for pattern in EXCLUDED_GENERIC_DOMAINS_REGEX):
                                topical_associated_domains.add(domain)

    return (author_topic_results_count, topical_authority_ratio, ai_overview_summary,
            sorted(list(set(top_stories_mentions))), sorted(list(topical_associated_domains)), sorted(list(perspectives_domains)))
