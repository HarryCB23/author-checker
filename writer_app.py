import streamlit as st
import requests
import time
import pandas as pd
import re
import json

st.set_page_config(layout="wide", page_title="Author Quality Evaluator")

# --- Configuration & API Endpoints ---
try:
    API_USERNAME = st.secrets["API_USERNAME"]
    API_PASSWORD = st.secrets["API_PASSWORD"]
except KeyError as e:
    st.error(f"Missing API secret: {e}. Please configure secrets in Streamlit Cloud dashboard.")
    st.stop()

DATAFORSEO_ORGANIC_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"

UK_PUBLISHER_DOMAINS = [
    "thetimes.com", "theguardian.com", "bbc.co.uk", "express.co.uk",
    "standard.co.uk", "dailymail.co.uk", "independent.co.uk", "thesun.co.uk",
    "mirror.co.uk", "metro.co.uk", "gbnews.com", "telegraph.co.uk"
]
EXCLUDED_GENERIC_DOMAINS_REGEX = [
    r"wikipedia\.org", r"linkedin\.com", r"twitter\.com", r"x\.com",
    r"facebook\.com", r"instagram\.com", r"youtube\.com", r"pinterest\.com",
    r"tiktok\.com", r"medium\.com", r"quora\.com", r"reddit\.com",
    r"threads\.net", r"amazon\.",
    r"audible\.com", r"audible\.com\.au",
    r"goodreads\.com", r"imdb\.com", r"substack\.com",
    r"acast\.com", r"apple\.com", r"spotify\.com", r"yahoo\.com",
    r"wordpress\.com", r"msn\.com", r"bylinetimes\.com", r"pressreader\.com",
    r"champions-speakers\.co\.uk", r"thriftbooks\.com", r"abebooks\.com",
    r"researchgate\.net", r"prowly\.com", r"shutterstock\.com",
    r"brainyquote\.com", r"fantasticfiction\.com", r"addall\.com",
    r"waterstones\.com", r"penguinrandomhouse\.com", r"penguin\.co\.uk",
    r"barr\.com", r"american\.edu", r"ashenden\.org",
    r"arrse\.co\.uk", r"mumsnet\.com",
    r"ebay\.com", r"pangobooks\.com", r"gettyimages\.co.uk",
    r"socialistworker\.co\.uk", r"newstatesman\.com", r"spectator\.co.uk",
    r"echo-news\.co\.uk", r"times-series\.co.uk", r"thenational\.scot", r"oxfordmail\.co.uk",
    r"moneyweek\.com", r"politeia\.co.uk", r"theweek\.com",
    r"innertemplelibrary\.com", r"san\.com", r"unherd\.com", r"padstudio\.co\.uk",
    r"deepsouthmedia\.co\.uk", r"dorsetchamber\.co\.uk", r"mattrossphysiotherapy\.co\.uk",
    r"company-information\.service\.gov\.uk", r"infogo\.gov\.on\.ca"
]

DEBUG = True

def debug_section(title, obj):
    if DEBUG:
        st.markdown(f"**DEBUG: {title}**")
        st.json(obj)

@st.cache_data(ttl=3600)
def make_dataforseo_call(payload):
    try:
        if not isinstance(payload, list):
            payload = [payload]
        for p in payload:
            if "limit" not in p:
                p["limit"] = 20
        response = requests.post(DATAFORSEO_ORGANIC_URL, auth=(API_USERNAME, API_PASSWORD), json=payload, timeout=60)
        response.raise_for_status()
        response_json = response.json()
        return response_json
    except Exception as e:
        st.error(f"API Call Error: {e}")
        return {"error": str(e)}

def extract_items_from_tasks(response, keyword):
    try:
        for task in response.get("tasks", []):
            task_kw = task.get("data", {}).get("keyword", "").replace('"','').strip().lower()
            if task_kw == keyword.replace('"','').strip().lower():
                results = task.get("result")
                if results and isinstance(results, list):
                    for result in results:
                        if isinstance(result, dict) and "items" in result and isinstance(result["items"], list):
                            return result["items"]
        # Fallback: return any items found
        for task in response.get("tasks", []):
            results = task.get("result")
            if results and isinstance(results, list):
                for result in results:
                    if isinstance(result, dict) and "items" in result and isinstance(result["items"], list):
                        return result["items"]
    except Exception as e:
        st.error(f"Error extracting items: {e}")
    return []

def extract_se_results_count(response, keyword):
    try:
        for task in response.get("tasks", []):
            task_kw = task.get("data", {}).get("keyword", "").replace('"','').strip().lower()
            results = task.get("result")
            if results and isinstance(results, list):
                if task_kw == keyword.replace('"','').strip().lower():
                    for result in results:
                        if isinstance(result, dict):
                            return int(result.get("se_results_count", 0))
        # fallback
        for task in response.get("tasks", []):
            results = task.get("result")
            if results and isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        return int(result.get("se_results_count", 0))
    except Exception as e:
        st.error(f"Error extracting se_results_count: {e}")
    return 0

def check_wikipedia(author):
    keyword = f'"{author}" site:wikipedia.org'
    payload = { "keyword": keyword, "language_code": "en", "location_name": "United Kingdom", "device": "desktop" }
    response = make_dataforseo_call(payload)
    items = extract_items_from_tasks(response, keyword)
    for item in items:
        if item.get("domain", "") == "en.wikipedia.org":
            return True, item.get("url")
    return False, None

def check_knowledge_panel(author):
    keyword = f'"{author}"'
    payload = { "keyword": keyword, "language_code": "en", "location_name": "United Kingdom", "device": "desktop" }
    response = make_dataforseo_call(payload)
    items = extract_items_from_tasks(response, keyword)
    for item in items:
        if item.get("type") == "knowledge_graph":
            return True, item.get("title"), item.get("description", "")
    return False, None, None

def extract_perspectives(author):
    keyword = f'"{author}"'
    payload = { "keyword": keyword, "language_code": "en", "location_name": "United Kingdom", "device": "desktop" }
    response = make_dataforseo_call(payload)
    items = extract_items_from_tasks(response, keyword)
    perspectives_domains = set()
    for item in items:
        if item.get("type") == "perspectives":
            for perspective_item in item.get("items", []):
                if perspective_item.get("domain"):
                    perspectives_domains.add(perspective_item["domain"])
    return sorted(list(perspectives_domains))

def extract_ai_overview(author):
    keyword = f'"{author}"'
    payload = { "keyword": keyword, "language_code": "en", "location_name": "United Kingdom", "device": "desktop" }
    response = make_dataforseo_call(payload)
    items = extract_items_from_tasks(response, keyword)
    for item in items:
        if item.get("type") == "ai_overview":
            ai_overview_content = item.get("ai_overview", {})
            if ai_overview_content.get("summary"):
                ai_overview_summary = ai_overview_content["summary"].strip()
                return ai_overview_summary[:500] + "..." if len(ai_overview_summary) > 500 else ai_overview_summary
            elif ai_overview_content.get("items"):
                ai_overview_parts = [ip.get("text") for ip in ai_overview_content["items"] if ip.get("text")]
                ai_overview_summary = " ".join(ai_overview_parts).strip()
                return ai_overview_summary[:500] + "..." if len(ai_overview_summary) > 500 else ai_overview_summary
            elif ai_overview_content.get("asynchronous_ai_overview"):
                return "AI Overview present (content loading dynamically)."
    return "N/A"

def check_google_scholar_citations(author):
    keyword = f'"{author}" "cited by" site:scholar.google.com'
    payload = { "keyword": keyword, "language_code": "en", "location_name": "United Kingdom", "device": "desktop" }
    response = make_dataforseo_call(payload)
    items = extract_items_from_tasks(response, keyword)
    count = 0
    for item in items:
        if item.get("domain", "") == "scholar.google.com":
            count += 1
    return count

@st.cache_data(ttl=3600)
def analyze_topical_serp(author: str, topic: str):
    topical_authority_serp_count = 0
    topical_authority_ratio = 0.0
    ai_overview_summary = extract_ai_overview(author)
    perspectives_domains = extract_perspectives(author)
    if topic:
        author_topic_query_str = f'"{author}" AND "{topic}"'
        topic_query_str = f'"{topic}"'
        payloads = [
            {"keyword": author_topic_query_str, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"},
            {"keyword": topic_query_str, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
        ]
        batch_data_response = make_dataforseo_call(payloads)
        topical_authority_serp_count = extract_se_results_count(batch_data_response, author_topic_query_str)
        total_topic_results_count = extract_se_results_count(batch_data_response, topic_query_str)
        if total_topic_results_count > 0:
            topical_authority_ratio = (topical_authority_serp_count / total_topic_results_count) * 100
    return (
        topical_authority_serp_count, 
        topical_authority_ratio, 
        ai_overview_summary,
        sorted(list(perspectives_domains)),
    )

@st.cache_data(ttl=3600)
def get_author_associated_brands(author: str):
    search_query = f'"{author}"'
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)
    all_associated_domains = set()
    matched_uk_publishers = set()
    items = extract_items_from_tasks(data, search_query)
    for item in items:
        if item.get("type") == "organic" and "domain" in item:
            domain = item["domain"]
            if not any(re.search(pattern, domain) for pattern in EXCLUDED_GENERIC_DOMAINS_REGEX):
                all_associated_domains.add(domain)
            if domain in UK_PUBLISHER_DOMAINS:
                matched_uk_publishers.add(domain)
    return sorted(list(all_associated_domains)), sorted(list(matched_uk_publishers))

def calculate_quality_score(
    has_kp: bool,
    has_wiki: bool,
    topical_authority_ratio: float,
    scholar_citations_count: int,
    linkedin_followers: int,
    x_followers: int,
    instagram_followers: int,
    tiktok_followers: int,
    facebook_followers: int,
    general_matched_uk_publishers_count: int,
    topical_matched_uk_publishers_count: int,
    ai_overview_present: bool,
    has_perspectives: bool
) -> int:
    score = 0
    if has_kp: score += 15
    if has_wiki: score += 10
    if ai_overview_present: score += 10
    if has_perspectives: score += 8
    score += min(int(topical_authority_ratio * 1), 15)
    score += min(scholar_citations_count // 5, 10)
    total_social_followers = linkedin_followers + x_followers + instagram_followers + tiktok_followers + facebook_followers
    score += min(total_social_followers // 10000, 10)
    general_matched_uk_publishers_score = min(general_matched_uk_publishers_count * 1, 5)
    score += general_matched_uk_publishers_score
    topical_matched_uk_publishers_score = min(topical_matched_uk_publishers_count * 2, 10)
    score += topical_matched_uk_publishers_score
    return max(0, min(score, 100))

def safe_int(val):
    try:
        out = pd.to_numeric(val, errors='coerce')
        return int(out) if not pd.isnull(out) and out is not None else 0
    except Exception:
        return 0

st.title("✍️ The Telegraph Recommended: Author Quality Evaluator")
st.markdown("---")
st.markdown("""
This tool helps assess the perceived authority and expertise of authors based on key online signals,
supporting the recruitment of high-quality freelancers for The Telegraph Recommended.
""")

if 'single_author_display_results' not in st.session_state:
    st.session_state['single_author_display_results'] = None
if 'bulk_analysis_results_df' not in st.session_state:
    st.session_state['bulk_analysis_results_df'] = None
if 'triggered_single_analysis' not in st.session_state:
    st.session_state['triggered_single_analysis'] = False
if 'triggered_bulk_analysis' not in st.session_state:
    st.session_state['triggered_bulk_analysis'] = False

with st.sidebar:
    st.header("Author Evaluation Inputs")
    st.subheader("Individual Author Analysis")
    st.markdown("Enter details to analyze a single author.")
    with st.expander("Expand for Single Author Input"):
        single_author_name = st.text_input("Author Name:", key="single_author_name_input", help="Full name of the author.")
        single_keyword_topic = st.text_input("Relevant Topic/Keyword for Expertise:", key="single_keyword_topic_input")
        single_author_url = st.text_input("Optional: Author Profile URL (e.g., Telegraph, personal site):", key="single_author_url_input")
        st.markdown("**Manual Social Media Followers:**")
        col_social1, col_social2 = st.columns(2)
        with col_social1:
            single_linkedin_followers = st.number_input("LinkedIn:", min_value=0, value=0, step=100, key="single_linkedin_followers_input")
            single_instagram_followers = st.number_input("Instagram:", min_value=0, value=0, step=100, key="single_instagram_followers_input")
            single_facebook_followers = st.number_input("Facebook:", min_value=0, value=0, step=100, key="single_facebook_followers_input")
        with col_social2:
            single_x_followers = st.number_input("X (Twitter):", min_value=0, value=0, step=100, key="single_x_followers_input")
            single_tiktok_followers = st.number_input("TikTok:", min_value=0, value=0, step=100, key="single_tiktok_followers_input")
        if st.button("Analyze Single Author", use_container_width=True):
            if single_author_name:
                with st.spinner(f"Analyzing '{single_author_name}'... This may take a moment due to API calls."):
                    wiki_exists, wiki_url = check_wikipedia(single_author_name)
                    kp_exists, kp_title, kp_desc = check_knowledge_panel(single_author_name)
                    perspectives_domains = extract_perspectives(single_author_name)
                    ai_overview_summary = extract_ai_overview(single_author_name)
                    scholar_citations_count = check_google_scholar_citations(single_author_name)
                    topical_authority_serp_count, topical_authority_ratio, topical_ai_overview, topical_perspectives_domains = analyze_topical_serp(single_author_name, single_keyword_topic)
                    all_associated_domains_general, matched_uk_publishers_general = get_author_associated_brands(single_author_name)
                    has_perspectives = len(perspectives_domains) > 0
                    has_topical_perspectives = len(topical_perspectives_domains) > 0
                    topical_matched_uk_publishers_count = len([d for d in all_associated_domains_general if d in UK_PUBLISHER_DOMAINS])
                    quality_score = calculate_quality_score(
                        kp_exists, wiki_exists, topical_authority_ratio,
                        scholar_citations_count, single_linkedin_followers, single_x_followers,
                        single_instagram_followers, single_tiktok_followers, single_facebook_followers,
                        len(matched_uk_publishers_general),
                        topical_matched_uk_publishers_count,
                        ai_overview_summary != "N/A" and "content loading dynamically" not in ai_overview_summary,
                        has_perspectives
                    )
                    st.session_state['single_author_display_results'] = pd.DataFrame([{
                        "Author": single_author_name,
                        "Keyword": single_keyword_topic,
                        "Author_URL": single_author_url,
                        "Quality_Score": quality_score,
                        "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
                        "KP_Title": kp_title if kp_exists else "N/A",
                        "KP_Description": kp_desc if kp_exists else "N/A",
                        "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
                        "Wikipedia_URL": wiki_url if wiki_exists else "N/A",
                        "AI_Overview_Summary": ai_overview_summary,
                        "Topical_Authority_SERP_Count": topical_authority_serp_count,
                        "Topical_Authority_Ratio": topical_authority_ratio,
                        "Has_Perspectives": "✅ Yes" if has_perspectives else "❌ No",
                        "Perspectives_Domains": ", ".join(perspectives_domains) if perspectives_domains else "None",
                        "Scholar_Citations_Count": scholar_citations_count,
                        "General_Matched_UK_Publishers": ", ".join(matched_uk_publishers_general) if matched_uk_publishers_general else "None",
                        "All_Associated_Domains_General": ", ".join(all_associated_domains_general),
                        "LinkedIn_Followers": single_linkedin_followers,
                        "X_Followers": single_x_followers,
                        "Instagram_Followers": single_instagram_followers,
                        "TikTok_Followers": single_tiktok_followers,
                        "Facebook_Followers": single_facebook_followers,
                    }])
                    st.session_state['triggered_single_analysis'] = True
            else:
                st.warning("Please enter an author name to analyze.")

    st.markdown("---")
    st.subheader("Bulk Author Analysis (CSV Upload)")
    st.markdown("""
    Upload a CSV file with the following columns:
    - **Author** (required): Full name of the author.
    - **Keyword** (optional): Relevant topic for expertise.
    - **Author_URL** (optional): Author's profile URL.
    - **LinkedIn_Followers** (optional): Manual LinkedIn follower count.
    - **X_Followers** (optional): Manual X (Twitter) follower count.
    - **Instagram_Followers** (optional): Manual Instagram follower count.
    - **TikTok_Followers** (optional): Manual TikTok follower count.
    - **Facebook_Followers** (optional): Manual Facebook follower count.
    """)
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="bulk_upload_file")
    if uploaded_file is not None:
        try:
            bulk_data = pd.read_csv(uploaded_file)
            if "Author" not in bulk_data.columns:
                st.error("The CSV file must contain an 'Author' column.")
                st.session_state['bulk_data_to_process'] = None
            else:
                st.success("CSV uploaded successfully. Click 'Run Bulk Analysis' below.")
                st.session_state['bulk_data_to_process'] = bulk_data
        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
            st.session_state['bulk_data_to_process'] = None
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.session_state['bulk_data_to_process'] = None
    else:
        st.session_state['bulk_data_to_process'] = None

    if st.button("Run Bulk Analysis", use_container_width=True, disabled=(st.session_state.get('bulk_data_to_process') is None)):
        if st.session_state['bulk_data_to_process'] is not None:
            st.session_state['triggered_bulk_analysis'] = True

st.header("Analysis Results")
st.markdown("---")

if st.session_state['triggered_single_analysis'] and st.session_state['single_author_display_results'] is not None:
    st.subheader("Individual Author Analysis Results")
    st.dataframe(st.session_state['single_author_display_results'], use_container_width=True)
    st.session_state['triggered_single_analysis'] = False
    st.session_state['single_author_display_results'] = None

elif st.session_state['triggered_bulk_analysis'] and st.session_state['bulk_data_to_process'] is not None:
    st.subheader("Bulk Author Analysis Results")
    bulk_data = st.session_state['bulk_data_to_process']
    results = []
    total_authors = len(bulk_data)
    progress_bar = st.progress(0)
    status_text = st.empty()
    st_spinner_placeholder = st.empty()
    with st_spinner_placeholder.container():
        with st.spinner("Processing your bulk request... This may take a moment due to API calls."):
            for index, row in bulk_data.iterrows():
                author = str(row["Author"]).strip()
                keyword = str(row["Keyword"]).strip() if "Keyword" in bulk_data.columns and pd.notna(row["Keyword"]) else ""
                linkedin_followers = safe_int(row.get("LinkedIn_Followers", 0))
                x_followers = safe_int(row.get("X_Followers", 0))
                instagram_followers = safe_int(row.get("Instagram_Followers", 0))
                tiktok_followers = safe_int(row.get("TikTok_Followers", 0))
                facebook_followers = safe_int(row.get("Facebook_Followers", 0))
                author_url = str(row["Author_URL"]).strip() if "Author_URL" in bulk_data.columns and pd.notna(row["Author_URL"]) else ""
                wiki_exists, wiki_url = check_wikipedia(author)
                kp_exists, kp_title, kp_desc = check_knowledge_panel(author)
                perspectives_domains = extract_perspectives(author)
                ai_overview_summary = extract_ai_overview(author)
                scholar_citations_count = check_google_scholar_citations(author)
                topical_authority_serp_count, topical_authority_ratio, topical_ai_overview, topical_perspectives_domains = analyze_topical_serp(author, keyword)
                all_associated_domains_general, matched_uk_publishers_general = get_author_associated_brands(author)
                has_perspectives = len(perspectives_domains) > 0
                has_topical_perspectives = len(topical_perspectives_domains) > 0
                topical_matched_uk_publishers_count = len([d for d in all_associated_domains_general if d in UK_PUBLISHER_DOMAINS])
                quality_score = calculate_quality_score(
                    kp_exists, wiki_exists, topical_authority_ratio,
                    scholar_citations_count, linkedin_followers, x_followers,
                    instagram_followers, tiktok_followers, facebook_followers,
                    len(matched_uk_publishers_general),
                    topical_matched_uk_publishers_count,
                    ai_overview_summary != "N/A" and "content loading dynamically" not in ai_overview_summary,
                    has_perspectives
                )
                results.append({
                    "Author": author,
                    "Keyword": keyword,
                    "Author_URL": author_url,
                    "Quality_Score": quality_score,
                    "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
                    "KP_Title": kp_title if kp_exists else "N/A",
                    "KP_Description": kp_desc if kp_exists else "N/A",
                    "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
                    "Wikipedia_URL": wiki_url if wiki_exists else "N/A",
                    "AI_Overview_Summary": ai_overview_summary,
                    "Topical_Authority_SERP_Count": topical_authority_serp_count,
                    "Topical_Authority_Ratio": topical_authority_ratio,
                    "Has_Perspectives": "✅ Yes" if has_perspectives else "❌ No",
                    "Perspectives_Domains": ", ".join(perspectives_domains) if perspectives_domains else "None",
                    "Scholar_Citations_Count": scholar_citations_count,
                    "General_Matched_UK_Publishers": ", ".join(matched_uk_publishers_general) if matched_uk_publishers_general else "None",
                    "All_Associated_Domains_General": ", ".join(all_associated_domains_general),
                    "LinkedIn_Followers": linkedin_followers,
                    "X_Followers": x_followers,
                    "Instagram_Followers": instagram_followers,
                    "TikTok_Followers": tiktok_followers,
                    "Facebook_Followers": facebook_followers,
                })
                progress_bar.progress((index + 1) / total_authors)
                time.sleep(1)
    results_df = pd.DataFrame(results)
    st.session_state['bulk_analysis_results_df'] = results_df
    st.session_state['triggered_bulk_analysis'] = False
    st.session_state['bulk_data_to_process'] = None
    st.markdown("### High-Level Summary")
    st.dataframe(results_df, use_container_width=True)
    csv_output = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv_output,
        file_name="author_expertise_results.csv",
        mime="text/csv",
        help="Download the analysis results as a CSV file."
    )
    progress_bar.empty()
    status_text.empty()
    st_spinner_placeholder.empty()
    st.success("Bulk analysis complete!")
else:
    st.info("Use the sidebar to input author details for individual analysis, or upload a CSV for bulk processing. Results will appear here.")
