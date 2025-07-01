import streamlit as st
import requests
import time
import pandas as pd
import re

st.set_page_config(layout="wide", page_title="Author Quality Evaluator")

# --- Configuration & API Endpoints ---
try:
    API_USERNAME = st.secrets["API_USERNAME"]
    API_PASSWORD = st.secrets["API_PASSWORD"]
except KeyError as e:
    st.error(f"Missing API secret: {e}. Please configure secrets in Streamlit Cloud dashboard.")
    st.stop()

DATAFORSEO_ORGANIC_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"

UK_PUBLISHER_DOMAINS = [
    "thetimes.com",
    "theguardian.com",
    "bbc.co.uk",
    "express.co.uk",
    "standard.co.uk",
    "dailymail.co.uk",
    "independent.co.uk",
    "thesun.co.uk",
    "mirror.co.uk",
    "metro.co.uk",
    "gbnews.com",
    "telegraph.co.uk" # Ensure Telegraph itself is considered a major publisher
]
EXCLUDED_GENERIC_DOMAINS_REGEX = [
    r"wikipedia\.org", r"linkedin\.com", r"twitter\.com", r"x\.com", # X (Twitter) and Twitter
    r"facebook\.com", r"instagram\.com", r"youtube\.com", r"pinterest\.com",
    r"tiktok\.com", r"medium\.com", r"quora\.com", r"reddit\.com",
    r"threads\.net", r"amazon\.", # Keep Amazon excluded
    r"audible\.com", r"audible\.com\.au", # Audible variations
    r"goodreads\.com", r"imdb\.com", r"substack\.com", # Common book/media/blog platforms
    r"acast\.com", r"apple\.com", r"spotify\.com", r"yahoo\.com", # Podcast/news platforms
    r"wordpress\.com", r"msn\.com", r"bylinetimes\.com", r"pressreader\.com", # Blogging/news platforms
    r"champions-speakers\.co\.uk", r"thriftbooks\.com", r"abebooks\.com", # Speaker/book sites
    r"researchgate\.net", r"prowly\.com", r"shutterstock\.com", # Academic/professional sites
    r"brainyquote\.com", r"fantasticfiction\.com", r"addall\.com", # Quote/book sites
    r"waterstones\.com", r"penguinrandomhouse\.com", r"penguin\.co\.uk", # Publishers (some are generic for author profiles)
    r"barr\.com", r"american\.edu", r"ashenden\.org", # Specific company/org sites
    r"arrse\.co\.uk", r"mumsnet\.com", # Forums/community
    r"ebay\.com", r"pangobooks\.com", r"gettyimages\.co\.uk", # Marketplace/image
    r"socialistworker\.co\.uk", r"newstatesman\.com", r"spectator\.co\.uk", # Political/commentary (not primary news)
    r"echo-news\.co\.uk", r"times-series\.co\.uk", r"thenational\.scot", r"oxfordmail\.co\.uk", # Regional/specific news
    r"moneyweek\.com", r"politeia\.co\.uk", r"theweek\.com", # Finance/political
    r"innertemplelibrary\.com", r"san\.com", r"unherd\.com", "padstudio.co.uk", # Other commentary/blogs
    r"deepsouthmedia\.co\.uk", r"dorsetchamber\.co\.uk", r"mattrossphysiotherapy\.co\.uk", # Local/specific business
    r"company-information\.service\.gov\.uk", r"infogo\.gov\.on\.ca" # Government/company info
]

# --- Helper Functions ---

@st.cache_data(ttl=3600)
def make_dataforseo_call(payload: dict) -> dict:
    """Helper function to make DataForSEO API requests."""
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
    except requests.exceptions.HTTPError as http_err:
        error_msg = f"DataForSEO HTTP error ({response.status_code}): {http_err} - {response.text}"
        return {"error": error_msg}
    except requests.exceptions.RequestException as req_err:
        error_msg = f"DataForSEO API request error: {req_err}"
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred during DataForSEO call: {e}"
        return {"error": error_msg}

@st.cache_data(ttl=3600)
def check_knowledge_panel(author: str) -> tuple[bool, str]:
    """Checks DataForSEO for a Google Knowledge Panel by searching ONLY the author's name."""
    search_query = f'"{author}"' # Essential: search only author name for KP
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)

    # st.write(f"DEBUG KP Query: {search_query}") # Debugging
    # if data: st.json(data) # Debugging raw DataForSEO response

    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                for result_item in task["result"]: # Iterate through top-level items in 'result'
                    # Check if the current result_item itself is the knowledge_graph block
                    if result_item.get("type") == "knowledge_graph":
                        kp_title = result_item.get("title", "").lower()
                        # Be slightly flexible with title match or check description/subtitle
                        if kp_title == author.lower() or author.lower() in result_item.get("description", "").lower():
                            return True, "Knowledge Panel found"
                        # Sometimes the KP is on the right, but its full title is not exactly the author name,
                        # but one of its sub-items will clearly identify the author.
                        if "items" in result_item:
                            for sub_item in result_item["items"]:
                                if sub_item.get("type") in ["knowledge_graph_description_item", "knowledge_graph_row_item"]:
                                    if author.lower() in sub_item.get("text", "").lower() or author.lower() in sub_item.get("title", "").lower():
                                        return True, "Knowledge Panel found (via sub-item content)"
                
                # Fallback: if the KP is not a direct top-level block, but buried as an item in another type (less common but possible)
                for result_block in task["result"]:
                    if "items" in result_block:
                        for item_nested in result_block["items"]:
                            if item_nested.get("type") == "knowledge_graph":
                                kp_title = item_nested.get("title", "").lower()
                                if kp_title == author.lower() or author.lower() in item_nested.get("description", "").lower():
                                    return True, "Knowledge Panel found (nested)"
                                if "items" in item_nested: # Check sub-items if nested KP has them
                                    for sub_item_nested in item_nested["items"]:
                                        if sub_item_nested.get("type") in ["knowledge_graph_description_item", "knowledge_graph_row_item"]:
                                            if author.lower() in sub_item_nested.get("text", "").lower() or author.lower() in sub_item_nested.get("title", "").lower():
                                                return True, "Knowledge Panel found (nested via sub-item)"
        return False, "No Knowledge Panel found in SERP results"
    return False, data.get("error", "No data or task in DataForSEO response.")

@st.cache_data(ttl=3600)
def check_wikipedia(author: str) -> tuple[bool, str, str]:
    """Checks Wikipedia API for a page matching the author's name. Returns (found, message, URL)."""
    wiki_author_query = author.replace(' ', '_')
    wikipedia_api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={wiki_author_query}&prop=info&inprop=url&format=json"
    try:
        response = requests.get(wikipedia_api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        if "-1" not in pages:
            page_id = list(pages.keys())[0]
            wikipedia_url = pages[page_id].get("fullurl", "")
            return True, "Wikipedia page found", wikipedia_url
        return False, "No Wikipedia page found", ""
    except requests.exceptions.RequestException as e:
        return False, f"Wikipedia API error: {e}", ""
    except Exception as e:
        return False, f"An unexpected error occurred checking Wikipedia: {e}", ""

@st.cache_data(ttl=3600)
def analyze_topical_serp(author: str, topic: str) -> tuple[int, float, str, list[str], list[str], list[str]]:
    """
    Analyzes the SERP for 'author AND topic' and 'author' alone to get topical authority metrics,
    AI Overview, top stories, topic-specific associated brands, and perspectives.
    """
    author_only_query = f'"{author}"'

    payloads = []
    author_topic_query = ""
    topic_query = ""

    if topic: # Only add topic-related queries if a topic is provided
        author_topic_query = f'"{author}" AND "{topic}"'
        topic_query = f'"{topic}"'
        payloads.append({"keyword": author_topic_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"})
        payloads.append({"keyword": topic_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"})
    
    payloads.append({"keyword": author_only_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"})

    # Deduplicate payloads by keyword, ensuring all needed queries are sent
    seen_keywords = set()
    unique_payloads = []
    for p in payloads:
        if p["keyword"] not in seen_keywords:
            unique_payloads.append(p)
            seen_keywords.add(p["keyword"])
    
    batch_data_response = make_dataforseo_call(unique_payloads)

    # Initialize task data holders
    author_topic_data_task = {}
    topic_only_data_task = {}
    author_only_data_task = {} # This will hold data for `"Allison Pearson"` search

    if batch_data_response and "tasks" in batch_data_response:
        for task in batch_data_response["tasks"]:
            keyword_in_task = task.get("keyword") 
            if keyword_in_task == author_topic_query and "result" in task:
                author_topic_data_task = task
            elif keyword_in_task == topic_query and "result" in task:
                topic_only_data_task = task
            elif keyword_in_task == author_only_query and "result" in task:
                author_only_data_task = task

    # Initialize return values
    author_topic_results_count = 0
    total_topic_results_count = 0
    ai_overview_summary = "N/A"
    topical_authority_ratio = 0.0
    top_stories_mentions = []
    topical_associated_domains = set()
    perspectives_domains = set() # Always collected

    # --- Extract Perspectives Domains (from author_only_data_task and author_topic_data_task if available) ---
    # Check both data sources where perspectives might appear
    for data_source_task in [author_only_data_task, author_topic_data_task]:
        if "result" in data_source_task and data_source_task["result"]:
            for result_block in data_source_task["result"]: # Iterate through top-level result blocks
                if result_block.get("type") == "perspectives" and result_block.get("items"):
                    for perspective_item in result_block["items"]:
                        if perspective_item.get("domain"):
                            perspectives_domains.add(perspective_item["domain"])
    
    # If no topic is provided for topical authority calculation, return early for those fields
    if not topic:
        return 0, 0.0, "N/A", [], [], sorted(list(perspectives_domains))


    # --- Extract Topical Authority Results Count ---
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

    # --- Extract AI Overview ---
    if "result" in author_topic_data_task and author_topic_data_task["result"]:
        for result_item in author_topic_data_task["result"]:
            if result_item.get("type") == "ai_overview" and result_item.get("ai_overview"):
                ai_overview_content = result_item["ai_overview"]
                
                if ai_overview_content.get("summary"): 
                    ai_overview_summary = ai_overview_content["summary"].strip()
                elif ai_overview_content.get("items"):
                    ai_overview_parts = []
                    for item_part in ai_overview_content["items"]:
                        if item_part.get("text"):
                            ai_overview_parts.append(item_part["text"])
                    if ai_overview_parts:
                        ai_overview_summary = " ".join(ai_overview_parts).strip()
                elif ai_overview_content.get("asynchronous_ai_overview"):
                    ai_overview_summary = "AI Overview present (content loading dynamically)." 
                
                if ai_overview_summary != "N/A" and len(ai_overview_summary) > 500:
                    ai_overview_summary = ai_overview_summary[:500] + "..."
                break

    # --- Extract Top Stories Mentions and Topical Associated Domains ---
    if "result" in author_topic_data_task and author_topic_data_task["result"]:
        for result_item in author_topic_data_task["result"]:
            if result_item.get("type") == "top_stories" and result_item.get("items"):
                for news_item in result_item["items"]:
                    if (author.lower() in news_item.get("title", "").lower() or 
                        author.lower() in news_item.get("description", "").lower()):
                        top_stories_mentions.append(news_item.get("domain"))

            if "items" in result_item: # Check if 'items' array is in this result_item
                for item in result_item["items"]:
                    if item.get("type") == "organic" and "domain" in item:
                        domain = item["domain"]
                        if (author.lower() in item.get("title", "").lower() or 
                            author.lower() in item.get("description", "").lower()) and \
                           not any(re.search(pattern, domain) for pattern in EXCLUDED_GENERIC_DOMAINS_REGEX):
                            topical_associated_domains.add(domain)
                            
    return (author_topic_results_count, topical_authority_ratio, ai_overview_summary,
            sorted(list(set(top_stories_mentions))), sorted(list(topical_associated_domains)), sorted(list(perspectives_domains)))


@st.cache_data(ttl=3600)
def get_author_associated_brands(author: str) -> tuple[list[str], list[str]]:
    """
    Finds unique domains an author is generally associated with by searching their name.
    Separates into general associated domains and matches with predefined UK publishers.
    """
    search_query = f'"{author}"' # General search for author's overall brand presence
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)

    all_associated_domains = set()
    matched_uk_publishers = set()

    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                for result_block in task["result"]: # Changed to result_block for clarity
                    if "items" in result_block:
                        for item in result_block["items"]:
                            if item.get("type") == "organic" and "domain" in item:
                                domain = item["domain"]
                                if not any(re.search(pattern, domain) for pattern in EXCLUDED_GENERIC_DOMAINS_REGEX):
                                    all_associated_domains.add(domain)
                                if domain in UK_PUBLISHER_DOMAINS:
                                    matched_uk_publishers.add(domain)

    return sorted(list(all_associated_domains)), sorted(list(matched_uk_publishers))

@st.cache_data(ttl=3600)
def check_google_scholar_citations(author: str) -> int:
    """Counts search results for author citations on Google Scholar."""
    search_query = f'"{author}" "cited by" site:scholar.google.com'
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)
    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                for result in task["result"]:
                    return result.get("serp", {}).get("results_count", 0)
    return 0

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
    top_stories_mentions_count: int,
    topical_matched_uk_publishers_count: int,
    ai_overview_present: bool,
    has_perspectives: bool # Changed to boolean
) -> int:
    """Calculates a quality score based on various signals."""
    score = 0
    if has_kp: score += 15
    if has_wiki: score += 10
    if ai_overview_present: score += 10
    if has_perspectives: score += 8 # Points for presence of perspectives

    score += min(int(topical_authority_ratio * 1), 15)
    
    score += min(scholar_citations_count // 5, 10)

    total_social_followers = linkedin_followers + x_followers + instagram_followers + tiktok_followers + facebook_followers
    score += min(total_social_followers // 10000, 10)

    general_matched_uk_publishers_score = min(general_matched_uk_publishers_count * 1, 5)
    score += general_matched_uk_publishers_score

    topical_matched_uk_publishers_score = min(topical_matched_uk_publishers_count * 2, 10)
    score += topical_matched_uk_publishers_score

    if top_stories_mentions_count > 0:
        score += min(top_stories_mentions_count * 3, 10)

    return max(0, min(score, 100))

# --- Main Page Title ---
st.title("✍️ The Telegraph Recommended: Author Quality Evaluator")
st.markdown("---")
st.markdown("""
This tool helps assess the perceived authority and expertise of authors based on key online signals,
supporting the recruitment of high-quality freelancers for The Telegraph Recommended.
""")

# --- Initialize Session State for Results ---
if 'single_author_display_results' not in st.session_state:
    st.session_state['single_author_display_results'] = None
if 'bulk_analysis_results_df' not in st.session_state:
    st.session_state['bulk_analysis_results_df'] = None
if 'triggered_single_analysis' not in st.session_state:
    st.session_state['triggered_single_analysis'] = False
if 'triggered_bulk_analysis' not in st.session_state:
    st.session_state['triggered_bulk_analysis'] = False

# --- Sidebar ---
with st.sidebar:
    st.header("Author Evaluation Inputs")

    # --- Individual Author Evaluation ---
    st.subheader("Individual Author Analysis")
    st.markdown("Enter details to analyze a single author.")
    with st.expander("Expand for Single Author Input"):
        single_author_name = st.text_input("Author Name:", key="single_author_name_input", help="Full name of the author.")
        single_keyword_topic = st.text_input("Relevant Topic/Keyword for Expertise:", key="single_keyword_topic_input", help="""
            **Guidance for Keywords:**
            - **Be specific:** "smartwatch reviews" not just "watches".
            - **Focus on expertise:** "AI ethics" not just "AI".
            - **Combine terms (if complex):** "Dr. Jane Smith diabetes research".
            **Avoid overly broad terms** that might dilute relevance.
            """)
        single_author_url = st.text_input("Optional: Author Profile URL (e.g., Telegraph, personal site):", key="single_author_url_input", help="e.g., telegraph.co.uk/authors/jane-doe, or author's personal website.")
        
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
                    # API Calls
                    kp_exists, kp_details = check_knowledge_panel(single_author_name)
                    wiki_exists, wiki_details, wiki_url = check_wikipedia(single_author_name)
                    
                    # Consolidated topical SERP analysis
                    topical_authority_serp_count, topical_authority_ratio, ai_overview_summary, \
                        top_stories_mentions, topical_associated_domains, perspectives_domains = analyze_topical_serp(single_author_name, single_keyword_topic)

                    all_associated_domains_general, matched_uk_publishers_general = get_author_associated_brands(single_author_name)
                    scholar_citations_count = check_google_scholar_citations(single_author_name)
                    
                    has_perspectives = len(perspectives_domains) > 0 # Boolean for score and display
                    
                    quality_score = calculate_quality_score(
                        kp_exists, wiki_exists, topical_authority_ratio,
                        scholar_citations_count, single_linkedin_followers, single_x_followers,
                        single_instagram_followers, single_tiktok_followers, single_facebook_followers,
                        len(matched_uk_publishers_general), # General publishers
                        len(top_stories_mentions), # New
                        len([d for d in topical_associated_domains if d in UK_PUBLISHER_DOMAINS]), # Count topical publishers
                        ai_overview_summary != "N/A" and "content loading dynamically" not in ai_overview_summary, # AI overview exists and has content
                        has_perspectives # New
                    )

                    st.session_state['single_author_display_results'] = pd.DataFrame([{
                        "Author": single_author_name,
                        "Keyword": single_keyword_topic,
                        "Author_URL": single_author_url,
                        "Quality_Score": quality_score,
                        "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
                        "KP_Details": kp_details,
                        "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
                        "Wikipedia_Details": wiki_details,
                        "Wikipedia_URL": wiki_url if wiki_exists else "N/A",
                        "AI_Overview_Summary": ai_overview_summary,
                        "Topical_Authority_SERP_Count": topical_authority_serp_count,
                        "Topical_Authority_Ratio": topical_authority_ratio,
                        "Top_Stories_Mentions_Domains": ", ".join(top_stories_mentions) if top_stories_mentions else "None",
                        "Has_Perspectives": "✅ Yes" if has_perspectives else "❌ No", # New
                        "Perspectives_Domains": ", ".join(perspectives_domains) if perspectives_domains else "None", # New
                        "Scholar_Citations_Count": scholar_citations_count,
                        "General_Matched_UK_Publishers": ", ".join(matched_uk_publishers_general) if matched_uk_publishers_general else "None",
                        "Topic_Associated_Domains": ", ".join(topical_associated_domains) if topical_associated_domains else "None",
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

    # --- Bulk Author Evaluation (moved to sidebar) ---
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

# --- Main Content Area (Visualization) ---
st.header("Analysis Results")
st.markdown("---")

# Process and display single author results if triggered
if st.session_state['triggered_single_analysis'] and st.session_state['single_author_display_results'] is not None:
    st.subheader("Individual Author Analysis Results")
    
    def highlight_score_color_row(s):
        score_val = s['Quality_Score']
        if score_val >= 70:
            return ['background-color: #d4edda'] * len(s)
        elif score_val >= 40:
            return ['background-color: #ffeeba'] * len(s)
        else:
            return ['background-color: #f8d7da'] * len(s)
    
    def highlight_tick_cross_bg_cell(val):
        if '✅' in str(val):
            return 'background-color: #e0ffe0'
        elif '❌' in str(val):
            return 'background-color: #fff0f0'
        return ''

    def make_clickable_wikipedia(val):
        if val and val != "N/A":
            return f'<a href="{val}" target="_blank">Link</a>'
        return val

    styled_single_df = st.session_state['single_author_display_results'].style.apply(
        highlight_score_color_row, axis=1
    ).applymap(
        highlight_tick_cross_bg_cell,
        subset=['Has_Knowledge_Panel', 'Has_Wikipedia_Page', 'Has_Perspectives']
    ).format(make_clickable_wikipedia, subset=['Wikipedia_URL'], escape=False) \
    .format({
        'Topical_Authority_SERP_Count': "{:,.0f}",
        'Topical_Authority_Ratio': "{:.2f}%",
        'Scholar_Citations_Count': "{:,.0f}",
        'LinkedIn_Followers': "{:,.0f}",
        'X_Followers': "{:,.0f}",
        'Instagram_Followers': "{:,.0f}",
        'TikTok_Followers': "{:,.0f}",
        'Facebook_Followers': "{:,.0f}"
    })

    st.dataframe(styled_single_df, use_container_width=True)

    st.session_state['triggered_single_analysis'] = False
    st.session_state['single_author_display_results'] = None

# Process and display bulk analysis results if triggered
elif st.session_state['triggered_bulk_analysis'] and st.session_state['bulk_data_to_process'] is not None:
    st.subheader("Bulk Author Analysis Results")
    
    bulk_data = st.session_state['bulk_data_to_process']
    results = []
    total_authors = len(bulk_data)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    st_spinner_placeholder = st.empty()

    with st_spinner_placeholder.container():
        with st.spinner("Processing your bulk request... This may take some time due to API calls."):
            for index, row in bulk_data.iterrows():
                author = str(row["Author"]).strip()
                keyword = str(row["Keyword"]).strip() if "Keyword" in bulk_data.columns and pd.notna(row["Keyword"]) else ""
                
                linkedin_followers = pd.to_numeric(row.get("LinkedIn_Followers", 0), errors='coerce').fillna(0).astype(int)
                x_followers = pd.to_numeric(row.get("X_Followers", 0), errors='coerce').fillna(0).astype(int)
                instagram_followers = pd.to_numeric(row.get("Instagram_Followers", 0), errors='coerce').fillna(0).astype(int)
                tiktok_followers = pd.to_numeric(row.get("TikTok_Followers", 0), errors='coerce').fillna(0).astype(int)
                facebook_followers = pd.to_numeric(row.get("Facebook_Followers", 0), errors='coerce').fillna(0).astype(int)

                author_url = str(row["Author_URL"]).strip() if "Author_URL" in bulk_data.columns and pd.notna(row["Author_URL"]) else ""

                status_text.text(f"Processing: {author} ({index + 1}/{total_authors})")

                # API Calls
                kp_exists, kp_details = check_knowledge_panel(author)
                wiki_exists, wiki_details, wiki_url = check_wikipedia(author)
                
                topical_authority_serp_count, topical_authority_ratio, ai_overview_summary, \
                    top_stories_mentions, topical_associated_domains, perspectives_domains = analyze_topical_serp(author, keyword)

                all_associated_domains_general, matched_uk_publishers_general = get_author_associated_brands(author)
                scholar_citations_count = check_google_scholar_citations(author)
                
                has_perspectives = len(perspectives_domains) > 0 # Boolean for score and display

                quality_score = calculate_quality_score(
                    kp_exists, wiki_exists, topical_authority_ratio,
                    scholar_citations_count, linkedin_followers, x_followers,
                    instagram_followers, tiktok_followers, facebook_followers,
                    len(matched_uk_publishers_general),
                    len(top_stories_mentions),
                    len([d for d in topical_associated_domains if d in UK_PUBLISHER_DOMAINS]),
                    ai_overview_summary != "N/A" and "content loading dynamically" not in ai_overview_summary,
                    has_perspectives
                )

                results.append({
                    "Author": author,
                    "Keyword": keyword,
                    "Author_URL": author_url,
                    "Quality_Score": quality_score,
                    "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
                    "KP_Details": kp_details,
                    "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
                    "Wikipedia_Details": wiki_details,
                    "Wikipedia_URL": wiki_url if wiki_exists else "N/A",
                    "AI_Overview_Summary": ai_overview_summary,
                    "Topical_Authority_SERP_Count": topical_authority_serp_count,
                    "Topical_Authority_Ratio": topical_authority_ratio,
                    "Top_Stories_Mentions_Domains": ", ".join(top_stories_mentions) if top_stories_mentions else "None",
                    "Has_Perspectives": "✅ Yes" if has_perspectives else "❌ No", # New
                    "Perspectives_Domains": ", ".join(perspectives_domains) if perspectives_domains else "None", # New
                    "Scholar_Citations_Count": scholar_citations_count,
                    "General_Matched_UK_Publishers": ", ".join(matched_uk_publishers_general) if matched_uk_publishers_general else "None",
                    "Topic_Associated_Domains": ", ".join(topical_associated_domains) if topical_associated_domains else "None",
                    "All_Associated_Domains_General": ", ".join(all_associated_domains_general),
                    "LinkedIn_Followers": linkedin_followers,
                    "X_Followers": x_followers,
                    "Instagram_Followers": instagram_followers,
                    "TikTok_Followers": tiktok_followers,
                    "Facebook_Followers": facebook_followers,
                })
                time.sleep(1)

    results_df = pd.DataFrame(results)
    st.session_state['bulk_analysis_results_df'] = results_df
    
    st.session_state['triggered_bulk_analysis'] = False
    st.session_state['bulk_data_to_process'] = None

    st.markdown("### High-Level Summary")
    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Total Authors Analyzed", total_authors)
    with col_sum2:
        kp_count = results_df["Has_Knowledge_Panel"].value_counts().get("✅ Yes", 0)
        st.metric("Authors with Knowledge Panels", kp_count)
    with col_sum3:
        wiki_count = results_df["Has_Wikipedia_Page"].value_counts().get("✅ Yes", 0)
        st.metric("Authors with Wikipedia Pages", wiki_count)
    
    st.markdown("---")
    st.markdown("### Detailed Results Table")
    
    def highlight_score_color_cell_bulk(val):
        score_val = int(val) 
        if score_val >= 70:
            return 'background-color: #d4edda' 
        elif score_val >= 40:
            return 'background-color: #ffeeba' 
        else:
            return 'background-color: #f8d7da' 

    def highlight_tick_cross_bg_cell_bulk(val):
        if '✅' in str(val):
            return 'background-color: #e0ffe0' 
        elif '❌' in str(val):
            return 'background-color: #fff0f0' 
        return ''
    
    def make_clickable_wikipedia_bulk(val):
        if val and val != "N/A":
            return f'<a href="{val}" target="_blank">Link</a>'
        return val

    styled_df = results_df.style \
        .map(highlight_score_color_cell_bulk, subset=['Quality_Score']) \
        .applymap(highlight_tick_cross_bg_cell_bulk, subset=['Has_Knowledge_Panel', 'Has_Wikipedia_Page', 'Has_Perspectives']) \
        .format(make_clickable_wikipedia_bulk, subset=['Wikipedia_URL'], escape=False) \
        .format({
            'Topical_Authority_SERP_Count': "{:,.0f}",
            'Topical_Authority_Ratio': "{:.2f}%",
            'Scholar_Citations_Count': "{:,.0f}",
            'LinkedIn_Followers': "{:,.0f}",
            'X_Followers': "{:,.0f}",
            'Instagram_Followers': "{:,.0f}",
            'TikTok_Followers': "{:,.0f}",
            'Facebook_Followers': "{:,.0f}"
        })

    st.dataframe(styled_df, use_container_width=True, height=500) 

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
