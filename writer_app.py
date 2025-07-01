import streamlit as st
import requests
import time
import pandas as pd
import re # For regular expressions, useful for extracting domain names

st.set_page_config(layout="wide", page_title="Author Quality Evaluator")

# --- Configuration & API Endpoints ---
# Load API credentials from Streamlit secrets
try:
    API_USERNAME = st.secrets["API_USERNAME"]
    API_PASSWORD = st.secrets["API_PASSWORD"]
except KeyError as e:
    st.error(f"Missing API secret: {e}. Please configure secrets in Streamlit Cloud dashboard.")
    st.stop() # Stop the app if secrets are missing

DATAFORSEO_ORGANIC_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"

# Define a list of major UK publisher domains for "Associated Brands" check
UK_PUBLISHER_DOMAINS = [
    "thetimes.com",
    "theguardian.com",
    "bbc.co.uk",
    "express.co.uk",
    "standard.co.uk",
    "dailymail.co.uk",
    "independent.co.uk",
    "thesun.co.uk", # Corrected for .co.uk
    "mirror.co.uk",
    "metro.co.uk",
    "gbnews.com"
]
# Exclude these generic domains from "Associated Brands"
EXCLUDED_GENERIC_DOMAINS_REGEX = [
    r"wikipedia\.org", r"linkedin\.com", r"twitter\.com", r"x\.com",
    r"facebook\.com", r"instagram\.com", r"youtube\.com", r"pinterest\.com",
    r"tiktok\.com", r"medium\.com", r"quora\.com", r"reddit\.com",
    r"threads\.net", r"telegraph\.co\.uk", # Exclude your own domain
    r"amazon\." # Exclude Amazon links
]

# --- Helper Functions ---

@st.cache_data(ttl=3600) # Cache results for 1 hour
def make_dataforseo_call(payload: dict) -> dict:
    """Helper function to make DataForSEO API requests."""
    try:
        # Limit to top 20 results to save credits
        payload["limit"] = 20
        # Ensure a list of payloads is sent
        if not isinstance(payload, list):
            payload = [payload]

        response = requests.post(DATAFORSEO_ORGANIC_URL, auth=(API_USERNAME, API_PASSWORD), json=payload, timeout=60)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
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
    """Checks DataForSEO for a Google Knowledge Panel."""
    search_query = f'"{author}"'
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)

    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                for result in task["result"]:
                    if "knowledge_graph" in result and result["knowledge_graph"]:
                        return True, "Knowledge Panel found"
        return False, "No Knowledge Panel found in SERP results"
    return False, data.get("error", "No data or task in DataForSEO response.")

@st.cache_data(ttl=3600)
def check_wikipedia(author: str) -> tuple[bool, str, str]: # Added str for URL
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
def get_topical_authority_metrics(author: str, topic: str) -> tuple[int, float]:
    """
    Gets total search results for 'author AND topic' and for 'topic' alone.
    Calculates topical authority ratio.
    """
    if not topic:
        return 0, 0.0 # No topic provided

    # 1. Search for "author AND topic"
    author_topic_query = f'"{author}" AND "{topic}"'
    author_topic_payload = {"keyword": author_topic_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    author_topic_data = make_dataforseo_call(author_topic_payload)
    author_topic_results_count = 0
    if author_topic_data and "tasks" in author_topic_data and author_topic_data["tasks"]:
        for task in author_topic_data["tasks"]:
            if "result" in task and task["result"]:
                for result in task["result"]:
                    author_topic_results_count = result.get("serp", {}).get("results_count", 0)
                    break # Assuming first result is sufficient
            if author_topic_results_count > 0: break # Break outer loop if count found

    # 2. Search for "topic" alone
    topic_query = f'"{topic}"'
    topic_payload = {"keyword": topic_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    topic_data = make_dataforseo_call(topic_payload)
    total_topic_results_count = 0
    if topic_data and "tasks" in topic_data and topic_data["tasks"]:
        for task in topic_data["tasks"]:
            if "result" in task and task["result"]:
                for result in task["result"]:
                    total_topic_results_count = result.get("serp", {}).get("results_count", 0)
                    break
            if total_topic_results_count > 0: break


    # Calculate ratio
    topical_authority_ratio = 0.0
    if total_topic_results_count > 0:
        topical_authority_ratio = (author_topic_results_count / total_topic_results_count) * 100 # As a percentage

    return author_topic_results_count, topical_authority_ratio

@st.cache_data(ttl=3600)
def get_author_associated_brands(author: str) -> tuple[list[str], list[str]]:
    """
    Finds unique domains an author is associated with by searching their name.
    Separates into general associated domains and matches with predefined UK publishers.
    """
    search_query = f'"{author}"'
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)

    all_associated_domains = set()
    matched_uk_publishers = set()

    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                # Iterate through the actual results within the 'result' list
                for result_item in task["result"]: # Renamed to result_item for clarity
                    if "items" in result_item: # Check if 'items' array is in this result_item
                        for item in result_item["items"]: # Now iterate through the actual organic items
                            if item.get("type") == "organic" and "domain" in item:
                                domain = item["domain"]
                                # Check against generic exclusions
                                if not any(re.search(pattern, domain) for pattern in EXCLUDED_GENERIC_DOMAINS_REGEX):
                                    all_associated_domains.add(domain)
                                # Check against specific UK publishers
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
    topical_authority_ratio: float, # Now using ratio
    scholar_citations_count: int,
    linkedin_followers: int,
    x_followers: int,
    instagram_followers: int,
    tiktok_followers: int,
    facebook_followers: int,
    matched_uk_publishers_count: int,
) -> int:
    """Calculates a quality score based on various signals."""
    score = 0
    if has_kp: score += 15 # High value for KP
    if has_wiki: score += 10 # Good value for Wikipedia
    
    # Scale topical authority ratio: e.g., 1 point per 1% up to 20% capped at 20 points
    score += min(int(topical_authority_ratio * 2), 20) # 2 points per %

    # Scale scholar citations: 1 point per citation, capped at 10 points
    score += min(scholar_citations_count, 10)

    # Social followers: 1 point per 10k followers across all platforms, capped at 20 points
    total_social_followers = linkedin_followers + x_followers + instagram_followers + tiktok_followers + facebook_followers
    score += min(total_social_followers // 10000, 20)

    # Points for writing for major UK publishers (2 points per publisher, capped at 10 points)
    score += min(matched_uk_publishers_count * 2, 10)

    return max(0, score) # Ensure score doesn't go negative

# --- Main Page Title ---
st.title("✍️ The Telegraph Recommended: Author Quality Evaluator")
st.markdown("---") # Separator after title
st.markdown("""
This tool helps assess the perceived authority and expertise of authors based on key online signals,
supporting the recruitment of high-quality freelancers for The Telegraph Recommended.
""")
# --- END Main Page Title ---

# --- Initialize Session State for Results ---
# This ensures results persist across reruns and are visible in the main section
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
            - **Avoid overly broad terms** that might dilute relevance.
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
                    wiki_exists, wiki_details, wiki_url = check_wikipedia(single_author_name) # Get URL
                    topical_authority_serp_count, topical_authority_ratio = get_topical_authority_metrics(single_author_name, single_keyword_topic)
                    all_associated_domains, matched_uk_publishers = get_author_associated_brands(single_author_name)
                    scholar_citations_count = check_google_scholar_citations(single_author_name)
                    
                    quality_score = calculate_quality_score(
                        kp_exists, wiki_exists, topical_authority_ratio, # Pass ratio
                        scholar_citations_count, single_linkedin_followers, single_x_followers,
                        single_instagram_followers, single_tiktok_followers, single_facebook_followers,
                        len(matched_uk_publishers)
                    )

                    # Store results in session state for display in main section
                    st.session_state['single_author_display_results'] = pd.DataFrame([{
                        "Author": single_author_name,
                        "Keyword": single_keyword_topic,
                        "Author_URL": single_author_url,
                        "Quality_Score": quality_score, # Keep as int for styling to work smoothly
                        "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
                        "KP_Details": kp_details, # For more detail if needed, could be hidden in expander
                        "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
                        "Wikipedia_Details": wiki_details, # Keep original detail
                        "Wikipedia_URL": wiki_url if wiki_exists else "N/A", # New URL field
                        "Topical_Authority_SERP_Count": f"{topical_authority_serp_count:,}",
                        "Topical_Authority_Ratio": f"{topical_authority_ratio:.2f}%", # New ratio field
                        "Scholar_Citations_Count": f"{scholar_citations_count:,}",
                        "Matched_UK_Publishers": ", ".join(matched_uk_publishers) if matched_uk_publishers else "None",
                        "All_Associated_Domains": ", ".join(all_associated_domains),
                        "LinkedIn_Followers": f"{single_linkedin_followers:,}",
                        "X_Followers": f"{single_x_followers:,}",
                        "Instagram_Followers": f"{single_instagram_followers:,}",
                        "TikTok_Followers": f"{tiktok_followers:,}",
                        "Facebook_Followers": f"{single_facebook_followers:,}",
                    }])
                    st.session_state['triggered_single_analysis'] = True # Set flag to display
            else:
                st.warning("Please enter an author name to analyze.")

    st.markdown("---") # Separator

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
                st.session_state['bulk_data_to_process'] = bulk_data # Store for processing later
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
            st.session_state['triggered_bulk_analysis'] = True # Set flag to display


# --- Main Content Area (Visualization) ---
st.header("Analysis Results")
st.markdown("---") # Separator for results section

# Process and display single author results if triggered
if st.session_state['triggered_single_analysis'] and st.session_state['single_author_display_results'] is not None:
    st.subheader("Individual Author Analysis Results")
    
    # Define styling functions here to ensure they are properly scoped for single display
    def highlight_score_color_row(s):
        score_val = s['Quality_Score'].iloc[0] # Quality_Score is already int
        if score_val >= 30:
            return ['background-color: #d4edda'] * len(s) # Light green
        elif score_val >= 15:
            return ['background-color: #ffeeba'] * len(s) # Light yellow
        else:
            return ['background-color: #f8d7da'] * len(s) # Light red
    
    def highlight_tick_cross_bg_cell(val):
        if '✅' in str(val):
            return 'background-color: #e0ffe0' # Very light green
        elif '❌' in str(val):
            return 'background-color: #fff0f0' # Very light red
        return ''

    styled_single_df = st.session_state['single_author_display_results'].style.apply(
        highlight_score_color_row, axis=1
    ).applymap(
        highlight_tick_cross_bg_cell,
        subset=['Has_Knowledge_Panel', 'Has_Wikipedia_Page']
    )
    # Custom format Wikipedia URL to be clickable
    def make_clickable_wikipedia(val):
        if val and val != "N/A":
            return f'<a href="{val}" target="_blank">Link</a>'
        return val
    
    styled_single_df = styled_single_df.format(make_clickable_wikipedia, subset=['Wikipedia_URL'], escape=False)

    st.dataframe(styled_single_df, use_container_width=True)

    # Reset the trigger after display
    st.session_state['triggered_single_analysis'] = False
    st.session_state['single_author_display_results'] = None # Clear previous results after display

# Process and display bulk analysis results if triggered
elif st.session_state['triggered_bulk_analysis'] and st.session_state['bulk_data_to_process'] is not None:
    st.subheader("Bulk Author Analysis Results")
    
    bulk_data = st.session_state['bulk_data_to_process']
    results = []
    total_authors = len(bulk_data)
    
    progress_bar = st.progress(0)
    status_text = st.empty() 
    st_spinner_placeholder = st.empty() # Placeholder for a spinner during bulk processing

    with st_spinner_placeholder.container(): # Use a container for the spinner and text
        with st.spinner("Processing your bulk request... This may take some time due to API calls."):
            for index, row in bulk_data.iterrows():
                author = str(row["Author"]).strip()
                keyword = str(row["Keyword"]).strip() if "Keyword" in bulk_data.columns and pd.notna(row["Keyword"]) else ""
                # Ensure all follower variables are always defined and converted to int robustly
                linkedin_followers = pd.to_numeric(row.get("LinkedIn_Followers", 0), errors='coerce').fillna(0).astype(int)
                x_followers = pd.to_numeric(row.get("X_Followers", 0), errors='coerce').fillna(0).astype(int)
                instagram_followers = pd.to_numeric(row.get("Instagram_Followers", 0), errors='coerce').fillna(0).astype(int)
                tiktok_followers = pd.to_numeric(row.get("TikTok_Followers", 0), errors='coerce').fillna(0).astype(int)
                facebook_followers = pd.to_numeric(row.get("Facebook_Followers", 0), errors='coerce').fillna(0).astype(int)
                author_url = str(row["Author_URL"]).strip() if "Author_URL" in bulk_data.columns and pd.notna(row["Author_URL"]) else ""

                status_text.text(f"Processing: {author} ({index + 1}/{total_authors})")

                # API Calls
                kp_exists, kp_details = check_knowledge_panel(author)
                wiki_exists, wiki_details, wiki_url = check_wikipedia(author) # Get URL
                topical_authority_serp_count, topical_authority_ratio = get_topical_authority_metrics(author, keyword)
                all_associated_domains, matched_uk_publishers = get_author_associated_brands(author)
                scholar_citations_count = check_google_scholar_citations(author)
                
                quality_score = calculate_quality_score(
                    kp_exists, wiki_exists, topical_authority_ratio,
                    scholar_citations_count, linkedin_followers, x_followers,
                    instagram_followers, tiktok_followers, facebook_followers,
                    len(matched_uk_publishers)
                )

                results.append({
                    "Author": author,
                    "Keyword": keyword,
                    "Author_URL": author_url,
                    "Quality_Score": quality_score, # Keep as int for styling to work smoothly
                    "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
                    "KP_Details": kp_details,
                    "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
                    "Wikipedia_Details": wiki_details,
                    "Wikipedia_URL": wiki_url if wiki_exists else "N/A", # New URL field
                    "Topical_Authority_SERP_Count": f"{topical_authority_serp_count:,}",
                    "Topical_Authority_Ratio": f"{topical_authority_ratio:.2f}%", # New ratio field
                    "Scholar_Citations_Count": f"{scholar_citations_count:,}",
                    "Matched_UK_Publishers": ", ".join(matched_uk_publishers) if matched_uk_publishers else "None",
                    "All_Associated_Domains": ", ".join(all_associated_domains),
                    "LinkedIn_Followers": f"{linkedin_followers:,}",
                    "X_Followers": f"{x_followers:,}",
                    "Instagram_Followers": f"{instagram_followers:,}",
                    "TikTok_Followers": f"{tiktok_followers:,}",
                    "Facebook_Followers": f"{facebook_followers:,}",
                })
                time.sleep(1) # Small delay to be mindful of API rate limits and display updates

    results_df = pd.DataFrame(results)
    st.session_state['bulk_analysis_results_df'] = results_df # Store for display
    
    # Reset flags and data after processing
    st.session_state['triggered_bulk_analysis'] = False
    st.session_state['bulk_data_to_process'] = None # Clear data after processing

    # --- Visualisation of Bulk Results ---
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
    
    # Define styling functions here to ensure they are accessible
    def highlight_score_color_cell(val):
        score_val = int(val) 
        if score_val >= 30:
            return 'background-color: #d4edda' 
        elif score_val >= 15:
            return 'background-color: #ffeeba' 
        else:
            return 'background-color: #f8d7da' 

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

    styled_df = results_df.style \
        .map(highlight_score_color_cell, subset=['Quality_Score']) \
        .applymap(highlight_tick_cross_bg_cell, subset=['Has_Knowledge_Panel', 'Has_Wikipedia_Page']) \
        .format(make_clickable_wikipedia, subset=['Wikipedia_URL'], escape=False) \
        .format(subset=['Topical_Authority_SERP_Count', 'Scholar_Citations_Count',
                       'LinkedIn_Followers', 'X_Followers', 'Instagram_Followers',
                       'TikTok_Followers', 'Facebook_Followers', 'Topical_Authority_Ratio'], formatter='{:}')

    st.dataframe(styled_df, use_container_width=True, height=500) 

    csv_output = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv_output,
        file_name="author_expertise_results.csv",
        mime="text/csv",
        help="Download the analysis results as a CSV file."
    )
    progress_bar.empty() # Clear the processing bar
    status_text.empty() # Clear the processing status message
    st_spinner_placeholder.empty() # Clear the spinner
    st.success("Bulk analysis complete!")

# If neither analysis has been triggered yet, show an initial message
else:
    st.info("Use the sidebar to input author details for individual analysis, or upload a CSV for bulk processing. Results will appear here.")
