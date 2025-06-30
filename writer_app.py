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
    "mirror.co.uk",
    "thesun.co.uk",
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
        # st.error(error_msg) # Commented out for cleaner UI, errors will be in data response
        return {"error": error_msg}
    except requests.exceptions.RequestException as req_err:
        error_msg = f"DataForSEO API request error: {req_err}"
        # st.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred during DataForSEO call: {e}"
        # st.error(error_msg)
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
def check_wikipedia(author: str) -> tuple[bool, str]:
    """Checks Wikipedia API for a page matching the author's name."""
    wiki_author_query = author.replace(' ', '_')
    wikipedia_api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={wiki_author_query}&format=json"
    try:
        response = requests.get(wikipedia_api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        if "-1" not in pages:
            return True, "Wikipedia page found"
        return False, "No Wikipedia page found"
    except requests.exceptions.RequestException as e:
        return False, f"Wikipedia API error: {e}"
    except Exception as e:
        return False, f"An unexpected error occurred checking Wikipedia: {e}"

@st.cache_data(ttl=3600)
def get_topical_authority_serp_count(author: str, topic: str) -> int:
    """Gets the total search results count for 'author AND topic'."""
    if not topic:
        return 0 # No topic provided, no specific authority to measure
    # Use Google's AND operator for precise search
    search_query = f'"{author}" AND "{topic}"'
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)
    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                for result in task["result"]:
                    return result.get("serp", {}).get("results_count", 0)
    return 0

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
                for result in task["result"]:
                    if "items" in result:
                        for item in result["items"]:
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

# Removed check_competitor_author as per request

def calculate_quality_score(
    has_kp: bool,
    has_wiki: bool,
    topical_authority_serp_count: int,
    scholar_citations_count: int,
    linkedin_followers: int,
    x_followers: int,
    instagram_followers: int,
    tiktok_followers: int,
    facebook_followers: int,
    matched_uk_publishers_count: int,
    # is_competitor: bool - Removed as per request
) -> int:
    """Calculates a quality score based on various signals."""
    score = 0
    if has_kp: score += 15 # High value for KP
    if has_wiki: score += 10 # Good value for Wikipedia
    
    # Scale topical authority: e.g., 1 point per 1000 results, capped at 20 points
    score += min(topical_authority_serp_count // 1000, 20)

    # Scale scholar citations: 1 point per citation, capped at 10 points
    score += min(scholar_citations_count, 10)

    # Social followers: 1 point per 10k followers across all platforms, capped at 20 points
    total_social_followers = linkedin_followers + x_followers + instagram_followers + tiktok_followers + facebook_followers
    score += min(total_social_followers // 10000, 20)

    # Points for writing for major UK publishers (2 points per publisher, capped at 10 points)
    score += min(matched_uk_publishers_count * 2, 10)

    # if is_competitor: score -= 20 # Removed as per request

    return max(0, score) # Ensure score doesn't go negative

# --- Main Page Title ---
st.title("✍️ The Telegraph Recommended: Author Quality Evaluator")
st.markdown("---") # Separator after title
st.markdown("""
This tool helps assess the perceived authority and expertise of authors based on key online signals,
supporting the recruitment of high-quality freelancers for The Telegraph Recommended.
""")

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
        with col_social2:
            single_x_followers = st.number_input("X (Twitter):", min_value=0, value=0, step=100, key="single_x_followers_input")
            single_tiktok_followers = st.number_input("TikTok:", min_value=0, value=0, step=100, key="single_tiktok_followers_input")
            single_facebook_followers = st.number_input("Facebook:", min_value=0, value=0, step=100, key="single_facebook_followers_input")

        # Define a dummy competitor_authors_list for single author logic, as the input is removed
        # The quality score logic for is_competitor is also removed now.
        dummy_competitor_authors_list = [] 

        if st.button("Analyze Single Author", use_container_width=True):
            if single_author_name:
                # Store single analysis results in session state to display in main section
                st.session_state['single_author_results'] = {
                    "Author": single_author_name,
                    "Keyword": single_keyword_topic,
                    "Author_URL": single_author_url,
                    "LinkedIn_Followers": single_linkedin_followers,
                    "X_Followers": single_x_followers,
                    "Instagram_Followers": single_instagram_followers,
                    "TikTok_Followers": single_tiktok_followers,
                    "Facebook_Followers": single_facebook_followers
                }
                st.session_state['run_single_analysis'] = True
                # Rerun the app to show results in main section
                st.experimental_rerun()
            else:
                st.warning("Please enter an author name to analyze.")

    st.markdown("---") # Separator

    # --- Bulk Author Evaluation (moved to sidebar) ---
    st.subheader("Bulk Author Analysis (CSV Upload)")
    st.markdown("""
    Upload a CSV file with the following columns:
    - **Author** (required): Full name of the author.
    - **Keyword** (optional): Relevant topic for expertise (e.g., 'personal finance').
    - **Author_URL** (optional): Author's profile URL.
    - **LinkedIn_Followers** (optional): Manual LinkedIn follower count.
    - **X_Followers** (optional): Manual X follower count.
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
                st.stop()
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
        st.session_state['run_bulk_analysis'] = True
        st.experimental_rerun()


# --- Main Content Area (Visualization) ---
st.header("Analysis Results")
st.markdown("---") # Separator for results section

# Initialize session state for displaying results
if 'single_author_results' not in st.session_state:
    st.session_state['single_author_results'] = None
if 'bulk_analysis_results_df' not in st.session_state:
    st.session_state['bulk_analysis_results_df'] = None
if 'run_single_analysis' not in st.session_state:
    st.session_state['run_single_analysis'] = False
if 'run_bulk_analysis' not in st.session_state:
    st.session_state['run_bulk_analysis'] = False

# Process and display single author results if triggered
if st.session_state['run_single_analysis'] and st.session_state['single_author_results']:
    st.subheader("Individual Author Analysis Results")
    
    author_info = st.session_state['single_author_results']
    author = author_info["Author"]
    keyword = author_info["Keyword"]
    linkedin_followers = author_info["LinkedIn_Followers"]
    x_followers = author_info["X_Followers"]
    instagram_followers = author_info["Instagram_Followers"]
    tiktok_followers = author_info["TikTok_Followers"]
    facebook_followers = author_info["Facebook_Followers"]
    author_url = author_info["Author_URL"]

    with st.spinner(f"Analyzing '{author}'... This may take a moment due to API calls."):
        # API Calls
        kp_exists, kp_details = check_knowledge_panel(author)
        wiki_exists, wiki_details = check_wikipedia(author)
        topical_authority_serp_count = get_topical_authority_serp_count(author, keyword)
        all_associated_domains, matched_uk_publishers = get_author_associated_brands(author)
        scholar_citations_count = check_google_scholar_citations(author)

        # Removed competitor check for single author as input is gone
        is_competitor = False # Default to False since input is removed
        
        quality_score = calculate_quality_score(
            kp_exists, wiki_exists, topical_authority_serp_count,
            scholar_citations_count, linkedin_followers, x_followers,
            instagram_followers, tiktok_followers, facebook_followers,
            len(matched_uk_publishers)
            # is_competitor is removed from this score calculation
        )

        single_author_results_row = pd.DataFrame([{
            "Author": author,
            "Keyword": keyword,
            "Author_URL": author_url,
            "Quality_Score": quality_score,
            "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
            "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
            "Topical_Authority_SERP_Count": f"{topical_authority_serp_count:,}",
            "Scholar_Citations_Count": f"{scholar_citations_count:,}",
            "Matched_UK_Publishers": ", ".join(matched_uk_publishers) if matched_uk_publishers else "None",
            "LinkedIn_Followers": f"{linkedin_followers:,}",
            "X_Followers": f"{x_followers:,}",
            "Instagram_Followers": f"{instagram_followers:,}",
            "TikTok_Followers": f"{tiktok_followers:,}",
            "Facebook_Followers": f"{facebook_followers:,}",
            "Associated_Brands_Domains": ", ".join(all_associated_domains),
            "KP_Details": kp_details, # For more detail if needed, could be hidden in expander
            "Wikipedia_Details": wiki_details # For more detail
        }])

        st.dataframe(single_author_results_row.style.apply(
            lambda s: ['background-color: #d4edda'] * len(s) if s['Quality_Score'].iloc[0] >= 30 else (
                ['background-color: #ffeeba'] * len(s) if s['Quality_Score'].iloc[0] >= 15 else
                ['background-color: #f8d7da'] * len(s)
            ), axis=1
        ).applymap(
            lambda x: 'background-color: #e0ffe0' if '✅' in str(x) else ('background-color: #fff0f0' if '❌' in str(x) else ''),
            subset=['Has_Knowledge_Panel', 'Has_Wikipedia_Page']
        ), use_container_width=True)

        # Reset flag
        st.session_state['run_single_analysis'] = False
        st.session_state['single_author_results'] = None # Clear previous results

# Process and display bulk analysis results if triggered
elif st.session_state['run_bulk_analysis'] and st.session_state['bulk_data_to_process'] is not None:
    st.subheader("Bulk Author Analysis Results")
    bulk_data = st.session_state['bulk_data_to_process']
    results = []
    total_authors = len(bulk_data)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Define a dummy competitor_authors_list for bulk processing as input is removed
    dummy_competitor_authors_list = [] # No longer needed for logic, but kept for function signature

    for index, row in bulk_data.iterrows():
        author = str(row["Author"]).strip()
        keyword = str(row["Keyword"]).strip() if "Keyword" in bulk_data.columns and pd.notna(row["Keyword"]) else ""
        linkedin_followers = int(row["LinkedIn_Followers"]) if "LinkedIn_Followers" in bulk_data.columns and pd.notna(row["LinkedIn_Followers"]) else 0
        x_followers = int(row["X_Followers"]) if "X_Followers" in bulk_data.columns and pd.notna(row["X_Followers"]) else 0
        instagram_followers = int(row["Instagram_Followers"]) if "Instagram_Followers" in bulk_data.columns and pd.notna(row["Instagram_Followers"]) else 0
        tiktok_followers = int(row["TikTok_Followers"]) if "TikTok_Followers" in bulk_data.columns and pd.notna(row["TikTok_Followers"]) else 0
        facebook_followers = int(row["Facebook_Followers"]) if "Facebook_Followers" in bulk_data.columns and pd.notna(row["Facebook_Followers"]) else 0
        author_url = str(row["Author_URL"]).strip() if "Author_URL" in bulk_data.columns and pd.notna(row["Author_URL"]) else ""


        status_text.text(f"Processing: {author} ({index + 1}/{total_authors})")

        # API Calls
        kp_exists, kp_details = check_knowledge_panel(author)
        wiki_exists, wiki_details = check_wikipedia(author)
        topical_authority_serp_count = get_topical_authority_serp_count(author, keyword)
        all_associated_domains, matched_uk_publishers = get_author_associated_brands(author)
        scholar_citations_count = check_google_scholar_citations(author)

        # Removed competitor check
        is_competitor = False # Default to False since input is removed
        
        quality_score = calculate_quality_score(
            kp_exists, wiki_exists, topical_authority_serp_count,
            scholar_citations_count, linkedin_followers, x_followers,
            instagram_followers, tiktok_followers, facebook_followers,
            len(matched_uk_publishers)
            # is_competitor is removed from this score calculation
        )

        results.append({
            "Author": author,
            "Keyword": keyword,
            "Author_URL": author_url,
            "Quality_Score": quality_score,
            "Has_Knowledge_Panel": "✅ Yes" if kp_exists else "❌ No",
            "Has_Wikipedia_Page": "✅ Yes" if wiki_exists else "❌ No",
            "Topical_Authority_SERP_Count": f"{topical_authority_serp_count:,}",
            "Scholar_Citations_Count": f"{scholar_citations_count:,}",
            "Matched_UK_Publishers": ", ".join(matched_uk_publishers) if matched_uk_publishers else "None",
            "All_Associated_Domains": ", ".join(all_associated_domains), # All domains for CSV
            "LinkedIn_Followers": f"{linkedin_followers:,}",
            "X_Followers": f"{x_followers:,}",
            "Instagram_Followers": f"{instagram_followers:,}",
            "TikTok_Followers": f"{tiktok_followers:,}",
            "Facebook_Followers": f"{facebook_followers:,}",
            "KP_Details": kp_details,
            "Wikipedia_Details": wiki_details
        })
        progress_bar.progress((index + 1) / total_authors)
        time.sleep(1) # Small delay to be mindful of API rate limits and display updates

    results_df = pd.DataFrame(results)
    st.session_state['bulk_analysis_results_df'] = results_df # Store for display
    
    # Reset flag
    st.session_state['run_bulk_analysis'] = False
    st.session_state['bulk_data_to_process'] = None # Clear data after processing

    # Display results immediately after processing
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
    
    def highlight_score_color(val):
        score_val = int(val) # Convert to int for comparison
        if score_val >= 30:
            return 'background-color: #d4edda' # Light green
        elif score_val >= 15:
            return 'background-color: #ffeeba' # Light yellow
        else:
            return 'background-color: #f8d7da' # Light red

    def highlight_tick_cross_bg(val):
        if '✅' in str(val):
            return 'background-color: #e0ffe0' # Very light green
        elif '❌' in str(val):
            return 'background-color: #fff0f0' # Very light red
        return ''

    # Apply styling
    styled_df = results_df.style \
        .applymap(highlight_score_color, subset=['Quality_Score']) \
        .applymap(highlight_tick_cross_bg, subset=['Has_Knowledge_Panel', 'Has_Wikipedia_Page']) \
        .format(subset=['Topical_Authority_SERP_Count', 'Scholar_Citations_Count',
                       'LinkedIn_Followers', 'X_Followers', 'Instagram_Followers',
                       'TikTok_Followers', 'Facebook_Followers'], formatter='{:}') # Ensure numbers are formatted as strings already

    st.dataframe(styled_df, use_container_width=True, height=500) # Added height for scrollability

    # Add a download button for the results
    csv_output = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv_output,
        file_name="author_expertise_results.csv",
        mime="text/csv",
        help="Download the analysis results as a CSV file."
    )
    status_text.success("Bulk analysis complete!")

# Initial display or if no analysis has run yet
else:
    st.info("Use the sidebar to input author details for individual analysis, or upload a CSV for bulk processing.")
