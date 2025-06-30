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
    "thetimes.com", # Updated to .com
    "theguardian.com",
    "bbc.co.uk",
    "express.co.uk",
    "standard.co.uk",
    "dailymail.co.uk",
    "independent.co.uk",
    "mirror.co.uk",
    "thesun.co.uk",
    "metro.co.uk",
    "gbnews.com" # Added GB News
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
        st.error(error_msg)
        return {"error": error_msg}
    except requests.exceptions.RequestException as req_err:
        error_msg = f"DataForSEO API request error: {req_err}"
        st.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred during DataForSEO call: {e}"
        st.error(error_msg)
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
    """Gets the total search results count for 'author + topic'."""
    if not topic:
        return 0 # No topic provided, no specific authority to measure
    search_query = f'"{author}" "{topic}"'
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

def check_competitor_author(author: str, competitor_authors_list: list[str]) -> bool:
    """Checks if the author is in the manually provided competitor list."""
    return author.lower() in [comp_author.lower() for comp_author in competitor_authors_list]

def calculate_quality_score(
    has_kp: bool,
    has_wiki: bool,
    topical_authority_serp_count: int,
    scholar_citations_count: int,
    linkedin_followers: int,
    x_followers: int,
    instagram_followers: int,
    tiktok_followers: int,
    matched_uk_publishers_count: int, # Based on new function output
    is_competitor: bool
) -> int:
    """Calculates a quality score based on various signals."""
    score = 0
    if has_kp: score += 15 # High value for KP
    if has_wiki: score += 10 # Good value for Wikipedia
    
    # Scale topical authority: e.g., 1 point per 1000 results, capped at 20 points
    score += min(topical_authority_serp_count // 1000, 20)

    # Scale scholar citations: 1 point per citation, capped at 10 points
    score += min(scholar_citations_count, 10)

    # Social followers: 1 point per 10k followers across all platforms, capped at 15 points
    total_social_followers = linkedin_followers + x_followers + instagram_followers + tiktok_followers
    score += min(total_social_followers // 10000, 15)

    # Points for writing for major UK publishers (2 points per publisher, capped at 10 points)
    score += min(matched_uk_publishers_count * 2, 10)

    if is_competitor: score -= 20 # Significant penalty for being a competitor author

    return max(0, score) # Ensure score doesn't go negative

# --- Streamlit UI Layout ---
st.title("✍️ The Telegraph Recommended: Author Quality Evaluator")
st.markdown("---") # Separator after title
st.markdown("""
This tool helps assess the perceived authority and expertise of authors based on key online signals,
supporting the recruitment of high-quality freelancers for The Telegraph Recommended.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Author Evaluation & Settings")

    st.subheader("Individual Author Evaluation")
    st.markdown("Enter details to analyze a single author.")
    single_author_name = st.text_input("Author Name:", help="Full name of the author.")
    single_keyword_topic = st.text_input("Relevant Topic/Keyword for Expertise:", help="""
        **Guidance for Keywords:**
        - **Be specific:** "smartwatch reviews" not just "watches".
        - **Focus on expertise:** "AI ethics" not just "AI".
        - **Combine terms (if complex):** "Dr. Jane Smith diabetes research".
        - **Avoid overly broad terms** that might dilute relevance.
        """)
    single_author_url = st.text_input("Optional: Author Profile URL (e.g., Telegraph, personal site):", help="e.g., telegraph.co.uk/authors/jane-doe, or author's personal website.")
    
    st.markdown("**Manual Social Media Followers:**")
    col_social1, col_social2 = st.columns(2)
    with col_social1:
        single_linkedin_followers = st.number_input("LinkedIn:", min_value=0, value=0, step=100)
        single_instagram_followers = st.number_input("Instagram:", min_value=0, value=0, step=100)
    with col_social2:
        single_x_followers = st.number_input("X (Twitter):", min_value=0, value=0, step=100)
        single_tiktok_followers = st.number_input("TikTok:", min_value=0, value=0, step=100)

    if st.button("Analyze Single Author"):
        if single_author_name:
            with st.spinner(f"Analyzing '{single_author_name}'... This may take a moment due to API calls."):
                kp_exists, kp_details = check_knowledge_panel(single_author_name)
                wiki_exists, wiki_details = check_wikipedia(single_author_name)
                topical_authority_serp_count = get_topical_authority_serp_count(single_author_name, single_keyword_topic)
                all_associated_domains, matched_uk_publishers = get_author_associated_brands(single_author_name)
                scholar_citations_count = check_google_scholar_citations(single_author_name)
                is_competitor = check_competitor_author(single_author_name, competitor_authors_list)

                quality_score = calculate_quality_score(
                    kp_exists, wiki_exists, topical_authority_serp_count,
                    scholar_citations_count, single_linkedin_followers, single_x_followers,
                    single_instagram_followers, single_tiktok_followers, len(matched_uk_publishers),
                    is_competitor
                )

                st.markdown("---") # Visual separator in sidebar
                st.subheader(f"Results for '{single_author_name}'")
                
                st.metric(label="Quality Score", value=quality_score)
                if quality_score >= 20: st.success("Strong authority signals! Highly recommended.")
                elif quality_score >= 10: st.info("Good authority signals. Recommended.")
                else: st.warning("Limited strong authority signals. Further manual review recommended.")

                st.markdown(f"**Knowledge Panel:** {'✅ Yes' if kp_exists else '❌ No'} ({kp_details})")
                st.markdown(f"**Wikipedia Page:** {'✅ Yes' if wiki_exists else '❌ No'} ({wiki_details})")
                st.markdown(f"**Competitor Author:** {'⚠️ Yes' if is_competitor else '🟢 No'}")
                st.markdown(f"**Topical Authority ('{single_keyword_topic}'):** {topical_authority_serp_count:,} results")
                st.markdown(f"**Google Scholar Citations:** {scholar_citations_count:,} results")
                
                st.markdown(f"**Written for Major UK Publishers:**")
                if matched_uk_publishers:
                    for domain in matched_uk_publishers:
                        st.markdown(f"- {domain}")
                else:
                    st.markdown("None of the specified UK publishers found.")

                if all_associated_domains:
                    with st.expander("See all other associated domains (top 20 search results)"):
                        st.markdown(", ".join(all_associated_domains))
                
                st.markdown(f"**Manual Social Followers:**")
                st.markdown(f"- LinkedIn: {single_linkedin_followers:,}")
                st.markdown(f"- X (Twitter): {single_x_followers:,}")
                st.markdown(f"- Instagram: {single_instagram_followers:,}")
                st.markdown(f"- TikTok: {single_tiktok_followers:,}")

        else:
            st.warning("Please enter an author name to analyze.")

    st.markdown("---")
    st.subheader("Competitor Authors List")
    competitor_authors_raw = st.text_area(
        "Enter competitor author names (one per line):",
        value="Jane Doe\nJohn Smith" # Example placeholder
    )
    competitor_authors_list = [name.strip() for name in competitor_authors_raw.split('\n') if name.strip()]

    st.caption(f"Loaded {len(competitor_authors_list)} competitor authors.")

# --- Main Content Area ---
st.header("Bulk Author Evaluation (CSV Upload)")
st.markdown("""
Upload a CSV file with the following columns:
- **Author** (required): Full name of the author.
- **Keyword** (optional): Relevant topic for expertise (e.g., 'personal finance').
- **Author_URL** (optional): Author's profile URL on your site (e.g., telegraph.co.uk/authors/jane-doe) or personal website.
- **LinkedIn_Followers** (optional): Manual input for LinkedIn follower count.
- **X_Followers** (optional): Manual input for X (Twitter) follower count.
- **Instagram_Followers** (optional): Manual input for Instagram follower count.
- **TikTok_Followers** (optional): Manual input for TikTok follower count.
""")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if "Author" not in data.columns:
            st.error("The CSV file must contain an 'Author' column.")
            st.stop() # Stop execution if critical column is missing

        st.write("Preview of uploaded data:")
        st.dataframe(data.head())

        if st.button("Run Bulk Analysis"):
            results = []
            total_authors = len(data)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for index, row in data.iterrows():
                author = str(row["Author"]).strip()
                keyword = str(row["Keyword"]).strip() if "Keyword" in data.columns and pd.notna(row["Keyword"]) else ""
                linkedin_followers = int(row["LinkedIn_Followers"]) if "LinkedIn_Followers" in data.columns and pd.notna(row["LinkedIn_Followers"]) else 0
                x_followers = int(row["X_Followers"]) if "X_Followers" in data.columns and pd.notna(row["X_Followers"]) else 0
                instagram_followers = int(row["Instagram_Followers"]) if "Instagram_Followers" in data.columns and pd.notna(row["Instagram_Followers"]) else 0
                tiktok_followers = int(row["TikTok_Followers"]) if "TikTok_Followers" in data.columns and pd.notna(row["TikTok_Followers"]) else 0
                author_url = str(row["Author_URL"]).strip() if "Author_URL" in data.columns and pd.notna(row["Author_URL"]) else ""


                status_text.text(f"Processing: {author} ({index + 1}/{total_authors})")

                # API Calls
                kp_exists, kp_details = check_knowledge_panel(author)
                wiki_exists, wiki_details = check_wikipedia(author)
                topical_authority_serp_count = get_topical_authority_serp_count(author, keyword)
                all_associated_domains, matched_uk_publishers = get_author_associated_brands(author)
                scholar_citations_count = check_google_scholar_citations(author)

                # Manual Checks / Derived Data
                is_competitor = check_competitor_author(author, competitor_authors_list)
                
                quality_score = calculate_quality_score(
                    kp_exists, wiki_exists, topical_authority_serp_count,
                    scholar_citations_count, linkedin_followers, x_followers,
                    instagram_followers, tiktok_followers, len(matched_uk_publishers),
                    is_competitor
                )

                results.append({
                    "Author": author,
                    "Keyword": keyword,
                    "Author_URL": author_url,
                    "Has_Knowledge_Panel": "Yes" if kp_exists else "No",
                    "KP_Details": kp_details,
                    "Has_Wikipedia_Page": "Yes" if wiki_exists else "No",
                    "Wikipedia_Details": wiki_details,
                    "Topical_Authority_SERP_Count": topical_authority_serp_count,
                    "Scholar_Citations_Count": scholar_citations_count,
                    "Associated_Brands_Domains": ", ".join(all_associated_domains), # All domains for CSV
                    "Matched_UK_Publishers": ", ".join(matched_uk_publishers) if matched_uk_publishers else "None",
                    "Manual_LinkedIn_Followers": linkedin_followers,
                    "Manual_X_Followers": x_followers,
                    "Manual_Instagram_Followers": instagram_followers,
                    "Manual_TikTok_Followers": tiktok_followers,
                    "Is_Competitor_Author": "Yes" if is_competitor else "No",
                    "Quality_Score": quality_score
                })
                progress_bar.progress((index + 1) / total_authors)
                time.sleep(1) # Small delay to be mindful of API rate limits and display updates

            results_df = pd.DataFrame(results)
            st.subheader("Bulk Analysis Results:")
            
            # --- Visualisation of Bulk Results ---
            st.markdown("### Top-Level Summary")
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            with col_sum1:
                st.metric("Total Authors Analyzed", total_authors)
            with col_sum2:
                kp_count = results_df["Has_Knowledge_Panel"].value_counts().get("Yes", 0)
                st.metric("Authors with Knowledge Panels", kp_count)
            with col_sum3:
                wiki_count = results_df["Has_Wikipedia_Page"].value_counts().get("Yes", 0)
                st.metric("Authors with Wikipedia Pages", wiki_count)
            
            st.markdown("### Detailed Results Table")
            # Using st.dataframe for interactive table
            st.dataframe(results_df)

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

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
    except Exception as e:
        st.error(f"Error processing CSV file or running analysis: {e}")
        st.info("Please ensure your CSV has the required 'Author' column and is correctly formatted with optional columns.")
