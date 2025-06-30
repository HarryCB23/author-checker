import streamlit as st
import requests
import time
import pandas as pd
import re # For regular expressions, useful for extracting domain names

st.set_page_config(layout="wide", page_title="Author Quality Evaluator")

st.info("App started: Imports complete.") # Debugging log

# --- Configuration & API Endpoints ---
# Load API credentials from Streamlit secrets
try:
    API_USERNAME = st.secrets["API_USERNAME"]
    API_PASSWORD = st.secrets["API_PASSWORD"]
    st.success("API secrets loaded successfully.")
except KeyError as e:
    st.error(f"Missing API secret: {e}. Please configure secrets in Streamlit Cloud dashboard.")
    st.stop() # Stop the app if secrets are missing

DATAFORSEO_ORGANIC_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"

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
        st.error(f"DataForSEO HTTP error: {http_err} - Response: {response.text}")
        return {"error": str(http_err), "response_text": response.text}
    except requests.exceptions.RequestException as req_err:
        st.error(f"DataForSEO API request error: {req_err}")
        return {"error": str(req_err)}
    except Exception as e:
        st.error(f"An unexpected error occurred during DataForSEO call: {e}")
        return {"error": str(e)}

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
def get_author_related_domains(author: str) -> list[str]:
    """
    Finds unique domains an author is associated with by searching their name.
    Filters out common social media, Wikipedia, and self-references.
    """
    search_query = f'"{author}"'
    payload = {"keyword": search_query, "language_code": "en", "location_name": "United Kingdom", "device": "desktop"}
    data = make_dataforseo_call(payload)

    domains = set()
    excluded_domains_regex = [
        r"wikipedia\.org", r"linkedin\.com", r"twitter\.com", r"x\.com",
        r"facebook\.com", r"instagram\.com", r"youtube\.com", r"pinterest\.com",
        r"tiktok\.com", r"medium\.com", r"quora\.com", r"reddit\.com",
        r"threads\.net", r"telegraph\.co\.uk" # Exclude your own domain
    ]

    if data and "tasks" in data and data["tasks"]:
        for task in data["tasks"]:
            if "result" in task and task["result"]:
                for result in task["result"]:
                    if "items" in result:
                        for item in result["items"]:
                            if item.get("type") == "organic" and "domain" in item:
                                domain = item["domain"]
                                if not any(re.search(pattern, domain) for pattern in excluded_domains_regex):
                                    domains.add(domain)
    return sorted(list(domains))

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
    social_followers: int, # Manually input
    other_brands_count: int,
    is_competitor: bool
) -> int:
    """Calculates a quality score based on various signals."""
    score = 0
    if has_kp: score += 15 # High value for KP
    if has_wiki: score += 10 # Good value for Wikipedia
    
    # Scale topical authority: e.g., 1 point per 1000 results, capped
    score += min(topical_authority_serp_count // 1000, 20) # Max 20 points for topical authority (20k results)

    # Scale scholar citations: 1 point per citation, capped
    score += min(scholar_citations_count, 10) # Max 10 points for citations

    # Scale social followers: 1 point per 10k followers, capped
    score += min(social_followers // 10000, 15) # Max 15 points for social (150k followers)

    score += min(other_brands_count * 2, 10) # Max 10 points for 5+ other brands

    if is_competitor: score -= 20 # Significant penalty for being a competitor author

    return max(0, score) # Ensure score doesn't go negative

st.info("Helper functions defined.") # Debugging log

# --- Streamlit UI Layout ---
st.title("✍️ The Telegraph Recommended: Author Quality Evaluator")
st.markdown("""
This tool helps assess the perceived authority and expertise of authors based on key online signals,
supporting the recruitment of high-quality freelancers for The Telegraph Recommended.
""")

# --- Keyword Guidance ---
st.sidebar.markdown("---")
st.sidebar.subheader("Keyword Input Guidance")
st.sidebar.markdown("""
For best results in "Topical Authority" and "Author Ranks For Topic":
- **Be specific:** "smartwatch reviews" not just "watches".
- **Focus on expertise:** "AI ethics" not just "AI".
- **Combine terms:** "Dr. Jane Smith diabetes research"
- **Avoid overly broad terms** that might dilute the relevance.
""")


# --- Sidebar for additional settings/inputs ---
with st.sidebar:
    st.header("Global Settings & Data")
    st.markdown("---")
    st.subheader("Competitor Authors List")
    competitor_authors_raw = st.text_area(
        "Enter competitor author names (one per line):",
        value="Jane Doe\nJohn Smith" # Example placeholder
    )
    competitor_authors_list = [name.strip() for name in competitor_authors_raw.split('\n') if name.strip()]

    st.info(f"Loaded {len(competitor_authors_list)} competitor authors in sidebar.") # Debugging log
    st.markdown("---")
    st.markdown("### How it works:")
    st.markdown("- Checks for Google Knowledge Panel & Wikipedia page.")
    st.markdown("- Measures topical authority via Google search count ('Author + Topic').")
    st.markdown("- Identifies other major domains author has written for.")
    st.markdown("- Checks for Google Scholar citations.")
    st.markdown("- Flags if the author is on your competitor list.")
    st.markdown("- Calculates a customizable quality score.")

st.info("UI setup complete.") # Debugging log

# --- Main Content Area ---

# --- Single Author Evaluation ---
st.header("Individual Author Evaluation")
st.markdown("Enter an author's name and relevant details to get an immediate assessment.")
col1, col2 = st.columns(2)
with col1:
    single_author_name = st.text_input("Author Name:", help="Full name of the author.")
    single_linkedin_followers = st.number_input("LinkedIn Followers (manual input):", min_value=0, value=0, step=100)
with col2:
    single_keyword_topic = st.text_input("Relevant Topic/Keyword for Expertise:", help="e.g., 'personal finance', 'climate science'.")
    single_x_followers = st.number_input("X (Twitter) Followers (manual input):", min_value=0, value=0, step=100)
    single_author_url = st.text_input("Optional: Author Profile URL (on your site or personal site):", help="e.g., telegraph.co.uk/authors/jane-doe, or author's personal website.")

if st.button("Analyze Single Author"):
    if single_author_name:
        with st.spinner(f"Analyzing '{single_author_name}'... This may take a moment due to API calls."):
            kp_exists, kp_details = check_knowledge_panel(single_author_name)
            wiki_exists, wiki_details = check_wikipedia(single_author_name)
            topical_authority_serp_count = get_topical_authority_serp_count(single_author_name, single_keyword_topic)
            author_domains = get_author_related_domains(single_author_name)
            scholar_citations_count = check_google_scholar_citations(single_author_name)
            is_competitor = check_competitor_author(single_author_name, competitor_authors_list)

            total_social_followers = single_linkedin_followers + single_x_followers # Simple sum
            quality_score = calculate_quality_score(
                kp_exists, wiki_exists, topical_authority_serp_count,
                scholar_citations_count, total_social_followers, len(author_domains),
                is_competitor
            )

            st.subheader(f"Results for '{single_author_name}'")
            st.markdown(f"**Google Knowledge Panel:** {'✅ Found' if kp_exists else '❌ Not Found'} ({kp_details})")
            st.markdown(f"**Wikipedia Page:** {'✅ Found' if wiki_exists else '❌ Not Found'} ({wiki_details})")
            st.markdown(f"**Topical Authority (SERP count for '{single_author_name} {single_keyword_topic}'):** {topical_authority_serp_count:,}")
            st.markdown(f"**Cited in Google Scholar:** {scholar_citations_count:,} results")
            st.markdown(f"**Associated Brands/Domains (Top {len(author_domains)}):** {', '.join(author_domains) if author_domains else 'None found'}")
            st.markdown(f"**Total Social Media Followers (Manual):** {total_social_followers:,}")
            st.markdown(f"**Competitor Author:** {'⚠️ Yes' if is_competitor else '🟢 No'}")
            st.markdown(f"**Calculated Quality Score:** `{quality_score}`")

            if is_competitor:
                st.warning("This author appears on your competitor list. Exercise caution.")
            elif quality_score >= 20:
                 st.success("Strong authority signals! Highly recommended.")
            elif quality_score >= 10:
                 st.info("Good authority signals. Recommended.")
            else:
                st.warning("Limited strong authority signals. Further manual review is highly recommended.")
    else:
        st.warning("Please enter an author name to analyze.")

st.markdown("---")

# --- Batch Author Evaluation ---
st.header("Batch Author Evaluation (CSV Upload)")
st.markdown("Upload a CSV file containing 'Author', 'Keyword' (optional), 'Author_URL' (optional), 'LinkedIn_Followers' (optional), 'X_Followers' (optional) columns for bulk analysis.")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if "Author" not in data.columns:
            st.error("The CSV file must contain an 'Author' column.")
            st.stop() # Stop execution if critical column is missing

        st.write("Preview of uploaded data:")
        st.dataframe(data.head())

        if st.button("Run Batch Analysis"):
            results = []
            total_authors = len(data)
            progress_bar = st.progress(0)
            status_text = st.empty()

            for index, row in data.iterrows():
                author = str(row["Author"]).strip()
                keyword = str(row["Keyword"]).strip() if "Keyword" in data.columns and pd.notna(row["Keyword"]) else ""
                # Optional manual inputs from CSV
                linkedin_followers = int(row["LinkedIn_Followers"]) if "LinkedIn_Followers" in data.columns and pd.notna(row["LinkedIn_Followers"]) else 0
                x_followers = int(row["X_Followers"]) if "X_Followers" in data.columns and pd.notna(row["X_Followers"]) else 0
                author_url = str(row["Author_URL"]).strip() if "Author_URL" in data.columns and pd.notna(row["Author_URL"]) else ""


                status_text.text(f"Processing: {author} ({index + 1}/{total_authors})")

                # API Calls
                kp_exists, kp_details = check_knowledge_panel(author)
                wiki_exists, wiki_details = check_wikipedia(author)
                topical_authority_serp_count = get_topical_authority_serp_count(author, keyword)
                author_domains = get_author_related_domains(author)
                scholar_citations_count = check_google_scholar_citations(author)

                # Manual Checks / Derived Data
                is_competitor = check_competitor_author(author, competitor_authors_list)
                total_social_followers = linkedin_followers + x_followers

                quality_score = calculate_quality_score(
                    kp_exists, wiki_exists, topical_authority_serp_count,
                    scholar_citations_count, total_social_followers, len(author_domains),
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
                    "Associated_Brands_Domains": ", ".join(author_domains),
                    "Manual_LinkedIn_Followers": linkedin_followers,
                    "Manual_X_Followers": x_followers,
                    "Total_Social_Followers": total_social_followers,
                    "Is_Competitor_Author": "Yes" if is_competitor else "No",
                    "Quality_Score": quality_score
                })
                progress_bar.progress((index + 1) / total_authors)
                time.sleep(1) # Small delay to be mindful of API rate limits and display updates

            results_df = pd.DataFrame(results)
            st.subheader("Batch Analysis Results:")
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
            status_text.success("Batch analysis complete!")

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
    except Exception as e:
        st.error(f"Error processing CSV file or running analysis: {e}")
        st.info("Please ensure your CSV has an 'Author' column and is correctly formatted.")
