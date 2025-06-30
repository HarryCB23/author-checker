import streamlit as st
import requests
import json

st.set_page_config(layout="centered", page_title="DataForSEO Test")
st.title("DataForSEO API Test App")
st.markdown("This app tests connectivity to the DataForSEO API and retrieves a basic SERP result.")

# --- API Configuration (Read from Streamlit Secrets) ---
try:
    API_USERNAME = st.secrets["API_USERNAME"]
    API_PASSWORD = st.secrets["API_PASSWORD"]
    st.success("API secrets loaded successfully.")
except KeyError as e:
    st.error(f"Missing API secret: {e}. Please configure secrets in Streamlit Cloud dashboard.")
    st.stop() # Stop the app if secrets are missing

DATAFORSEO_URL = "https://api.dataforseo.com/v3/serp/google/organic/live/regular"

# --- Test API Call Function ---
@st.cache_data(ttl=3600) # Cache results for 1 hour
def run_dataforseo_test_call(test_keyword: str) -> dict:
    payload = {
        "keyword": test_keyword,
        "language_code": "en",
        "location_name": "United Kingdom",
        "device": "desktop"
    }
    st.info(f"Attempting API call for keyword: '{test_keyword}'...")
    try:
        response = requests.post(DATAFORSEO_URL, auth=(API_USERNAME, API_PASSWORD), json=[payload], timeout=30)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        st.success("API call successful!")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API call failed: {e}")
        # Try to include response text if available for more context
        error_details = {"error": str(e)}
        if 'response' in locals() and response is not None:
             error_details["status_code"] = response.status_code
             error_details["response_text"] = response.text
        st.json(error_details)
        return {"error": str(e)}
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return {"error": str(e)}

# --- UI for Triggering Test ---
st.subheader("Trigger a Test API Call")
keyword_input = st.text_input("Enter a keyword to test (e.g., 'best headphones'):", "test keyword")

if st.button("Run DataForSEO Test"):
    if keyword_input:
        with st.spinner("Making API call to DataForSEO..."):
            result = run_dataforseo_test_call(keyword_input)
            st.subheader("API Response:")
            st.json(result) # Display the full JSON response
    else:
        st.warning("Please enter a keyword to test.")

st.markdown("---")
st.caption("This app is for testing DataForSEO API connectivity. **Remember to secure your API keys!**")
