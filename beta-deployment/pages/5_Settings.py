from Nuggt_Playground import *
from sidebar_functions import sidebar_logo

st.set_page_config(page_title="Settings", layout="wide")

sidebar_logo("nuggt-logo.png")

st.subheader("Settings")

col1, col2 = st.columns(2)

with col1:
    openai_key = st.session_state.openai_api_key if "openai_api_key" in st.session_state else "" 
    openai_api_key = st.text_input("OpenAI API Key", value=openai_key, type="password")
    st.caption("*Required for all apps; get it [here](https://platform.openai.com/account/api-keys).*")

    serper_key = st.session_state.serper_api_key if "serper_api_key" in st.session_state else "" 
    serper_api_key = st.text_input("Serper API Key", value=serper_key, type="password")
    st.caption("*Required for all apps; get it [here](https://serpapi.com/).*")

    scenex_key = st.session_state.scenex_api_key if "scenex_api_key" in st.session_state else "" 
    scenex_api_key = st.text_input("Scenex API Key", value=scenex_key, type="password")
    st.caption("*Required for all apps; get it [here](https://scenex.jina.ai/?ref&ref=steemhunt).*")

with col2:
    google_key = st.session_state.google_api_key if "google_api_key" in st.session_state else "" 
    google_api_key = st.text_input("Google API Key", value=google_key, type="password")
    st.caption("*Required for all apps; get it [here](https://support.google.com/googleapi/answer/6158862?hl=en).*")

    google_cse_key = st.session_state.google_cse_api_key if "google_cse_api_key" in st.session_state else "" 
    google_cse_api_key = st.text_input("Google CSE Key", value=google_cse_key, type="password")
    st.caption("*Required for all apps; get it [here](https://developers.google.com/custom-search/v1/introduction).*")


if st.button("Save"):
    keys = [
        ("OpenAI API Key", openai_api_key, "openai_api_key"),
        ("Serper API Key", serper_api_key, "serper_api_key"),
        ("Scenex API Key", scenex_api_key, "scenex_api_key"),
        ("Google API Key", google_api_key, "google_api_key"),
        ("Google CSE Key", google_cse_api_key, "google_cse_api_key")
    ]

    missing_keys = [name for name, value, _ in keys if not value.strip()]
    
    if missing_keys:
        for key in missing_keys:
            st.error(f"Please provide the missing {key}.")
    else:
        for _, value, state_key in keys:
            st.session_state[state_key] = value
        st.success("API keys saved successfully.")












