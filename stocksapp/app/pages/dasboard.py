import streamlit as st
looker_studio_url = "https://lookerstudio.google.com/s/tMaPPw4oro4"

# Embed the dashboard in Streamlit using an iframe
st.markdown(
    f'<iframe src="{looker_studio_url}" width="100%" height="800"></iframe>',
    unsafe_allow_html=True)