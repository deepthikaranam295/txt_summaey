import streamlit as st
import nltk
from text_summarizer import summarize_text, get_text_stats, get_tldr

st.title("Text Summarizer")


# Input text area
text_input = st.text_area(
    "Type or paste your text here",
    height=200,
    placeholder="Enter your text here..."
)

# Action button
tldr_button = st.button("Generate TL;DR", type="primary")

if text_input:
    if tldr_button:
        try:
            tldr_text, score = get_tldr(text_input)
            st.markdown("### TL;DR Summary")
            st.markdown(f"<div style='margin-bottom: 1em;'><span style='background-color: #0e4c92; color: white; padding: 0.2em 0.5em; border-radius: 3px;'>Score: {score:.2f}</span></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 1.5em; border-radius: 8px; border: 1px solid #ddd; font-size: 1.1em;'>{tldr_text}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top: 1em; font-size: 0.9em; color: #666;'>This sentence represents {score*100:.0f}% of your text's main ideas based on TF-IDF scoring using NLTK and scikit-learn.</div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred while generating TL;DR: {str(e)}")
else:
    if tldr_button:
        st.warning("Please enter text first.")
