import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize words, convert to lowercase, remove stopwords and punctuation
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words]
    
    # Calculate word frequency
    freq_dist = FreqDist(words)
    
    # Score sentences based on word frequency
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in freq_dist:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = freq_dist[word]
                else:
                    sentence_scores[sentence] += freq_dist[word]
    
    # Get top N sentences
    summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
    return summary_sentences

# Streamlit UI
st.title("Text Summarizer")

# Text input
text = st.text_area("Type or paste your text here", height=200)

# Number of sentences selector
num_sentences = st.slider("Select number of sentences for summary", min_value=1, max_value=10, value=3)

# Add custom CSS for red buttons
st.markdown("""
<style>
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
    }
    .stButton > button:hover {
        background-color: #ff3333;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Create columns for buttons
col1, col2 = st.columns(2)

# Process buttons
summary_button = col1.button("Summarize")
tldr_button = col2.button("TL;DR")

if text.strip():
    try:
        if summary_button or tldr_button:
            # Get summary
            summary_sentences = preprocess_text(text)
            
            # Set number of sentences based on button
            if tldr_button:
                num_sentences = min(3, len(summary_sentences))  # TLDR is always 3 sentences or less
            
            # Display summary
            st.subheader("Summary:" if summary_button else "TL;DR:")
            summary = ' '.join([sentence for sentence, score in summary_sentences[:num_sentences]])
            st.write(summary)
            
            # Display original text length vs summary length
            original_length = len(word_tokenize(text))
            summary_length = len(word_tokenize(summary))
            st.info(f"Original text: {original_length} words\nSummary: {summary_length} words\nReduction: {((original_length - summary_length) / original_length * 100):.1f}%")
                
    except Exception as e:
        st.error(f"An error occurred while processing the text: {str(e)}")
else:
    if summary_button or tldr_button:
        st.warning("Please enter text first.")