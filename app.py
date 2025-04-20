import streamlit as st
import nltk
import PyPDF2
import docx
import spacy
from io import StringIO
import re
from text_summarizer import summarize_text as tfidf_summarize

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except:
    st.warning('Downloading language model for the first time...')
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def preprocess_text(text):
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def generate_summary(text, num_sentences=3):
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Calculate word frequencies
    word_freq = {}
    for word in doc:
        if not word.is_stop and not word.is_punct and not word.is_space:
            word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1
    
    # Calculate sentence scores
    sentence_scores = {}
    for sentence in sentences:
        for word in nlp(sentence):
            if word.text.lower() in word_freq:
                if sentence not in sentence_scores:
                    sentence_scores[sentence] = word_freq[word.text.lower()]
                else:
                    sentence_scores[sentence] += word_freq[word.text.lower()]
    
    # Get top sentences
    summary_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    summary = ' '.join([sentence[0] for sentence in summary_sentences])
    
    return summary

def main():
    st.set_page_config(page_title="Text Summarizer", page_icon="📝")
    
    st.title("Text Summarizer")
    st.write("Upload a document or enter text to get a concise summary!")

    # File upload
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
    
    # Text input
    text_input = st.text_area("Or enter your text here", height=200)
    
    # Summary length selection
    st.subheader("Summary Length")
    col1, col2 = st.columns([3, 1])
    with col1:
        num_sentences = st.slider("Select number of sentences", 1, 10, 3, help="Move the slider to choose how many sentences you want in the summary")
    with col2:
        st.metric("Lines", num_sentences)
    
    # Summary method selection
    summary_method = st.radio(
        "Choose summarization method:",
        ["Frequency-based", "TF-IDF based"]
    )
    
    if st.button("Summarize"):
        if uploaded_file is not None:
            try:
                if uploaded_file.type == "application/pdf":
                    text = extract_text_from_pdf(uploaded_file)
                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(uploaded_file)
                else:  # txt file
                    text = str(uploaded_file.read(), 'utf-8')
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return
        elif text_input:
            text = text_input
        else:
            st.warning("Please either upload a file or enter text to summarize!")
            return
        
        if text:
            with st.spinner("Generating summary..."):
                processed_text = preprocess_text(text)
                
                if summary_method == "Frequency-based":
                    summary = generate_summary(processed_text, num_sentences)
                    st.subheader("Frequency-based Summary")
                    st.write(summary)
                else:  # TF-IDF based
                    summary, highlighted = tfidf_summarize(processed_text, num_sentences)
                    st.subheader("TF-IDF based Summary")
                    st.write(summary)
                    st.subheader("Original Text with Highlighted Key Sentences")
                    st.markdown(highlighted, unsafe_allow_html=True)
                
                # Display text statistics
                st.subheader("Text Statistics")
                doc = nlp(processed_text)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", len([token for token in doc if not token.is_punct]))
                with col2:
                    st.metric("Sentences", len(list(doc.sents)))
                with col3:
                    st.metric("Summary Sentences", num_sentences)

if __name__ == "__main__":
    main()
