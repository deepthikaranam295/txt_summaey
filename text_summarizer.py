import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Tuple, List
import re
from datetime import datetime
from dataclasses import dataclass

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

@dataclass
class WordStats:
    word: str
    count: int
    frequency_percentage: float

@dataclass
class TextAnalysis:
    total_words: int
    unique_words: int
    avg_word_length: float
    sentence_count: int
    avg_words_per_sentence: float
    word_stats: List[WordStats]
    timestamp: str

def summarize_text(text: str, num_sentences: int = 3) -> Tuple[str, str]:
    """
    Generate a summary of the input text using TF-IDF scoring and return both
    the summary and the original text with highlighted summary sentences.

    Args:
        text (str): Input text to summarize
        num_sentences (int): Number of sentences to include in summary

    Returns:
        Tuple[str, str]: (summary, highlighted_text)
    """
    # Handle edge cases
    if not text or not text.strip():
        return "", ""
    
    # Clean the text
    text = text.strip()
    
    # Tokenize into sentences
    sentences = sent_tokenize(text)
    
    # Handle short texts
    if len(sentences) <= num_sentences:
        return text, f"<div style='background-color: #f0f0f0'>{text}</div>"
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        # Handle case where no valid features are found
        return text, f"<div style='background-color: #f0f0f0'>{text}</div>"
    
    # Calculate sentence scores based on TF-IDF values
    sentence_scores = []
    for i in range(len(sentences)):
        score = np.sum(tfidf_matrix[i].toarray())
        sentence_scores.append((i, score))
    
    # Sort sentences by score and get top n
    top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:num_sentences]
    top_indices = sorted([idx for idx, _ in top_sentences])
    
    # Create summary
    summary_sentences = [sentences[idx] for idx in top_indices]
    summary = ' '.join(summary_sentences)
    
    # Create highlighted original text
    highlighted_sentences = []
    for i, sentence in enumerate(sentences):
        if i in top_indices:
            highlighted_sentences.append(
                f"<span style='background-color: #ffeb3b'>{sentence}</span>"
            )
        else:
            highlighted_sentences.append(sentence)
    
    highlighted_text = ' '.join(highlighted_sentences)
    
    return summary, highlighted_text

def get_tldr(text: str) -> Tuple[str, float]:
    """
    Generate a one-sentence TL;DR summary by selecting the highest-scoring sentence.

    Args:
        text (str): Input text to summarize

    Returns:
        Tuple[str, float]: (best_sentence, score)
    """
    # Handle edge cases
    if not text or not text.strip():
        return "", 0.0
    
    text = text.strip()
    sentences = sent_tokenize(text)
    
    if len(sentences) <= 1:
        return text, 1.0
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    try:
        # Get TF-IDF matrix
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Calculate sentence scores
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        
        # Normalize scores to 0-1 range
        if len(sentence_scores) > 0:
            max_score = max(sentence_scores)
            if max_score > 0:
                sentence_scores = sentence_scores / max_score
        
        # Get the highest scoring sentence
        best_idx = np.argmax(sentence_scores)
        return sentences[best_idx], float(sentence_scores[best_idx])
        
    except Exception as e:
        # Fallback to first sentence if TF-IDF fails
        return sentences[0], 1.0

def get_text_stats(text: str) -> dict:
    """Get basic statistics about the text."""
    if not text or not text.strip():
        return {
            "sentences": 0,
            "words": 0,
            "avg_sentence_length": 0
        }
    
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    words = [word for word in words if word.isalnum()]
    
    return {
        "sentences": len(sentences),
        "words": len(words),
        "avg_sentence_length": len(words) / len(sentences) if sentences else 0
    }

def analyze_word_frequency(text: str, top_n: int = 10, include_stopwords: bool = False, min_sentences: int = 3) -> TextAnalysis:
    """
    Generate a summary of the input text using TF-IDF scoring and return both
    the summary and the original text with highlighted summary sentences.

    Args:
        text (str): Input text to summarize
        top_n (int): Number of top words to include in the analysis (default: 10)
        include_stopwords (bool): Whether to include stopwords in the analysis (default: False)
        min_sentences (int): Minimum number of sentences required for the analysis (default: 3)

    Returns:
        TextAnalysis: A dataclass containing the text analysis results
    """
    # Clean and validate input text
    text = text.strip()
    if not text:
        return TextAnalysis(
            total_words=0,
            unique_words=0,
            avg_word_length=0,
            sentence_count=0,
            avg_words_per_sentence=0,
            word_stats=[],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    
    # Handle short texts
    if len(sentences) < min_sentences:
        return TextAnalysis(
            total_words=len(word_tokenize(text)),
            unique_words=len(set(word.lower() for word in word_tokenize(text) if word.isalnum())),
            avg_word_length=sum(len(word) for word in word_tokenize(text) if word.isalnum()) / 
                          len([word for word in word_tokenize(text) if word.isalnum()]) if word_tokenize(text) else 0,
            sentence_count=len(sentences),
            avg_words_per_sentence=len(word_tokenize(text)) / len(sentences) if sentences else 0,
            word_stats=[WordStats(word=s.strip(), count=1, frequency_percentage=100.0/len(sentences)) 
                       for s in sentences],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    # Process normal-length text
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Remove stopwords if specified
    if not include_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Remove non-alphabetic tokens and single characters
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1]
    
    # Ensure we have enough valid tokens
    if len(tokens) < min_sentences:
        return TextAnalysis(
            total_words=len(word_tokenize(text)),
            unique_words=len(set(word.lower() for word in word_tokenize(text) if word.isalnum())),
            avg_word_length=sum(len(word) for word in word_tokenize(text) if word.isalnum()) / 
                          len([word for word in word_tokenize(text) if word.isalnum()]) if word_tokenize(text) else 0,
            sentence_count=len(sentences),
            avg_words_per_sentence=len(word_tokenize(text)) / len(sentences) if sentences else 0,
            word_stats=[WordStats(word=s.strip(), count=1, frequency_percentage=100.0/len(sentences)) 
                       for s in sentences[:min_sentences]],
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    
    # Calculate frequency distribution
    freq_dist = FreqDist(tokens)
    total_words = len(tokens)
    unique_words = len(set(tokens))
    avg_word_length = sum(len(word) for word in tokens) / len(tokens)
    sentence_count = len(sentences)
    avg_words_per_sentence = len(tokens) / len(sentences)
    word_stats = [WordStats(word=word, count=freq, frequency_percentage=freq/total_words*100) 
                 for word, freq in freq_dist.most_common(top_n)]
    
    return TextAnalysis(
        total_words=total_words,
        unique_words=unique_words,
        avg_word_length=avg_word_length,
        sentence_count=sentence_count,
        avg_words_per_sentence=avg_words_per_sentence,
        word_stats=word_stats,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# Example usage
if __name__ == "__main__":
    sample_text = """
    Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret, 
    and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics. 
    The field of NLP has existed for more than 50 years and has roots in the field of linguistics. 
    Modern NLP algorithms are based on machine learning, especially statistical and deep learning methods. 
    These algorithms take as input a large set of "training" examples and produce a model that can process text. 
    Today, NLP is used in many applications such as machine translation, chatbots, and sentiment analysis. 
    The technology continues to evolve, and new breakthroughs are being made regularly in areas like language understanding and generation.
    """
    
    # Generate summary and highlighted text
    summary, highlighted = summarize_text(sample_text)
    stats = get_text_stats(sample_text)
