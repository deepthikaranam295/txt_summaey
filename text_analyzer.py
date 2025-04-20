import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from typing import Dict, List, Tuple
import string
from dataclasses import dataclass
from datetime import datetime

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

    def generate_report(self) -> str:
        """Generate a formatted report of the text analysis."""
        report = [
            "=== TEXT ANALYSIS REPORT ===",
            f"Generated at: {self.timestamp}\n",
            "=== GENERAL STATISTICS ===",
            f"Total Words: {self.total_words:,}",
            f"Unique Words: {self.unique_words:,}",
            f"Average Word Length: {self.avg_word_length:.1f} characters",
            f"Number of Sentences: {self.sentence_count:,}",
            f"Average Words per Sentence: {self.avg_words_per_sentence:.1f}\n",
            "=== TOP WORD FREQUENCIES ===",
            "-" * 40
        ]

        # Add word frequency table
        report.append("Rank  Word          Count     %")
        report.append("-" * 40)
        for i, stat in enumerate(self.word_stats, 1):
            report.append(
                f"{i:2d}.   {stat.word:<12} {stat.count:6d}  {stat.frequency_percentage:5.1f}%"
            )

        return "\n".join(report)

def analyze_word_frequency(text: str, top_n: int = 10, include_stopwords: bool = False) -> TextAnalysis:
    """Analyze word frequency and generate detailed statistics for the input text.

    Args:
        text (str): Input text to analyze
        top_n (int): Number of most frequent words to return (default: 10)
        include_stopwords (bool): Whether to include stopwords in analysis (default: False)

    Returns:
        TextAnalysis object containing detailed statistics and formatted report
    """
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
    
    # Remove stopwords if specified
    if not include_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    
    # Remove non-alphabetic tokens and single characters
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1]
    
    # Calculate frequency distribution
    freq_dist = FreqDist(tokens)
    total_words = len(tokens)
    
    # Calculate word statistics
    word_stats = []
    for word, count in freq_dist.most_common(top_n):
        frequency_percentage = (count / total_words) * 100
        word_stats.append(WordStats(word, count, frequency_percentage))
    
    # Calculate additional statistics
    avg_word_length = sum(len(word) for word in tokens) / total_words if tokens else 0
    avg_words_per_sentence = total_words / len(sentences) if sentences else 0
    
    return TextAnalysis(
        total_words=total_words,
        unique_words=len(set(tokens)),
        avg_word_length=avg_word_length,
        sentence_count=len(sentences),
        avg_words_per_sentence=avg_words_per_sentence,
        word_stats=word_stats,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

# Example usage
if __name__ == "__main__":
    sample_text = """
    Natural language processing (NLP) is a fascinating subfield of linguistics, computer science, and artificial intelligence. 
    It focuses on the interactions between computers and human language. Researchers in NLP develop algorithms to process 
    and analyze large amounts of natural language data. Modern NLP applications include machine translation, sentiment analysis, 
    and chatbots that can understand human queries.
    """
    
    analysis = analyze_word_frequency(sample_text)
    print(analysis.generate_report())
