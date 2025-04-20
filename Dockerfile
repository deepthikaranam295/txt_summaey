# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create NLTK data directory and set permissions
RUN mkdir -p /usr/local/share/nltk_data && \
    chmod -R 777 /usr/local/share/nltk_data

# Download NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords

# Skip NLTK verification for now since we already downloaded the data
RUN echo "NLTK data already downloaded"

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose Streamlit port
EXPOSE 8501

# Create a non-root user
RUN useradd -m -r streamlit
RUN chown -R streamlit:streamlit /app
USER streamlit

# Command to run the application
CMD ["streamlit", "run", "--server.address", "0.0.0.0", "--server.port", "8501", "streamlit_app.py"]
