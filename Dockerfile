FROM --platform=linux/amd64 python:3.11-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
# tesseract-ocr for pytesseract
# libgl1 for opencv-python-headless
# build-essential and pkg-config if any libs require compilation (good general practice)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- IMPORTANT FIX: NLTK Data Download during build ---
# This ensures the NLTK data is available when the container runs.
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab');nltk.download('averaged_perceptron_tagger_eng')"

# Copy your pre-trained model file
COPY heading_model.joblib .

# Copy your main processing script
COPY process_pdfs.py .

# Command to run the script when the container starts
CMD ["python", "process_pdfs.py"]