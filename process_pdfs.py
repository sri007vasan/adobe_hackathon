# process_pdfs.py
import os
import json
import pandas as pd
from nltk import pos_tag, word_tokenize
import joblib
import pdfplumber
import numpy as np
from PIL import Image
import fitz # PyMuPDF

# --- REMOVED NLTK Downloads from here ---
# They are now handled in the Dockerfile build process.
# import nltk # You still need to import nltk at the top of the file

# Ensure nltk is imported
import nltk

# --- Functions from your Colab notebook (no changes needed to functions below this point) ---
# ... (all your functions: extract_features_from_text_lines, detect_title_with_pymupdf, predict_headings) ...

def extract_features_from_text_lines(text_lines, mode_font_size):
    rows = []
    for text, font_size in text_lines:
        font_flag = int(font_size > mode_font_size)
        num_words = len(text.split())

        def text_case(t):
            if t.islower(): return 0
            elif t.isupper(): return 1
            elif t.istitle(): return 2
            return 3
        case = text_case(text)

        def count_pos(t, prefix):
            tags = pos_tag(word_tokenize(t))
            return sum(1 for _, tag in tags if tag.startswith(prefix))

        rows.append({
            "text": text,
            "font_flag": font_flag,
            "num_words": num_words,
            "text_case": case,
            "verbs": count_pos(text, 'VB'),
            "nouns": count_pos(text, 'NN'),
            "cardinals": count_pos(text, 'CD')
        })
    return pd.DataFrame(rows)

def detect_title_with_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    heading_candidates = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                merged_text = " ".join([span["text"].strip() for span in spans])
                font = spans[0].get("font", "")
                font_size = spans[0]["size"]
                flags = spans[0].get("flags", 0)
                if font_size >= 13 and merged_text.lower() != "overview":
                    heading_candidates.append({
                        "text": merged_text,
                        "font_size": font_size,
                        "font": font,
                        "flags": flags,
                        "page": page_num + 1
                    })

    # Safe title extraction
    if len(heading_candidates) >= 2:
        # Prioritize larger font size for multi-part titles or just take the first two
        title_parts = sorted(heading_candidates, key=lambda x: x['font_size'], reverse=True)[:2]
        title = " ".join([p["text"] for p in sorted(title_parts, key=lambda x: x['page'])])
    elif len(heading_candidates) == 1:
        title = heading_candidates[0]["text"]
    else:
        title = "Untitled Document"

    return title.strip()


def predict_headings(pdf_path, model):
    text_lines = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            words = page.extract_words(extra_attrs=["size", "top"])

            # Group words by lines using y-position
            lines = {}
            for w in words:
                text = w['text']
                size = w.get('size', 12)
                y = round(w['top'], 1)

                if y not in lines:
                    lines[y] = []
                lines[y].append((w['x0'], text, size))

            for y in sorted(lines.keys()):
                sorted_words = sorted(lines[y], key=lambda x: x[0])
                line_text = ' '.join([w[1] for w in sorted_words])
                font_size = max([w[2] for w in sorted_words])
                text_lines.append((line_text, font_size, page_number + 1))

    # Get mode font size
    font_sizes = [fs for _, fs, _ in text_lines]
    if not font_sizes:
        return []
    mode_font_size = pd.Series(font_sizes).mode()[0]

    # Group by page
    lines_by_page = {}
    for txt, sz, pg in text_lines:
        if pg not in lines_by_page:
            lines_by_page[pg] = []
        lines_by_page[pg].append((txt, sz))

    # Predict headings
    predictions = []
    for pg, lines in lines_by_page.items():
        df_lines = extract_features_from_text_lines(lines, mode_font_size)
        if df_lines.empty:
            continue
        df_lines['prediction'] = model.predict(df_lines[['font_flag', 'num_words', 'text_case', 'verbs', 'nouns', 'cardinals']])
        df_lines['font_size'] = [line[1] for line in lines]  # Reuse font sizes

        for idx, row in df_lines[df_lines['prediction'] == 1].iterrows():
            font_size = row['font_size']
            # Adjusted heading level logic
            if font_size >= mode_font_size * 1.2: # Example: 20% larger than mode font
                level = "H1"
            elif font_size >= mode_font_size * 1.1: # Example: 10% larger than mode font
                level = "H2"
            else:
                level = "H3"
            predictions.append({
                "level": level,
                "text": row['text'],
                "page": pg
            })

    return predictions

# --- Main execution block for Docker ---
if __name__ == "__main__":
    # Define paths for Docker environment
    INPUT_PDF_DIR = "/app/input"
    OUTPUT_JSON_DIR = "/app/output" # Output will be a JSON file for each PDF
    MODEL_PATH = "heading_model.joblib" # Model is copied into /app

    # Ensure output directory exists
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)

    # --- Load the pre-trained model ---
    print(f"Loading pre-trained model from {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Please ensure 'heading_model.joblib' is in the container's /app directory.")
        exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # --- Process PDFs from the input directory ---
    print(f"\nProcessing PDFs from: {INPUT_PDF_DIR}")
    pdf_files_found = False
    for filename in os.listdir(INPUT_PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_files_found = True
            pdf_path = os.path.join(INPUT_PDF_DIR, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_filepath = os.path.join(OUTPUT_JSON_DIR, output_filename)

            print(f"Processing {pdf_path}...")
            try:
                # Detect title
                title = detect_title_with_pymupdf(pdf_path)

                # Predict headings
                headings = predict_headings(pdf_path, model)

                # Use first heading as title if original title is "Untitled Document" and headings exist
                if title == "Untitled Document" and headings:
                    title = headings[0]["text"]
                    headings = headings[1:]  # remove the first heading since it's now used as title

                output_data = {
                    "title": title,
                    "headings": headings
                }

                # Save to output JSON
                with open(output_filepath, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                print(f"✅ Output for {filename} written to {output_filepath}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    if not pdf_files_found:
        print(f"No PDF files found in {INPUT_PDF_DIR}. Please ensure your PDFs are mounted correctly.")
    print("PDF processing script finished.")