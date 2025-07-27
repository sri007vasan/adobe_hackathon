PDF Title and Heading Detection System
Overview
This project focuses on detecting the title and headings (up to level H3) in PDF documents. The model is trained using the DocBank dataset, and predictions are made using a trained XGBoost classifier. Additionally, PyMuPDF (fitz) is used to detect the document title based on font properties and visual layout heuristics.

Dataset
We used the DocBank dataset, which contains detailed layout annotations for PDF documents including structural information like section headings, paragraphs, etc. The dataset provides bounding box positions, text, and labels, which makes it suitable for training models to detect headers and structural components of a document.

Preprocessing
Parsing annotations: The dataset was parsed to extract relevant features such as text content, font size, and position.

Feature extraction:

Font size

Font flags (bold, italic etc.)

Word count in a line

Text case analysis (uppercase, lowercase, title case)

POS tagging (verbs, nouns, cardinals using spaCy)

Labeling: Headings in the dataset were labeled as 1 and normal text as 0.

The dataset was split into training and validation sets before training.

Model: XGBoost
An XGBoost classifier was trained for heading detection using the extracted features. It outperformed simple Decision Trees due to its ability to capture non-linear relationships and apply boosting to improve accuracy.

Model Training
Input: Feature vectors from each line of text

Target: Binary label indicating whether the line is a heading or not

Evaluation:

Accuracy: ~86%

Precision: Higher for non-headings, slightly lower for headings due to class imbalance

Confusion matrix and classification report were used for evaluation

Title Detection using PyMuPDF
For document title detection, a rule-based method using PyMuPDF (fitz) was used. The title is assumed to be among the first lines in the first page with the largest font size and bold style.

Steps:
Open the PDF with PyMuPDF.

Extract blocks of text from the first page.

Identify lines with font size above a certain threshold and bold styling.

Select the top 1 or 2 lines with the largest font size as the title.

Combined Output
The final script combines both approaches:

detect_title_with_pymupdf(pdf_path) extracts the title.

predict_headings(pdf_path, model) uses the trained model to detect all headings (H1–H3).

If the title predicted by PyMuPDF is missing or too generic, the top heading is used as a fallback.

The output is written into a structured JSON file with fields:

"title": Detected title

"headings": List of heading objects with "level", "text", and "page".


