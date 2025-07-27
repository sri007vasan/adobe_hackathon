# PDF Title and Heading Detection System

## Overview

This project focuses on detecting the **title** and **headings (up to level H3)** in PDF documents. It uses a hybrid approach combining:

- XGBoost-based heading detection (H1, H2, H3)
- Rule-based title detection using PyMuPDF

The goal is to extract structured, hierarchical heading information in JSON format, useful for document indexing, content analysis, or automated document summarization.

---
## Reference paper:
A Supervised Learning Approach For Heading Detection **doi:** https://doi.org/10.48550/arXiv.1809.01477
---
It was this paper that made us to choose either decision tree or XGboost for heading prediciton.This paper includes creatiion of own dataset by converting random pdf pages into latex format and then using them as lables to train the model.This comparative analysis study that was made in the paper reveals that decisontree or XG Boost performs well and we have finally opted the later model.

## Dataset

We used the **DocBank** dataset, which contains detailed layout annotations for scientific articles in PDF format. Each line is labeled with structural tags like section headings, paragraphs, figures, etc.
**Link:** https://www.kaggle.com/datasets/shashankpy/docbank-layout-segmentation-dataset
```
data/
├── docbank_training_data_gp/       # Training data from DocBank
│   ├── annotations/                # JSON annotation files for each PDF
│   └── images/                     # Corresponding page images
├── docbank_validation_data/        # Validation data from DocBank
│   ├── annotations/                # JSON annotation files for validation
│   └── images/                     # Page images for validation
└── docbank_testing_data_gp/        # Testing data from DocBank
    ├── annotations/                # Annotation files used for testing
    └── images/                     # Images for testing PDFs
```
DocBank provides:

- Bounding boxes
- Font and positional features
- Text annotations with labels

This makes it ideal for training supervised models for layout-based classification.

---

## Preprocessing

To prepare the data:

- **Annotation Parsing**: Extracted text, font size, and coordinates from JSON annotations.
- **Feature Engineering**:
  - Font size
  - Font flags (e.g., bold, italic)
  - Number of words in a line
  - Text case (uppercase, lowercase, title case)
  - Part-of-speech tagging using spaCy: nouns, verbs, cardinals
- **Labeling**:
  - Headings labeled as `1`
  - Non-headings labeled as `0`
- **Train-Test Split**: Dataset was split into training and validation sets.

---

## Model: XGBoost

A supervised classification model using **XGBoost** was trained to identify headings based on the extracted features.

### Training Setup

- **Input**: Feature vectors from each line
- **Output**: Binary label (heading or not)

### Performance

- **Accuracy**: ~86%
- **Precision/Recall**: Balanced, with slightly lower recall for headings due to class imbalance
- Evaluated using a confusion matrix and classification report
we have also tried **DECISION TREE** which also gave similar accuracy but lagged at sme complex cases 
---
###  ⚠️ ⚠️IMPORTANT NOTE!!! ⚠️ ⚠️ :
**THE ACCURACY CAN BE INCREASED IF A PROPER DATASET HAD EXISTED...SINCE IT IS VERY COMMON TO KNOW THAT IN A PAGE THE HEADINGS WILL BE LESSER THAN PARAGRAPHS IT WAS VERY HARD TO TRAIN TO MAKE MODEL TO CLASSIFY HEADING....**
**IN FACT OUR MODEL'S ACCURACY FOR NON-HEADING IS 96% THIS IS BECAUSE THERE IS MORE DATA TO TRAIN FOR NON-HEADING**
**SO WE PROMISE THAT WITH SOME UNBIASED DATASET WE CAN ACHIEVE 95%+ ACCURACY**
## Title Detection using PyMuPDF

For identifying the document title, a heuristic approach using **PyMuPDF (fitz)** was implemented.

### Logic

1. Load the first page
2. Extract all text spans and their attributes
3. Identify lines with:
   - Font size above a threshold (e.g., ≥13)
   - Bold font
   - Non-generic content
4. Select top 1–2 lines as the document title

If no confident title is found, the top heading from the XGBoost model is used as a fallback.

---

## Combined Output
Finally the output given by the model is attached in the heading part of the JSON output file and the output given by the Pymudf is attached at the title part of the JSON output file the combined output file is generated
The final output includes:

- `title`: Document title
- `headings`: A list of detected headings up to level H3 with fields:
  - `text`
  - `level` (H1, H2, H3)
  - `page`

Example JSON structure:

```json
{
  "title": "Application form for grant of LTC advance",
  "headings": [
    {
      "level": "H2",
      "text": "Application form for grant of LTC advance",
      "page": 1
    },
    {
      "level": "H3",
      "text": "Name of the Government Servant",
      "page": 1
    }
  ]
}
```


