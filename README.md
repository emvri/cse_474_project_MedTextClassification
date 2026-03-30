# CSE 474 Project - Biomedical Text Classification

## Project task
Classify sentences from the PubMed 200k RCT dataset into:
- BACKGROUND
- OBJECTIVE
- METHODS
- RESULTS
- CONCLUSIONS

## Setup

conda create -n cse474 python=3.11
conda activate cse474
pip install -r requirements.txt

## Dataset

PubMed 200k RCT dataset.
Download from:
https://github.com/Franck-Dernoncourt/pubmed-rct

Place files in:
data/train.txt
data/test.txt
data/dev.txt


## Current pipeline
- Preprocessing
- Bag-of-Words and TF-IDF features
- Naive Bayes
- Logistic Regression

## Next task
Add SVM into the same pipeline for fair comparison.

## How to run
From the project root:

```bash
python src/main.py