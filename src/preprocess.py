import re

def preprocess(text):
    # Basic preprocessing step before feature extraction
    # Similar to cleaning/transformation steps discussed in class
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text