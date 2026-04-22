from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_bow(texts):
    # Bag-of-Words (counts word frequency)
    vec = CountVectorizer(max_features=5000)
    return vec.fit_transform(texts), vec

def get_tfidf(texts):
    # In-class concept: TF-IDF (weights words by importance)
    # TF = frequency, IDF = rarity across documents
    # Referenced from class activity comparing BoW vs TF-IDF
    vec = TfidfVectorizer(max_features=5000)
    return vec.fit_transform(texts), vec