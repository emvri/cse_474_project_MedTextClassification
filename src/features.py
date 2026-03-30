from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_bow(texts):
    vec = CountVectorizer(max_features=5000)
    return vec.fit_transform(texts), vec

def get_tfidf(texts):
    vec = TfidfVectorizer(max_features=5000)
    return vec.fit_transform(texts), vec