import spacy
from scipy.sparse import hstack
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load("en_core_web_sm")

def extract_pos_dep_features(corpus):
    pos_features = []
    dep_features = []
    for sentence in tqdm(corpus):
        doc = nlp(sentence)
        pos_tags = [token.pos_ for token in doc]
        dep_rels = [token.dep_ for token in doc]
        pos_features.append(" ".join(pos_tags))
        dep_features.append(" ".join(dep_rels))
    return pos_features, dep_features


def feature_engineering(corpus):
    pos_features, dep_features = extract_pos_dep_features(corpus)

    pos_vectorizer = TfidfVectorizer()
    pos_tfidf = pos_vectorizer.fit_transform(pos_features)

    dep_vectorizer = TfidfVectorizer()
    dep_tfidf = dep_vectorizer.fit_transform(dep_features)

    vectorizer = TfidfVectorizer(max_features=10000)
    tfidf_features = vectorizer.fit_transform(corpus).toarray()

    # print(pos_tfidf.shape, dep_tfidf.shape, tfidf_features.shape)
    # combined_features = np.concatenate((pos_tfidf, dep_tfidf,tfidf_features), axis=1)
    combined_features = hstack([pos_tfidf, dep_tfidf, tfidf_features])
    combined_features = combined_features.tocsr()
    return combined_features
