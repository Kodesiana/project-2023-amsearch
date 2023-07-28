import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


class Embeddings:

    def __init__(self):
        self.tfidf: TfidfVectorizer = None
        self.bert: SentenceTransformer = None

    def load(self, data_dir: str):
        self.tfidf = joblib.load(
            os.path.join(data_dir, "tfidf-sunda-model.joblib"))
        self.bert = SentenceTransformer.load(
            os.path.join(data_dir, "bert-sunda-ams"))

    def extract(self, engine: str, q: str):
        if engine == 'tfidf':
            return self.extract_tfidf(q)
        elif engine == 'bert':
            return self.extract_bert(q)
        else:
            raise ValueError("Unknown engine")

    def extract_tfidf(self, q: str):
        return self.tfidf.transform([q]).toarray()[0]

    def extract_bert(self, q: str):
        return self.bert.encode([q])[0]
