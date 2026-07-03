import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=50):
        self.vector_size = vector_size
        self.model = None
    
    def fit(self, X, y=None):
        # Train Word2Vec model on the token lists
        self.model = Word2Vec(sentences=X, vector_size=self.vector_size, window=5, min_count=1, workers=4)
        return self
    
    def get_document_vector(self, tokens):
        """Get average vector for a document"""
        # Get vectors for all words that exist in vocabulary
        word_vectors = []
        for word in tokens:
            if word in self.model.wv:
                word_vectors.append(self.model.wv[word])
        
        # If no words found, return zero vector
        if len(word_vectors) == 0:
            return np.zeros(self.vector_size)
        
        # Return average of all word vectors
        return np.mean(word_vectors, axis=0)
    
    def transform(self, X):
        # X should be a list of token lists
        vectors = []
        for tokens in X:
            doc_vector = self.get_document_vector(tokens)
            vectors.append(doc_vector)
        return np.vstack(vectors)  # Shape: (n_samples, vector_size)
