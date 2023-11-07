import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def ngram_tfidf(X, model=None):
    if model == None:
        kwargs = {
            'ngram_range': (1, 2), 
            'dtype': 'int32',
            'decode_error': 'replace',
            'analyzer': 'word',
            'min_df': 2,
        }
        model = TfidfVectorizer(**kwargs)
        X = model.fit_transform(X)
        
    else:
        X = model.transform(X)
    return X, model


def bertimbau(X, model=None):
    if model == None:
        model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
        model.max_seq_length = 128
        
    return np.array(model.encode(X)), model