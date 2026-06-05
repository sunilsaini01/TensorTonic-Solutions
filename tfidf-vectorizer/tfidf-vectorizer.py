import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """

    # Tokenize documents
    tokenized_docs = [doc.lower().split() for doc in documents]

    # Build vocabulary (sorted alphabetically)
    vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))

    n_docs = len(documents)
    n_vocab = len(vocabulary)

    # Word -> column index
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    # Document Frequency (DF)
    df = Counter()
    for doc in tokenized_docs:
        for word in set(doc):
            df[word] += 1

    # IDF
    idf = {word: math.log(n_docs / df[word]) for word in vocabulary}

    # TF-IDF Matrix
    tfidf_matrix = np.zeros((n_docs, n_vocab))

    for i, doc in enumerate(tokenized_docs):
        total_terms = len(doc)
        term_counts = Counter(doc)

        for word, count in term_counts.items():
            tf = count / total_terms
            tfidf_matrix[i, word_to_idx[word]] = tf * idf[word]

    return tfidf_matrix, vocabulary