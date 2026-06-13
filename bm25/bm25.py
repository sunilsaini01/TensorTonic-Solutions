import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """

    # Empty corpus
    if len(docs) == 0:
        return np.array([])

    N = len(docs)

    # Document lengths
    doc_lengths = []
    for doc in docs:
        doc_lengths.append(len(doc))

    avgdl = np.mean(doc_lengths)

    # Document frequency of each term
    df = {}

    for doc in docs:
        unique_words = set(doc)

        for word in unique_words:
            if word not in df:
                df[word] = 1
            else:
                df[word] += 1

    scores = []

    for i in range(N):

        score = 0
        doc = docs[i]
        tf = Counter(doc)

        for term in query_tokens:

            # term not present anywhere in corpus
            if term not in df:
                continue

            term_freq = tf.get(term, 0)

            if term_freq == 0:
                continue

            dft = df[term]

            idf = math.log((N - dft + 0.5) / (dft + 0.5) + 1)

            numerator = term_freq * (k1 + 1)

            denominator = term_freq + k1 * (
                1 - b + b * (len(doc) / avgdl)
            )

            score += idf * (numerator / denominator)

        scores.append(score)

    return np.array(scores)