from rank_bm25 import BM25Okapi

# Dummy corpus for demo; in real project, load from ES/Wiki/news dataset
CORPUS = [
    "The sky is blue.",
    "Water boils at 100 degrees Celsius.",
    "The president announced new policies."
]

tokenized_corpus = [doc.lower().split() for doc in CORPUS]
bm25 = BM25Okapi(tokenized_corpus)

def retrieve(query: str, top_k: int = 5):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [CORPUS[i] for i in top_indices]
