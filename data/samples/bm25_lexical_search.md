# BM25 Lexical Search Notes

## Okapi BM25

**Okapi BM25** is a probabilistic ranking function for lexical relevance. It scores documents using **term frequency (TF)**, **inverse document frequency (IDF)**, and document length normalization. Rare terms with high IDF receive stronger weight when they match the query.

## rank_bm25 Integration

This project indexes tokenized document chunks with the **`rank_bm25`** Python package. At query time, the BM25 index returns a ranked list independent of embedding similarity. Lexical matching is especially effective for:

- Exact product or API names (`StateGraph`, `rank_bm25`, `ChromaDB`)
- Acronyms and version strings (`RRF_K`, `MiniLM-L6-v2`)
- Keyword-heavy technical queries where semantic drift would hurt recall

## Tokenization

Chunks are produced with tiktoken (`cl100k_base`) using `CHUNK_SIZE=512` tokens and `CHUNK_OVERLAP=64`. The BM25 index operates on the same chunk texts ingested into ChromaDB so both retrievers stay aligned.
