import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping token-based chunks."""
    tokens = _ENCODING.encode(text)
    if not tokens:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(_ENCODING.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = end - chunk_overlap

    return chunks
