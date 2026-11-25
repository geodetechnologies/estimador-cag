"""Compare two texts using the project's OpenAI embedding pipeline."""

import argparse
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import sys

# Support execution by path without requiring an installed project package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from app.config import get_settings
from app.embedding_pipeline.embedder import OpenAIEmbedder


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity for finite, nonzero vectors of equal length."""
    if not a or len(a) != len(b) or not all(math.isfinite(x) for x in (*a, *b)):
        raise ValueError('Invalid vectors')
    norm_a, norm_b = math.hypot(*a), math.hypot(*b)
    if not norm_a or not norm_b or not math.isfinite(norm_a) or not math.isfinite(norm_b):
        raise ValueError('Invalid vectors')
    return math.fsum((x / norm_a) * (y / norm_b) for x, y in zip(a, b, strict=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Compare embeddings using paid OpenAI calls; rate-limit retries may incur additional requests.')
    parser.add_argument('--text-a', required=True, help='First text')
    parser.add_argument('--text-b', required=True, help='Second text')
    args = parser.parse_args(argv)
    try:
        key = get_settings().openai_api_key
        if key is None:
            raise ValueError('Missing API key')
        client = OpenAI(api_key=key.get_secret_value(), max_retries=0)
        embedder = OpenAIEmbedder(client=client)
        # Operational batch logs are not part of the CLI's stable output.
        with redirect_stdout(io.StringIO()):
            a = embedder.embed_one(args.text_a)
            b = embedder.embed_one(args.text_b)
        similarity = cosine_similarity(a, b)
    except Exception:
        print('Embedding comparison failed', file=sys.stderr)
        return 1
    print(f'Text A: {args.text_a}\nText B: {args.text_b}\nCosine similarity: {similarity:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
