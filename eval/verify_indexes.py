"""
eval/verify_indexes.py
─────────────────────────────────────────────────────────────────────────────
Checks that every vector store holds exactly as many vectors as its chunk file
has chunks.

Why this is needed: build_index.py embeds in batches and retries on failure. If a
batch exhausts its retries the run continues, leaving a store with FEWER vectors
than chunks. The directory exists and is non-empty, so a naive "is it built?"
check passes and the index looks fine -- while silently missing content. Recall
measured against a half-built index is wrong in a way nothing else would reveal.

Exits non-zero if any index is incomplete, so it can gate a run.
"""
import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

config.load_env()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def main():
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"])
                      for c in config.CONFIGS.values()})
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)

    print("=" * 74)
    print("%-28s %10s %10s   %s" % ("INDEX", "CHUNKS", "VECTORS", "STATUS"))
    print("-" * 74)

    incomplete = []
    for doc_id in config.DOCUMENTS:
        for cs, co in schemes:
            key = config.index_key(doc_id, cs, co)
            persist_dir = config.CHROMA_EVAL_DIR / key
            chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")

            if not chunks_path.exists():
                print("%-28s %10s %10s   NO CHUNK FILE" % (key, "-", "-"))
                incomplete.append(key)
                continue
            n_chunks = len(json.loads(chunks_path.read_text(encoding="utf-8")))

            if not persist_dir.exists() or not any(persist_dir.iterdir()):
                print("%-28s %10d %10s   NOT EMBEDDED" % (key, n_chunks, "-"))
                incomplete.append(key)
                continue

            try:
                store = Chroma(persist_directory=str(persist_dir),
                               embedding_function=embeddings)
                n_vecs = store._collection.count()
            except Exception as e:
                print("%-28s %10d %10s   ERROR %s" % (key, n_chunks, "?", type(e).__name__))
                incomplete.append(key)
                continue

            if n_vecs == n_chunks:
                status = "OK"
            elif n_vecs == 0:
                status = "EMPTY -- rebuild"
                incomplete.append(key)
            else:
                status = "INCOMPLETE (%d missing) -- rebuild" % (n_chunks - n_vecs)
                incomplete.append(key)
            print("%-28s %10d %10d   %s" % (key, n_chunks, n_vecs, status))

    print("=" * 74)
    if incomplete:
        print("%d index(es) need rebuilding:" % len(incomplete))
        for k in incomplete:
            print("   %s" % k)
        print("\nRebuild with:  py -3.12 eval/build_index.py --force-missing")
        sys.exit(1)
    print("All indexes complete: vector count matches chunk count everywhere.")


if __name__ == "__main__":
    main()
