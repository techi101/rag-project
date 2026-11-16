"""
eval/verify/verify_indexes.py
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
# WHAT THIS FILE IS: for every saved vector store, compares chunk count (chunk file) vs vector count (ChromaDB).
# chunk = small piece of PDF text (~1000 chars); vector = 3072 numbers for one chunk's meaning; store = ChromaDB.
# Example (README): "bi_cs1000_co200" held 200 of 275 vectors -> "INCOMPLETE (75 missing) -- rebuild".
# Flow: list schemes -> per PDF x scheme: count chunks -> open store -> count vectors -> table -> exit 1 if broken.

import json      # read <key>_chunks.json to count chunks
import sys       # sys.path + sys.exit(1)
import pathlib   # file paths

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))  # add eval/ so "import config" works
import config    # eval/config.py: DOCUMENTS, CONFIGS, folders, models

config.load_env()  # API keys into env, BEFORE the Google client is made

from langchain_chroma import Chroma                              # opens a saved ChromaDB store
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Chroma needs it to open; counting makes no API call


# IN: config + files on disk -> OUT: printed table; exit 1 if any index incomplete
# e.g. "mysql_cs1000_co200": 71 chunks, 71 vectors -> "OK"
def main():
    # ── Step 1: setup ──
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"])
                      for c in config.CONFIGS.values()})  # 6 configs -> 3 schemes: (500,100) (1000,200) (2000,400)
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)  # reused for every store

    print("=" * 74)                                        # table top line
    print("%-28s %10s %10s   %s" % ("INDEX", "CHUNKS", "VECTORS", "STATUS"))  # header
    print("-" * 74)

    # ── Step 2: check each PDF x scheme ──
    incomplete = []                                        # broken index names
    for doc_id in config.DOCUMENTS:                        # e.g. "mysql", "bi", "objrec"
        for cs, co in schemes:                             # cs = chunk size, co = overlap
            key = config.index_key(doc_id, cs, co)         # e.g. "mysql_cs1000_co200"
            persist_dir = config.CHROMA_EVAL_DIR / key     # store folder
            chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")  # chunk list from build_index.py

            if not chunks_path.exists():                   # case 1: never split
                print("%-28s %10s %10s   NO CHUNK FILE" % (key, "-", "-"))
                incomplete.append(key)
                continue
            n_chunks = len(json.loads(chunks_path.read_text(encoding="utf-8")))  # chunks expected

            if not persist_dir.exists() or not any(persist_dir.iterdir()):  # case 2: folder missing/empty
                print("%-28s %10d %10s   NOT EMBEDDED" % (key, n_chunks, "-"))
                incomplete.append(key)
                continue

            try:                                           # case 3: open + count
                store = Chroma(persist_directory=str(persist_dir),
                               embedding_function=embeddings)  # open saved store
                n_vecs = store._collection.count()         # vectors inside
            except Exception as e:                         # corrupt store: report, keep going
                print("%-28s %10d %10s   ERROR %s" % (key, n_chunks, "?", type(e).__name__))
                incomplete.append(key)
                continue

            # ── Step 3: compare counts ──
            if n_vecs == n_chunks:                         # one vector per chunk
                status = "OK"
            elif n_vecs == 0:                              # nothing inside
                status = "EMPTY -- rebuild"
                incomplete.append(key)
            else:                                          # partly built
                status = "INCOMPLETE (%d missing) -- rebuild" % (n_chunks - n_vecs)  # e.g. 75 missing
                incomplete.append(key)
            print("%-28s %10d %10d   %s" % (key, n_chunks, n_vecs, status))  # one table row

    # ── Step 4: verdict ──
    print("=" * 74)
    if incomplete:                                         # anything broken?
        print("%d index(es) need rebuilding:" % len(incomplete))
        for k in incomplete:                               # list them
            print("   %s" % k)
        print("\nRebuild with:  py -3.12 eval/build_index.py --force-missing")
        sys.exit(1)                                        # non-zero = failed, gates a run
    print("All indexes complete: vector count matches chunk count everywhere.")


if __name__ == "__main__":  # only when run directly
    main()                  # start
