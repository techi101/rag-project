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
# WHAT THIS FILE IS: the "stock checker" of the evaluation warehouse. Before we trust any score, it counts
# what is actually on the shelves: for every saved vector store it compares "how many chunks should be
# inside" (from the chunk file) with "how many vectors are really inside" (from ChromaDB).
# A CHUNK = one small piece of a PDF's text (about 1000 characters in the baseline setting).
# A VECTOR (embedding) = a list of numbers (3072 for gemini-embedding-001) that captures the meaning of one chunk.
# A VECTOR STORE = a database (here ChromaDB) that keeps one vector per chunk so we can search by meaning.
# Real example from eval/README.md: the index "bi_cs1000_co200" held 200 of 275 vectors while looking complete,
# so this script would print it as "INCOMPLETE (75 missing) -- rebuild".
# Overall flow: list chunking schemes -> for each document and scheme: read chunk file -> open store -> count
# vectors -> compare -> print a table -> exit with code 1 if anything needs rebuilding.
#
# json: reads the "<key>_chunks.json" file (a list of chunks) so we can count them.
import json
# sys: lets us change the import path and exit with an error code (sys.exit(1)).
import sys
# pathlib: easy file paths (folder of this file, folder of each store).
import pathlib

# Add the eval/ folder to Python's search path, so "import config" finds eval/config.py
# even when the script is started from the project root (py -3.12 eval/verify_indexes.py).
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: shared settings (documents, configurations, folders, model names).
import config

# Read GOOGLE_API_KEY and GROQ_API_KEY from the project's .env file into environment variables.
# This must happen BEFORE the Google library is created, because the library reads the key from the environment.
config.load_env()

# langchain_chroma.Chroma: LangChain's wrapper around ChromaDB, used here only to open a saved store and count it.
from langchain_chroma import Chroma
# GoogleGenerativeAIEmbeddings: the Google embedding model (gemini-embedding-001). Chroma needs an embedding
# function to open a store, even though counting vectors makes no API call.
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# IN: nothing (reads config + files on disk) -> OUT: a printed table; exits with code 1 if any index is incomplete.
# WHY: a half-built store still has a non-empty folder, so "does the folder exist?" is not proof it is complete.
# Only counting vectors against chunks proves it.
# Example: key "mysql_cs1000_co200" with 71 chunks and 71 vectors -> "OK".
def main():
    # A chunking SCHEME = one (chunk_size, chunk_overlap) pair. The set {...} removes duplicates: the 6 configs
    # in config.CONFIGS share only 3 schemes -> (500, 100), (1000, 200), (2000, 400). sorted() gives a fixed order.
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"])
                      for c in config.CONFIGS.values()})
    # One embedding object, reused to open every store.
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)

    # Print the table header. "=" * 74 draws a line 74 characters wide (just the width of the table).
    # "%-28s" = text left-aligned in 28 characters; "%10s" = text right-aligned in 10 characters.
    print("=" * 74)
    print("%-28s %10s %10s   %s" % ("INDEX", "CHUNKS", "VECTORS", "STATUS"))
    print("-" * 74)

    # Names of the indexes that are missing or broken; filled in by the loops below.
    incomplete = []
    # Outer loop: every document in the corpus (e.g. "mysql", "bi", "objrec").
    for doc_id in config.DOCUMENTS:
        # Inner loop: every chunking scheme, so each document is checked at 500, 1000 and 2000 characters.
        for cs, co in schemes:
            # key = the index name, e.g. "mysql_cs1000_co200" (cs = chunk size, co = chunk overlap).
            # persist_dir = the store's folder on disk; chunks_path = the JSON list of chunks written by build_index.py.
            key = config.index_key(doc_id, cs, co)
            persist_dir = config.CHROMA_EVAL_DIR / key
            chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")

            # Case 1: no chunk file at all -> the document was never even split. Mark it and go to the next scheme.
            if not chunks_path.exists():
                print("%-28s %10s %10s   NO CHUNK FILE" % (key, "-", "-"))
                incomplete.append(key)
                continue
            # Count the chunks: the chunk file is a JSON list, so its length is the number of chunks.
            n_chunks = len(json.loads(chunks_path.read_text(encoding="utf-8")))

            # Case 2: the store folder is missing or empty -> chunks exist but were never embedded.
            # any(persist_dir.iterdir()) is True if the folder has at least one file in it.
            if not persist_dir.exists() or not any(persist_dir.iterdir()):
                print("%-28s %10d %10s   NOT EMBEDDED" % (key, n_chunks, "-"))
                incomplete.append(key)
                continue

            # Case 3: try to open the store and count its vectors. _collection.count() asks ChromaDB how many vectors it holds.
            # try/except: if the store is corrupted or cannot be opened, report the error's type name instead of crashing,
            # so the rest of the table still prints.
            try:
                store = Chroma(persist_directory=str(persist_dir),
                               embedding_function=embeddings)
                n_vecs = store._collection.count()
            except Exception as e:
                print("%-28s %10d %10s   ERROR %s" % (key, n_chunks, "?", type(e).__name__))
                incomplete.append(key)
                continue

            # Compare the two counts:
            # equal -> OK (one vector for every chunk).
            if n_vecs == n_chunks:
                status = "OK"
            # zero vectors -> the store is EMPTY and must be rebuilt.
            elif n_vecs == 0:
                status = "EMPTY -- rebuild"
                incomplete.append(key)
            # some but not all -> INCOMPLETE; show how many are missing (chunks minus vectors).
            else:
                status = "INCOMPLETE (%d missing) -- rebuild" % (n_chunks - n_vecs)
                incomplete.append(key)
            # Print one row of the table: index name, chunk count, vector count, status.
            print("%-28s %10d %10d   %s" % (key, n_chunks, n_vecs, status))

    print("=" * 74)
    # If anything is broken: list each index, print the rebuild command, and exit with code 1.
    # A non-zero exit code means "failed", so another script or a person can stop before running the evaluation.
    if incomplete:
        print("%d index(es) need rebuilding:" % len(incomplete))
        # Print each broken index name on its own line.
        for k in incomplete:
            print("   %s" % k)
        print("\nRebuild with:  py -3.12 eval/build_index.py --force-missing")
        sys.exit(1)
    # Nothing broken -> print the all-clear message (exit code 0 by default).
    print("All indexes complete: vector count matches chunk count everywhere.")


# Run main() only when this file is started directly, not when another file imports it.
if __name__ == "__main__":
    main()
