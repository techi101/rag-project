"""
eval/build_index.py
─────────────────────────────────────────────────────────────────────────────
Builds one vector store per (document, chunk_size, chunk_overlap) combination.

Configurations that share a chunking scheme share an index -- "baseline", "k8"
and "hybrid" all use 1000/200, so we embed that once and vary only retrieval at
query time. Six configs therefore need three indexes per document, not six.

Everything is written to eval/chroma_eval/. The live app's ../chroma_store/ is
never touched.

Chunks are also dumped to eval/chroma_eval/<key>_chunks.json so the hybrid
(BM25 + dense) retriever can build its keyword index without re-parsing the PDF.
"""
# WHAT THIS FILE IS: the "warehouse builder" of the evaluation. Before any question can be tested, each PDF must be
# cut into chunks and stored as embeddings. This script does that once per (document, chunk size, overlap) and
# saves the result in eval/chroma_eval/, so later runs just open the ready-made store.
# A CHUNK = a small piece of a page (up to chunk_size characters). OVERLAP = characters repeated between two
# neighbouring chunks so a sentence cut at the edge is not lost.
# Real example: the "bi" document with 1000/200 becomes the folder bi_cs1000_co200 (275 chunks per eval/README.md)
# plus bi_cs1000_co200_chunks.json (the plain text of every chunk, used by the BM25 keyword search).
# Overall flow: for each document -> for each chunking scheme -> already complete? skip
#   -> else PDF -> pages -> chunks -> save chunks.json -> embed in batches of 10 -> ChromaDB -> count vectors
#
# json = turn Python lists/dicts into text and back (used for the <key>_chunks.json files).
import json
# shutil = high-level file operations; here shutil.rmtree deletes a whole folder (a broken index).
import shutil
# sys = read command-line flags like --force and --chunks-only (sys.argv), and change the import path.
import sys
# time = pause between API calls (time.sleep) so the free tier's per-minute limit is not crossed.
import time
# pathlib = file paths that work on Windows and Linux.
import pathlib

# Add the eval/ folder to Python's import search list, so "import config" finds eval/config.py
# even when the script is started from the project root (py -3.12 eval/build_index.py).
sys.path.insert(0, str(pathlib.Path(__file__).parent))
# config = eval/config.py: folders, model names, CONFIGS, DOCUMENTS, index_key().
import config

# Load GOOGLE_API_KEY / GROQ_API_KEY from .env BEFORE the Google library is used, because it reads the key from os.environ.
config.load_env()

# PyPDFLoader (LangChain) = reads a PDF and returns one Document per page, with the page index in metadata["page"].
from langchain_community.document_loaders import PyPDFLoader
# RecursiveCharacterTextSplitter = cuts text into chunks, trying to break at paragraphs first, then lines, sentences, words.
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma = the LangChain wrapper around ChromaDB, a vector database that runs locally and saves to a folder on disk.
from langchain_chroma import Chroma
# GoogleGenerativeAIEmbeddings = calls Google's gemini-embedding-001 to turn text into vectors (lists of numbers).
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Batch size for embedding requests. 50 exceeds the free tier's per-minute limit
# outright; 20 still trips it partway through a long document. 10 with a longer
# pause sustains ~200 chunks without RESOURCE_EXHAUSTED.
# EMBED_BATCH = 10 chunks are sent to Google per request (reason in the comment above: 50 and 20 hit the rate limit).
# A RATE LIMIT = the most requests a free API allows per minute; going over gives the error RESOURCE_EXHAUSTED.
EMBED_BATCH = 10
# BATCH_PAUSE_S = wait 3 seconds after each batch, for the same reason.
BATCH_PAUSE_S = 3.0


# IN: path of a store folder  ->  OUT: how many vectors (embedded chunks) it holds, or -1 if it cannot be opened.
# WHY: "built" must mean "has one vector per chunk". Counting tells a complete store from a half-built one.
# Example: bi_cs1000_co200 once held 200 vectors for 275 chunks -> count_vectors gives 200 -> rebuild needed.
def count_vectors(persist_dir):
    """
    Count vectors, then RELEASE the store.

    On Windows, Chroma keeps its SQLite file open, and shutil.rmtree on a locked
    directory fails silently under ignore_errors=True. That left 200 stale
    vectors in place while a rebuild appended 275 more -- a 475-vector index for
    a 275-chunk document. The handle must be dropped and collected before the
    directory can be removed.
    """
    # gc = Python's garbage collector (frees objects no one uses). Forcing it releases the open database file on Windows.
    import gc
    store = None
    # try/except/finally: open the store and count; on any error return -1; ALWAYS (finally) close the store afterwards.
    try:
        store = Chroma(persist_directory=str(persist_dir),
                       embedding_function=get_embeddings())
        # _collection.count() = the number of vectors stored in the underlying ChromaDB collection (a named table of vectors).
        return store._collection.count()
    except Exception:
        return -1
    finally:
        # Only close if the store was actually opened.
        if store is not None:
            # Stop Chroma's internal system so it lets go of its SQLite file. If even that fails, ignore it.
            try:
                store._client._system.stop()
            except Exception:
                pass
            # Drop our reference and run the garbage collector, so Windows releases the file lock.
            del store
        gc.collect()


# IN: path of a store folder  ->  OUT: nothing; the folder is deleted, or an error is raised if it cannot be.
# WHY: rebuilding on top of old files would mix old and new vectors (once gave 475 vectors for 275 chunks).
# Example: remove_index_dir(eval/chroma_eval/bi_cs1000_co200) -> folder gone, or RuntimeError after 4 tries.
def remove_index_dir(persist_dir):
    """Delete a store directory, and confirm it is gone. Never assume."""
    import gc
    # Try up to 4 times. ignore_errors=True means rmtree does not raise, so we check ourselves if the folder still exists.
    for attempt in range(4):
        shutil.rmtree(persist_dir, ignore_errors=True)
        # Folder gone -> success.
        if not persist_dir.exists():
            return
        # Still there (file lock) -> free memory and wait longer each time: 1.5 s, 3 s, 4.5 s, 6 s.
        gc.collect()
        time.sleep(1.5 * (attempt + 1))
    # 4 failures -> stop loudly instead of corrupting the index.
    raise RuntimeError(
        "could not delete %s (file lock?). Close any process using it and retry; "
        "rebuilding on top of it would corrupt the index." % persist_dir)


# IN: nothing  ->  OUT: an embeddings object for gemini-embedding-001 (the API key is read from os.environ).
# WHY: one place that creates the embedder, so every store uses exactly the same model as the live app.
# Example: get_embeddings().embed_query("CREATE DATABASE") -> a list of 3072 numbers.
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)


# IN: PDF path, chunk size, overlap  ->  OUT: (list of pages, list of chunks), both LangChain Documents.
# WHY: same splitter settings as rag_engine.py, so the eval measures what the app really does.
# Example: chunk_document(mysql_pdf, 1000, 200) -> 72 pages -> 71 chunks (MySQL row of the chunk table in eval/README.md).
def chunk_document(path, chunk_size, chunk_overlap):
    # Load the PDF: one Document per page.
    pages = PyPDFLoader(path).load()
    # separators = where to cut, in order of preference: blank line (paragraph), newline, ". " (sentence end),
    # space (word), and "" (anywhere, the last resort).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],   # same as rag_engine.py
    )
    # split_documents splits each page on its own, so a chunk never spans two pages (verified: 0 of 792, eval/README.md).
    return pages, splitter.split_documents(pages)


# IN: doc id, its metadata (path/label), chunk size, overlap, flags  ->  OUT: nothing; writes the store + chunks.json.
# WHY: builds one index, but skips it when a complete one already exists (embedding costs API quota).
# Example: build_one("bi", meta, 1000, 200) -> prints "bi_cs1000_co200  CACHED  (275 chunks, 275 vectors)" if done.
# force=True rebuilds anyway; chunks_only=True only splits text (no embedding API calls).
def build_one(doc_id, meta, chunk_size, chunk_overlap, force=False, chunks_only=False):
    # Work out the names: folder for the vectors, and the JSON file for the chunk texts.
    key = config.index_key(doc_id, chunk_size, chunk_overlap)
    persist_dir = config.CHROMA_EVAL_DIR / key
    chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")

    # "Already built" must mean the store holds a vector for EVERY chunk, not
    # merely that a directory exists. A batch that exhausts its retries leaves a
    # partial store behind, and a directory-existence check passes on it -- which
    # silently produced a bi_cs1000_co200 index holding 200 of 275 vectors, and
    # recall numbers measured against a document missing 27% of its content.
    # Chunks file exists and no --force -> maybe already built. Read the chunk count n from the JSON file.
    if chunks_path.exists() and not force:
        n = len(json.loads(chunks_path.read_text(encoding="utf-8")))
        # --chunks-only: having the chunks file is enough. "%-28s" prints the key left-aligned in 28 characters (neat columns).
        if chunks_only:
            print("   %-28s CACHED  (%d chunks)" % (key, n))
            return
        # A non-empty store folder exists -> count its vectors and compare with the chunk count.
        if persist_dir.exists() and any(persist_dir.iterdir()):
            have = count_vectors(persist_dir)
            # Count matches -> complete -> skip.
            if have == n:
                print("   %-28s CACHED  (%d chunks, %d vectors)" % (key, n, have))
                return
            # Count is wrong (partial or doubled store) -> say so and delete the folder before rebuilding.
            # max(have, 0) prints 0 instead of -1 when the store could not be opened.
            print("   %-28s WRONG SIZE (%d vectors for %d chunks) -- rebuilding"
                  % (key, max(have, 0), n))
            remove_index_dir(persist_dir)

    # Split the PDF into chunks (no API call).
    pages, chunks = chunk_document(meta["path"], chunk_size, chunk_overlap)
    # end=" " keeps the cursor on the same line; flush=True prints it right away (before the slow embedding starts).
    print("   %-28s %d pages -> %d chunks ..." % (key, len(pages), len(chunks)),
          end=" ", flush=True)

    # Save every chunk's text and page index as JSON. ensure_ascii=False keeps non-English characters readable.
    # .get("page", 0) uses page 0 if a chunk has no page number.
    chunks_path.write_text(json.dumps(
        [{"text": c.page_content, "page": c.metadata.get("page", 0)} for c in chunks],
        ensure_ascii=False), encoding="utf-8")

    # Splitting costs no API calls -- only embedding does. --chunks-only stops
    # here, which is enough for BM25 (keyword) retrieval and lets the offline
    # baseline be completed while the embedding quota is exhausted.
    # --chunks-only stops here, before any embedding.
    if chunks_only:
        print("chunks only (no embedding)")
        return

    # Create the store folder (parents=True also creates eval/chroma_eval/ if missing; exist_ok=True = no error if present).
    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    # Open an empty (or existing) Chroma store in that folder.
    store = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    # Loop over the chunks 10 at a time: i = 0, 10, 20, ... (range with a step of EMBED_BATCH).
    for i in range(0, len(chunks), EMBED_BATCH):
        # Slice out this batch: chunks i to i+9.
        batch = chunks[i:i + EMBED_BATCH]
        # Up to 6 attempts per batch.
        for attempt in range(6):
            # Embed the batch and add it to the store. Success -> break out of the retry loop.
            try:
                store.add_documents(batch)
                break
            # Failure (usually the rate limit) -> EXPONENTIAL BACKOFF = wait longer after every failure:
            # 5 * 2^attempt = 5, 10, 20, 40, 80, 160 seconds. i // EMBED_BATCH = the batch number (// = whole-number division).
            except Exception as e:
                wait = 5 * (2 ** attempt)
                print("\n      batch %d failed (%s), retrying in %ds"
                      % (i // EMBED_BATCH, type(e).__name__, wait))
                time.sleep(wait)
        # for...else: the "else" runs only if the loop finished WITHOUT "break", i.e. all 6 attempts failed.
        else:
            # Never carry on quietly: a skipped batch means missing content, and
            # a store that looks built but is not corrupts every number measured
            # against it.
            raise RuntimeError(
                "batch %d of %s exhausted retries -- index would be incomplete"
                % (i // EMBED_BATCH, key))
        # Pause 3 seconds between batches to stay under the per-minute limit.
        time.sleep(BATCH_PAUSE_S)

    # Final check: the number of vectors must equal the number of chunks.
    final = store._collection.count()
    # Mismatch -> print INCOMPLETE (the next run's count check will catch and rebuild it).
    if final != len(chunks):
        print("INCOMPLETE %d/%d" % (final, len(chunks)))
    # Match -> done.
    else:
        print("done (%d vectors)" % final)


# IN: command-line flags  ->  OUT: nothing; builds every index for every document.
# WHY: the entry point. Run: py -3.12 eval/build_index.py [--chunks-only] [--force]
# Example: with the 6 CONFIGS there are 3 chunking schemes -> 3 indexes per document, 9 for the 3 documents.
def main():
    # Read the flags: True if the word appears on the command line.
    force = "--force" in sys.argv
    chunks_only = "--chunks-only" in sys.argv
    # Make sure eval/chroma_eval/ exists.
    config.CHROMA_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Collect the distinct (chunk_size, chunk_overlap) pairs from all configs. The set {...} removes repeats, so
    # baseline, k8, hybrid and hybrid_k8 (all 1000/200) count once. Result: [(500, 100), (1000, 200), (2000, 400)].
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"]) for c in config.CONFIGS.values()})
    print("Chunking schemes to build: %s" % ", ".join("%d/%d" % s for s in schemes))
    # Tell the user no API calls will be made in --chunks-only mode.
    if chunks_only:
        print("--chunks-only: splitting text only, no embedding API calls.")

    # Loop over every document in documents.json...
    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        # ...and every chunking scheme, building (or skipping) each index.
        for chunk_size, chunk_overlap in schemes:
            build_one(doc_id, meta, chunk_size, chunk_overlap, force=force,
                      chunks_only=chunks_only)

    print("\nAll indexes ready in %s" % config.CHROMA_EVAL_DIR)


# Run main() only when this file is started directly, not when it is imported by another script.
if __name__ == "__main__":
    main()
