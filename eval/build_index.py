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
# WHAT THIS FILE IS: cuts each PDF into chunks and saves their embeddings, once per (document, size, overlap).
# Chunk = small piece of a page; overlap = chars repeated between neighbour chunks.
# Example: "bi" at 1000/200 -> folder bi_cs1000_co200 (275 chunks) + bi_cs1000_co200_chunks.json (for BM25).
# Flow: each doc -> each scheme -> complete? skip : PDF -> pages -> chunks -> chunks.json -> embed x10 -> Chroma -> count
import json  # chunks.json read/write
import shutil  # rmtree deletes a broken index folder
import sys  # --force / --chunks-only flags + import path
import time  # sleep between API calls (rate limit)
import pathlib  # file paths

sys.path.insert(0, str(pathlib.Path(__file__).parent))  # so "import config" finds eval/config.py
import config  # eval/config.py: folders, models, CONFIGS, DOCUMENTS

config.load_env()  # API keys into os.environ BEFORE Google lib is used

from langchain_community.document_loaders import PyPDFLoader  # PDF -> one Document per page
from langchain_text_splitters import RecursiveCharacterTextSplitter  # cuts at paragraph, line, sentence, word
from langchain_chroma import Chroma  # local vector DB saved to a folder
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # gemini-embedding-001: text -> vector

# Batch size for embedding requests. 50 exceeds the free tier's per-minute limit
# outright; 20 still trips it partway through a long document. 10 with a longer
# pause sustains ~200 chunks without RESOURCE_EXHAUSTED.
# Rate limit = max requests per minute; over it -> RESOURCE_EXHAUSTED.
EMBED_BATCH = 10  # chunks per Google request
BATCH_PAUSE_S = 3.0  # wait 3 s after each batch


# IN: store folder -> OUT: vector count, or -1 if it can't open
# e.g. bi_cs1000_co200 once held 200 vectors for 275 chunks -> 200 -> rebuild
def count_vectors(persist_dir):
    """
    Count vectors, then RELEASE the store.

    On Windows, Chroma keeps its SQLite file open, and shutil.rmtree on a locked
    directory fails silently under ignore_errors=True. That left 200 stale
    vectors in place while a rebuild appended 275 more -- a 475-vector index for
    a 275-chunk document. The handle must be dropped and collected before the
    directory can be removed.
    """
    import gc  # garbage collector, frees the file lock
    store = None  # not opened yet
    try:  # error -> -1; finally always closes
        store = Chroma(persist_directory=str(persist_dir),  # open saved store
                       embedding_function=get_embeddings())  # same embedder
        return store._collection.count()  # vectors in the ChromaDB collection
    except Exception:  # can't open
        return -1
    finally:  # runs no matter what
        if store is not None:  # only if opened
            try:
                store._client._system.stop()  # let go of the SQLite file
            except Exception:  # ignore if even that fails
                pass
            del store  # drop our reference
        gc.collect()  # Windows releases the lock


# IN: store folder -> OUT: nothing, folder deleted or RuntimeError after 4 tries
# e.g. rebuilding on old files once gave 475 vectors for 275 chunks
def remove_index_dir(persist_dir):
    """Delete a store directory, and confirm it is gone. Never assume."""
    import gc  # garbage collector
    for attempt in range(4):  # up to 4 tries
        shutil.rmtree(persist_dir, ignore_errors=True)  # never raises, so we check below
        if not persist_dir.exists():  # gone -> success
            return
        gc.collect()  # still locked, free memory
        time.sleep(1.5 * (attempt + 1))  # wait 1.5, 3, 4.5, 6 s
    raise RuntimeError(  # 4 failures -> stop loudly
        "could not delete %s (file lock?). Close any process using it and retry; "
        "rebuilding on top of it would corrupt the index." % persist_dir)


# IN: nothing -> OUT: gemini-embedding-001 embedder (key from os.environ)
# e.g. get_embeddings().embed_query("CREATE DATABASE") -> 3072 numbers
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)  # same model as the live app


# IN: PDF path, chunk size, overlap -> OUT: (pages, chunks)
# e.g. chunk_document(mysql_pdf, 1000, 200) -> 72 pages -> 71 chunks (same splitter as documind/ingest.py)
def chunk_document(path, chunk_size, chunk_overlap):
    pages = PyPDFLoader(path).load()  # one Document per page
    splitter = RecursiveCharacterTextSplitter(  # the chunk cutter
        chunk_size=chunk_size,  # max chars per chunk
        chunk_overlap=chunk_overlap,  # chars shared with neighbour
        separators=config.CHUNK_SEPARATORS,   # imported from documind/settings.py
    )
    return pages, splitter.split_documents(pages)  # per page, so no chunk spans two pages (0 of 792)


# IN: doc id, meta, size, overlap, flags -> OUT: nothing, writes store + chunks.json
# e.g. build_one("bi", meta, 1000, 200) -> "bi_cs1000_co200  CACHED  (275 chunks, 275 vectors)" if done
def build_one(doc_id, meta, chunk_size, chunk_overlap, force=False, chunks_only=False):
    # ── Step 1: names for this index ──
    key = config.index_key(doc_id, chunk_size, chunk_overlap)  # e.g. "bi_cs1000_co200"
    persist_dir = config.CHROMA_EVAL_DIR / key  # vectors folder
    chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")  # chunk texts file

    # ── Step 2: skip if already complete ──
    # "Already built" must mean the store holds a vector for EVERY chunk, not
    # merely that a directory exists. A batch that exhausts its retries leaves a
    # partial store behind, and a directory-existence check passes on it -- which
    # silently produced a bi_cs1000_co200 index holding 200 of 275 vectors, and
    # recall numbers measured against a document missing 27% of its content.
    if chunks_path.exists() and not force:  # maybe built, no --force
        n = len(json.loads(chunks_path.read_text(encoding="utf-8")))  # chunk count
        if chunks_only:  # chunks file is enough
            print("   %-28s CACHED  (%d chunks)" % (key, n))  # %-28s = left-aligned, 28 wide
            return
        if persist_dir.exists() and any(persist_dir.iterdir()):  # non-empty store folder
            have = count_vectors(persist_dir)  # vectors stored
            if have == n:  # complete -> skip
                print("   %-28s CACHED  (%d chunks, %d vectors)" % (key, n, have))
                return
            print("   %-28s WRONG SIZE (%d vectors for %d chunks) -- rebuilding"  # partial or doubled
                  % (key, max(have, 0), n))  # -1 shown as 0
            remove_index_dir(persist_dir)  # delete before rebuild

    # ── Step 3: split the PDF (no API call) ──
    pages, chunks = chunk_document(meta["path"], chunk_size, chunk_overlap)  # pages + chunks
    print("   %-28s %d pages -> %d chunks ..." % (key, len(pages), len(chunks)),
          end=" ", flush=True)  # same line, show now before slow embedding

    # ── Step 4: save chunk texts for BM25 ──
    chunks_path.write_text(json.dumps(
        [{"text": c.page_content, "page": c.metadata.get("page", 0)} for c in chunks],  # page 0 if missing
        ensure_ascii=False), encoding="utf-8")  # keep non-English chars readable

    # Splitting costs no API calls -- only embedding does. --chunks-only stops
    # here, which is enough for BM25 (keyword) retrieval and lets the offline
    # baseline be completed while the embedding quota is exhausted.
    if chunks_only:  # no embedding wanted
        print("chunks only (no embedding)")
        return

    # ── Step 5: embed in batches of 10 with retries ──
    persist_dir.mkdir(parents=True, exist_ok=True)  # make folder (+ parents), ok if exists
    embeddings = get_embeddings()  # the embedder

    store = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)  # empty store
    for i in range(0, len(chunks), EMBED_BATCH):  # i = 0, 10, 20, ...
        batch = chunks[i:i + EMBED_BATCH]  # chunks i..i+9
        for attempt in range(6):  # up to 6 tries per batch
            try:
                store.add_documents(batch)  # embed + save
                break  # success, leave retry loop
            except Exception as e:  # usually the rate limit
                wait = 5 * (2 ** attempt)  # backoff: 5, 10, 20, 40, 80, 160 s
                print("\n      batch %d failed (%s), retrying in %ds"
                      % (i // EMBED_BATCH, type(e).__name__, wait))  # batch number, error name
                time.sleep(wait)  # wait, then retry
        else:  # for...else: runs only if no "break" (all 6 failed)
            # Never carry on quietly: a skipped batch means missing content, and
            # a store that looks built but is not corrupts every number measured
            # against it.
            raise RuntimeError(
                "batch %d of %s exhausted retries -- index would be incomplete"
                % (i // EMBED_BATCH, key))
        time.sleep(BATCH_PAUSE_S)  # 3 s pause, stay under per-minute limit

    # ── Step 6: vectors must equal chunks ──
    final = store._collection.count()  # vectors stored
    if final != len(chunks):  # mismatch; next run's check rebuilds it
        print("INCOMPLETE %d/%d" % (final, len(chunks)))
    else:  # match -> done
        print("done (%d vectors)" % final)


# IN: command-line flags -> OUT: nothing, builds every index
# e.g. py -3.12 eval/build_index.py [--chunks-only] [--force]; 3 schemes x 3 docs = 9 indexes
def main():
    force = "--force" in sys.argv  # rebuild even if complete
    chunks_only = "--chunks-only" in sys.argv  # split only, no embedding
    config.CHROMA_EVAL_DIR.mkdir(parents=True, exist_ok=True)  # make eval/chroma_eval/

    # Distinct (size, overlap) pairs -> [(500, 100), (1000, 200), (2000, 400)].
    schemes = sorted({(c["chunk_size"], c["chunk_overlap"]) for c in config.CONFIGS.values()})
    print("Chunking schemes to build: %s" % ", ".join("%d/%d" % s for s in schemes))  # e.g. "500/100, ..."
    if chunks_only:  # tell user: no API calls
        print("--chunks-only: splitting text only, no embedding API calls.")

    for doc_id, meta in config.DOCUMENTS.items():  # each document
        print("\n[%s] %s" % (doc_id, meta["label"]))  # e.g. "[bi] ..."
        for chunk_size, chunk_overlap in schemes:  # each chunking scheme
            build_one(doc_id, meta, chunk_size, chunk_overlap, force=force,
                      chunks_only=chunks_only)  # build or skip

    print("\nAll indexes ready in %s" % config.CHROMA_EVAL_DIR)  # final message


if __name__ == "__main__":  # only when run directly, not imported
    main()
