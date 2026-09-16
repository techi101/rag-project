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
import json
import shutil
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import config

config.load_env()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Batch size for embedding requests. 50 exceeds the free tier's per-minute limit
# outright; 20 still trips it partway through a long document. 10 with a longer
# pause sustains ~200 chunks without RESOURCE_EXHAUSTED.
EMBED_BATCH = 10
BATCH_PAUSE_S = 3.0


def count_vectors(persist_dir):
    """
    Count vectors, then RELEASE the store.

    On Windows, Chroma keeps its SQLite file open, and shutil.rmtree on a locked
    directory fails silently under ignore_errors=True. That left 200 stale
    vectors in place while a rebuild appended 275 more -- a 475-vector index for
    a 275-chunk document. The handle must be dropped and collected before the
    directory can be removed.
    """
    import gc
    store = None
    try:
        store = Chroma(persist_directory=str(persist_dir),
                       embedding_function=get_embeddings())
        return store._collection.count()
    except Exception:
        return -1
    finally:
        if store is not None:
            try:
                store._client._system.stop()
            except Exception:
                pass
            del store
        gc.collect()


def remove_index_dir(persist_dir):
    """Delete a store directory, and confirm it is gone. Never assume."""
    import gc
    for attempt in range(4):
        shutil.rmtree(persist_dir, ignore_errors=True)
        if not persist_dir.exists():
            return
        gc.collect()
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(
        "could not delete %s (file lock?). Close any process using it and retry; "
        "rebuilding on top of it would corrupt the index." % persist_dir)


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=config.EMBED_MODEL)


def chunk_document(path, chunk_size, chunk_overlap):
    pages = PyPDFLoader(path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],   # same as rag_engine.py
    )
    return pages, splitter.split_documents(pages)


def build_one(doc_id, meta, chunk_size, chunk_overlap, force=False, chunks_only=False):
    key = config.index_key(doc_id, chunk_size, chunk_overlap)
    persist_dir = config.CHROMA_EVAL_DIR / key
    chunks_path = config.CHROMA_EVAL_DIR / (key + "_chunks.json")

    # "Already built" must mean the store holds a vector for EVERY chunk, not
    # merely that a directory exists. A batch that exhausts its retries leaves a
    # partial store behind, and a directory-existence check passes on it -- which
    # silently produced a bi_cs1000_co200 index holding 200 of 275 vectors, and
    # recall numbers measured against a document missing 27% of its content.
    if chunks_path.exists() and not force:
        n = len(json.loads(chunks_path.read_text(encoding="utf-8")))
        if chunks_only:
            print("   %-28s CACHED  (%d chunks)" % (key, n))
            return
        if persist_dir.exists() and any(persist_dir.iterdir()):
            have = count_vectors(persist_dir)
            if have == n:
                print("   %-28s CACHED  (%d chunks, %d vectors)" % (key, n, have))
                return
            print("   %-28s WRONG SIZE (%d vectors for %d chunks) -- rebuilding"
                  % (key, max(have, 0), n))
            remove_index_dir(persist_dir)

    pages, chunks = chunk_document(meta["path"], chunk_size, chunk_overlap)
    print("   %-28s %d pages -> %d chunks ..." % (key, len(pages), len(chunks)),
          end=" ", flush=True)

    chunks_path.write_text(json.dumps(
        [{"text": c.page_content, "page": c.metadata.get("page", 0)} for c in chunks],
        ensure_ascii=False), encoding="utf-8")

    # Splitting costs no API calls -- only embedding does. --chunks-only stops
    # here, which is enough for BM25 (keyword) retrieval and lets the offline
    # baseline be completed while the embedding quota is exhausted.
    if chunks_only:
        print("chunks only (no embedding)")
        return

    persist_dir.mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()

    store = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        for attempt in range(6):
            try:
                store.add_documents(batch)
                break
            except Exception as e:
                wait = 5 * (2 ** attempt)
                print("\n      batch %d failed (%s), retrying in %ds"
                      % (i // EMBED_BATCH, type(e).__name__, wait))
                time.sleep(wait)
        else:
            # Never carry on quietly: a skipped batch means missing content, and
            # a store that looks built but is not corrupts every number measured
            # against it.
            raise RuntimeError(
                "batch %d of %s exhausted retries -- index would be incomplete"
                % (i // EMBED_BATCH, key))
        time.sleep(BATCH_PAUSE_S)

    final = store._collection.count()
    if final != len(chunks):
        print("INCOMPLETE %d/%d" % (final, len(chunks)))
    else:
        print("done (%d vectors)" % final)


def main():
    force = "--force" in sys.argv
    chunks_only = "--chunks-only" in sys.argv
    config.CHROMA_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    schemes = sorted({(c["chunk_size"], c["chunk_overlap"]) for c in config.CONFIGS.values()})
    print("Chunking schemes to build: %s" % ", ".join("%d/%d" % s for s in schemes))
    if chunks_only:
        print("--chunks-only: splitting text only, no embedding API calls.")

    for doc_id, meta in config.DOCUMENTS.items():
        print("\n[%s] %s" % (doc_id, meta["label"]))
        for chunk_size, chunk_overlap in schemes:
            build_one(doc_id, meta, chunk_size, chunk_overlap, force=force,
                      chunks_only=chunks_only)

    print("\nAll indexes ready in %s" % config.CHROMA_EVAL_DIR)


if __name__ == "__main__":
    main()
