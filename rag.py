# Handles loading research documents, building the FAISS vector index, and retrieving relevant context

import os
import pickle
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import RESEARCH_FOLDER, INDEX_PATH, DOCS_PATH, SIMILARITY_THRESHOLD

logger          = logging.getLogger(__name__)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faiss_index     = None
documents       = []


def init_rag():
    # Called once on startup — loads cached index or builds a new one from research folder
    global faiss_index, documents

    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        faiss_index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            documents = pickle.load(f)
        logger.info(f"FAISS index loaded ({faiss_index.ntotal} docs).")
        return

    if not os.path.isdir(RESEARCH_FOLDER):
        logger.warning(f"Folder '{RESEARCH_FOLDER}' not found. Running without RAG.")
        return

    for fname in os.listdir(RESEARCH_FOLDER):
        if fname.endswith(".txt"):
            with open(os.path.join(RESEARCH_FOLDER, fname), "r", encoding="utf-8") as f:
                documents.append(f.read())

    if not documents:
        return

    logger.info(f"Building FAISS index for {len(documents)} documents...")
    emb = embedding_model.encode(documents, convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(emb)
    faiss_index = faiss.IndexFlatIP(emb.shape[1])
    faiss_index.add(emb)

    faiss.write_index(faiss_index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(documents, f)
    logger.info("FAISS index saved.")


def retrieve_context(query: str) -> str:
    # Returns the most relevant research snippets for a given query, or empty string if none found
    if faiss_index is None or faiss_index.ntotal == 0:
        return ""

    q_emb = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = faiss_index.search(q_emb, 3)

    return "\n\n".join(
        documents[idx]
        for score, idx in zip(scores[0], indices[0])
        if score >= SIMILARITY_THRESHOLD
    )
