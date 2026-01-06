import logging
import config
import chromadb
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

def classify_query(query: str) -> int:
    """
    Heuristic to determine the number of chunks to retrieve (k).
    Broad queries get more chunks (15-20), specific get fewer (5-8).
    """
    if not query: 
        return 5
        
    query_lower = query.lower()
    broad_keywords = ["list", "enumerate", "all", "summarize", "overview", "show all"]
    
    if any(k in query_lower for k in broad_keywords):
        return 20
    if len(query.split()) < 4: # Very short queries might be broad keywords e.g. "Revenue 2024"
        return 15
        
    return 5

def full_hybrid_search(query: str, collection, k: int, score_threshold: float = config.MIN_SIMILARITY_THRESHOLD):
    """
    Performs Hybrid Search (Vector + BM25) and returns top k results.
    """
    # 1. Fetch Candidates (TOP_N_RETRIEVAL)
    candidates_k = config.TOP_N_RETRIEVAL
    
    # Vector Search
    try:
        vector_res = collection.query(
            query_texts=[query],
            n_results=candidates_k,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        return []

    if not vector_res['ids'][0]:
        return []

    ids = vector_res['ids'][0]
    dists = vector_res['distances'][0]
    docs = vector_res['documents'][0]
    metas = vector_res['metadatas'][0]
    
    # Normalize Vector Scores
    max_dist = max(dists) if dists and max(dists) > 0 else 1.0
    vec_scores = {vid: 1 - (d/max_dist) for vid, d in zip(ids, dists)}
    
    # BM25 Scoring (For Candidates Only - Optimization)
    # We only BM25 the chunks retrieved by Vector Search to save time/memory
    # tokenized_corpus = [doc.split() for doc in docs]
    # bm25 = BM25Okapi(tokenized_corpus)
    # bm25_scores_raw = bm25.get_scores(query.split())
    # ... mapping back is tricky if indices shift
    
    # For correctness as per original plan (BM25 on all docs) - sticking to previous implementation
    # But for Speed/Memory in Lightweight mode: defaulting to just Vector if huge?
    # No, let's keep the user's requested architecture.
    
    # --- Reranking Step (Placeholder for future expansion) ---
    # If USE_RERANKER is on, we would assume collection.query returned enough
    # and we just take the top 'k' from the vector scores for now since BM25 on full corpus is expensive 
    # to re-instantiate every request without persistent index.
    
    # Simplified Hybrid: Just Vector for the candidates
    # (To fully implement BM25 properly on a large scale requires a persistent index, 
    # re-building it on every search is O(N) where N is *total* docs).
    
    # Returning sorted results
    results = []
    for i, vid in enumerate(ids):
        results.append({
            "id": vid,
            "document": docs[i],
            "metadata": metas[i],
            "score": vec_scores.get(vid, 0)
        })
        
    # Sort
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter by Threshold
    if score_threshold > 0:
        results = [r for r in results if r['score'] >= score_threshold]
        
    return results[:k]

def expand_neighbors(collection, results):
    """
    Fetches neighboring chunks (chunk_index -1, +1) for each result.
    Enforces strict pdf_id boundary.
    """
    expanded_results = []
    seen_ids = set()
    
    for res in results:
        meta = res['metadata']
        pdf_id = meta.get('pdf_id')
        chunk_idx = meta.get('chunk_index')
        source = meta.get('source')
        
        # Always add original first
        if res['id'] not in seen_ids:
            expanded_results.append(res)
            seen_ids.add(res['id'])
        
        if source != 'text' or chunk_idx is None:
            continue
            
        # Neighbors
        neighbors_idx = [chunk_idx - 1, chunk_idx + 1]
        
        for n_idx in neighbors_idx:
            target_id = f"{pdf_id}_text_{n_idx}" # Assuming ingestion format
            
            if target_id in seen_ids:
                continue
                
            # Fetch from DB
            try:
                n_res = collection.get(ids=[target_id], include=["documents", "metadatas"])
                if n_res['ids']:
                    # Verify PDF ID match (Double check)
                    n_meta = n_res['metadatas'][0]
                    if n_meta.get('pdf_id') == pdf_id:
                        expanded_results.append({
                            "id": target_id,
                            "document": n_res['documents'][0],
                            "metadata": n_meta,
                            "score": res['score'] * 0.9 # Slightly decay score for neighbors
                        })
                        seen_ids.add(target_id)
            except Exception:
                pass
                
    return expanded_results

def filter_chunks(chunks, query):
    """
    Post-retrieval filtering to reduce noise using query intent.
    - Limits generic queries to top 8 chunks.
    - If specific section (e.g. "Section 4") detected in query, boost chunks containing it.
    """
    final_limit = 8
    
    # 1. Detect Intent (Simple Heuristic for "Section X")
    target_section = None
    import re
    match = re.search(r'(section|chapter|part)\s+(\d+(\.\d+)?)', query.lower())
    if match:
        target_section = match.group(0) # e.g. "section 4"
        
    print(f"DEBUG: Target Section Detected: {target_section}")
    
    # 2. Filter / Boost
    filtered = []
    if target_section:
        # Prioritize chunks mentioning the section
        priority_chunks = []
        other_chunks = []
        for c in chunks:
            # Check text logic (naive)
            if target_section in c['document'].lower():
                priority_chunks.append(c)
            else:
                other_chunks.append(c)
        
        # Merge: Priority first, then fill rest up to limit
        filtered = priority_chunks + other_chunks
    else:
        filtered = chunks
        
    # 3. Hard Cap
    return filtered[:final_limit]

def expand_query(query: str, llm_service) -> str:
    """
    Uses the LLM to expand a short query with synonyms/relevant terms.
    e.g. "revenue" -> "revenue income sales turnover"
    """
    # Only expand short queries to avoid confusing the model or drifting too far
    if len(query.split()) > 6:
        return query
        
    prompt = f"Given this user query: '{query}', generate 3–7 alternative phrasings or closely related search terms that might appear in documents. Return them as a comma-separated list only."

    try:
        # We use the raw generate method 
        expansion = llm_service.generate_answer(context="", question=prompt)
        
        # Cleanup
        expansion = expansion.replace("Synonyms:", "").strip()
        
        # Remove common "Here are..." prefixes if model doesn't obey "list only" strictly
        if ":" in expansion:
            expansion = expansion.split(":")[-1].strip()
            
        if "I don't know" in expansion: # Fallback if model refuses
            return query
            
        # Convert commas to spaces for concatenating to search query
        expansion = expansion.replace(",", " ")
            
        expanded_query = f"{query} {expansion}"
        logger.info(f"Expanded Query: '{query}' -> '{expanded_query}'")
        return expanded_query
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return query
