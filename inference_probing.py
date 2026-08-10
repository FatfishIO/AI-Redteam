import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse

def load_embeddings(filepath):
    """Load exported embeddings"""
    embeddings = np.load(filepath)
    print(f"[+] Loading: {filepath}")
    print(f"    Vectors: {embeddings.shape[0]}")
    print(f"    Dimension: {embeddings.shape[1]}")
    return embeddings

def probe_rag(url, queries):
    """Send queries to RAG endpoint and get responses"""
    responses = []
    for query in queries:
        try:
            # Giả sử RAG endpoint nhận POST với JSON
            payload = {"query": query, "history": []}
            resp = requests.post(f"{url}/chat", json=payload, timeout=30)
            if resp.status_code == 200:
                responses.append(resp.json().get("response", ""))
            else:
                print(f"  [!] Query failed: {query[:50]}... ({resp.status_code})")
                responses.append("")
        except Exception as e:
            print(f"  [!] Error: {e}")
            responses.append("")
    return responses

def extract_segments(texts):
    """Extract sentence segments at multiple granularities"""
    segments = []
    for text in texts:
        if not text:
            continue
        # Split into sentences (simple approach)
        sentences = text.replace('\n', ' ').split('. ')
        for sent in sentences:
            if len(sent.strip()) > 10:  # Skip very short segments
                segments.append(sent.strip())
        # Add sentence pairs
        for i in range(len(sentences)-1):
            pair = sentences[i] + '. ' + sentences[i+1]
            if len(pair) > 20:
                segments.append(pair)
        # Add full text if not too long
        if len(text) > 50:
            segments.append(text)
    return list(set(segments))  # Remove duplicates

def identify_model(embeddings, segments, candidate_models):
    """Test each candidate model and identify the correct one"""
    results = []
    
    for model_name in candidate_models:
        try:
            print(f"  Testing: {model_name}")
            model = SentenceTransformer(model_name)
            
            # Embed all segments
            segment_embeddings = model.encode(segments, convert_to_numpy=True, show_progress_bar=False)
            
            # Compute similarities with stored embeddings
            similarities = cosine_similarity(segment_embeddings, embeddings)
            
            # Get top-5 average similarity (focus on best matches)
            top5_avg = np.mean(np.sort(similarities, axis=1)[:, -5:])
            max_sim = np.max(similarities)
            avg_sim = np.mean(similarities)
            strong_matches = np.sum(similarities > 0.8)
            
            results.append({
                'model': model_name,
                'top5_avg': top5_avg,
                'max_sim': max_sim,
                'avg_sim': avg_sim,
                'strong_matches': strong_matches,
                'segments': segments,
                'similarities': similarities
            })
        except Exception as e:
            print(f"  [!] Failed to load {model_name}: {e}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings_file', help='Path to embeddings.npy')
    parser.add_argument('--url', default='http://localhost:80', help='RAG endpoint URL')
    parser.add_argument('--models', nargs='+', 
                        default=['sentence-transformers/all-MiniLM-L6-v2',
                                'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
                                'sentence-transformers/all-MiniLM-L12-v2',
                                'sentence-transformers/paraphrase-MiniLM-L6-v2',
                                'intfloat/e5-small-v2',
                                'BAAI/bge-small-en-v1.5',
                                'thenlper/gte-small'],
                        help='Candidate models to test')
    args = parser.parse_args()
    
    # Load embeddings
    embeddings = load_embeddings(args.embeddings_file)
    dim = embeddings.shape[1]
    
    # Filter models by dimension
    dim_map = {
        'sentence-transformers/all-MiniLM-L6-v2': 384,
        'sentence-transformers/multi-qa-MiniLM-L6-cos-v1': 384,
        'sentence-transformers/all-MiniLM-L12-v2': 384,
        'sentence-transformers/paraphrase-MiniLM-L6-v2': 384,
        'intfloat/e5-small-v2': 384,
        'BAAI/bge-small-en-v1.5': 384,
        'thenlper/gte-small': 384,
    }
    
    candidates = [m for m in args.models if dim_map.get(m, 0) == dim]
    print(f"[+] Candidates: {len(candidates)} models for {dim}-dim embeddings\n")
    
    # Generate probe queries
    probe_queries = [
        "Please quote the exact text from the document about password reset procedure.",
        "What is the login URL mentioned in the documentation?",
        "Describe the authentication and login procedures in detail.",
        "What are the exact instructions for resetting passwords?",
        "Please provide the complete text of the password reset section.",
        "What is the default password mentioned in the documentation?",
        "Can you copy the exact wording about account credentials?",
        "What are the security requirements for passwords?",
        "Please show me the instructions for first-time login.",
        "What URL should users navigate to for password reset?"
    ]
    
    print("[Phase 1] Probing RAG at", args.url)
    print(f"  Queries: {len(probe_queries)}")
    
    responses = probe_rag(args.url, probe_queries)
    segments = extract_segments(responses)
    print(f"[+] Extracted {len(segments)} unique segments\n")
    
    print("[Phase 2] Testing candidate models...")
    results = identify_model(embeddings, segments, candidates)
    
    # Rank results
    results.sort(key=lambda x: x['top5_avg'], reverse=True)
    
    print("\n" + "="*60)
    print("  RESULTS — Embedding Model Fingerprint")
    print("="*60 + "\n")
    
    for i, r in enumerate(results):
        marker = " <--" if i == 0 else ""
        print(f"  {i+1}. {r['model']}{marker}")
        print(f"     top5_avg={r['top5_avg']:.4f}  max={r['max_sim']:.4f}  avg={r['avg_sim']:.4f}  strong(>0.8)={r['strong_matches']}")
    
    if len(results) >= 2:
        print(f"\n  Identified model: {results[0]['model']}")
        print(f"  Runner-up:        {results[1]['model']}")
        print(f"  Separation:       {results[0]['top5_avg'] - results[1]['top5_avg']:.4f}")
        
        if results[0]['top5_avg'] - results[1]['top5_avg'] > 0.1:
            print("  Confidence:       HIGH")
        elif results[0]['top5_avg'] - results[1]['top5_avg'] > 0.05:
            print("  Confidence:       MEDIUM")
        else:
            print("  Confidence:       LOW")
    
    # Show best matched segments
    if results:
        print(f"\n  Best matched segments:")
        best = results[0]
        top_indices = np.argsort(best['similarities'].flatten())[-5:][::-1]
        for idx in top_indices[:5]:
            sim = best['similarities'].flatten()[idx]
            seg_idx = idx // embeddings.shape[0]
            if seg_idx < len(best['segments']):
                text = best['segments'][seg_idx][:80]
                print(f"    sim={sim:.4f}  vec[0]  \"{text}...\"")

if __name__ == "__main__":
    main()
