import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import argparse
import time

class ChunkTriage:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.positive_probes = []
        self.negative_probes = []
        self._load_probes()
    
    def _load_probes(self):
        """Load sensitivity probes"""
        self.positive_probes = [
            "Service account password is",
            "API key: sk-",
            "Database connection string: postgresql://",
            "JDBC URL: jdbc:postgresql://",
            "DATABASE_URL=postgresql://",
            "Bearer token: eyJ",
            "secret key:",
            "password reset procedure",
            "default password is",
            "temporary credential:",
            "AWS_ACCESS_KEY_ID=AKIA",
            "GITHUB_TOKEN=ghp_",
            "DOCKER_REGISTRY credentials",
            "SSH private key:",
            "client_secret:",
        ]
        
        self.negative_probes = [
            "Our company values are",
            "The quarterly report shows",
            "Welcome to the team",
            "Meeting minutes Q3",
            "Employee satisfaction survey",
            "Company policy handbook",
            "Performance review process",
            "Lunch menu next week",
            "Office holiday schedule",
            "Benefits enrollment form",
        ]
    
    def load_embeddings(self, filepath):
        """Load exported embeddings"""
        self.embeddings = np.load(filepath)
        print(f"[+] Loading: {filepath}")
        print(f"    Chunks: {self.embeddings.shape[0]}  |  Dim: {self.embeddings.shape[1]}")
        return self.embeddings
    
    def stage1_density(self, top_k=50):
        """Stage 1: Density-based filtering"""
        print("\n" + "-"*60)
        print("  STAGE 1: DENSITY  (all chunks -> top {})".format(top_k))
        print("-"*60 + "\n")
        
        start_time = time.time()
        
        # Compute k-nearest neighbors density
        nbrs = NearestNeighbors(n_neighbors=5, metric='cosine')
        nbrs.fit(self.embeddings)
        distances, indices = nbrs.kneighbors(self.embeddings)
        
        # Isolation score (1 - average cosine similarity to neighbors)
        isolation = 1 - np.mean(distances, axis=1)
        
        # Relevance to credential probes
        probe_embeddings = self.model.encode(
            self.positive_probes[:10], 
            convert_to_numpy=True
        )
        relevance = cosine_similarity(self.embeddings, probe_embeddings).max(axis=1)
        
        # Combined score
        scores = 0.5 * isolation + 0.5 * relevance
        scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize
        
        # Sort by score
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'chunk_idx': idx,
                'score': scores[idx],
                'isolation': isolation[idx],
                'relevance': relevance[idx]
            })
        
        # Print top 5
        print("    Top 5 density:")
        for i, r in enumerate(results[:5]):
            print(f"      #{i+1} chunk[{r['chunk_idx']:3d}]  score={r['score']:.4f}  "
                  f"(iso={r['isolation']:.3f} rel={r['relevance']:.3f})")
        
        print(f"    [{time.time()-start_time:.1f}s]  Passing {len(results)} candidates ->\n")
        return [r['chunk_idx'] for r in results]
    
    def stage2_pairwise(self, candidate_indices, top_k=20):
        """Stage 2: Contrastive pairwise scoring"""
        print("-"*60)
        print("  STAGE 2: PW  ({} chunks -> top {})".format(len(candidate_indices), top_k))
        print("-"*60 + "\n")
        
        start_time = time.time()
        
        # Encode positive and negative probes
        pos_embeddings = self.model.encode(self.positive_probes, convert_to_numpy=True)
        neg_embeddings = self.model.encode(self.negative_probes, convert_to_numpy=True)
        
        results = []
        for idx in candidate_indices:
            embedding = self.embeddings[idx:idx+1]
            
            # Compute similarities
            pos_sim = cosine_similarity(embedding, pos_embeddings).flatten()
            neg_sim = cosine_similarity(embedding, neg_embeddings).flatten()
            
            # Net score: max positive - mean negative
            net_score = np.max(pos_sim) - np.mean(neg_sim)
            
            # Normalized score
            z_score = (net_score - np.mean([np.max(cosine_similarity(self.embeddings[i:i+1], pos_embeddings)) 
                                           for i in candidate_indices])) / 0.5  # Approximate std
            
            results.append({
                'chunk_idx': idx,
                'score': net_score,
                'z_score': z_score,
                'max_pos': np.max(pos_sim),
                'mean_neg': np.mean(neg_sim)
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        top_indices = [r['chunk_idx'] for r in results[:top_k]]
        
        # Print top 5
        print("    Top 5 pw:")
        for i, r in enumerate(results[:5]):
            print(f"      #{i+1} chunk[{r['chunk_idx']:3d}]  final={r['score']:.4f}  "
                  f"(max_pos={r['max_pos']:.4f} mean_neg={r['mean_neg']:.4f})")
        
        print(f"    [{time.time()-start_time:.1f}s]  Passing {len(top_indices)} candidates ->\n")
        return top_indices
    
    def stage3_reconstruction(self, candidate_indices, top_k=10):
        """Stage 3: Shallow reconstruction scoring"""
        print("-"*60)
        print("  STAGE 3: RECON  ({} chunks, mode=seed)".format(len(candidate_indices)))
        print("-"*60 + "\n")
        
        start_time = time.time()
        
        # Template bank for reconstruction (simplified)
        templates = [
            "Please navigate to {URL} and click on Need help signing in. The default password is {PASSWORD}",
            "Your new password is {PASSWORD}. Please access {URL} and reset your credentials.",
            "Service account password: {PASSWORD} for {URL}",
            "Database connection string is postgresql://{USERNAME}:{PASSWORD}@{HOST}/db",
            "API key: {API_KEY} for endpoint {URL}",
            "The reset password is {PASSWORD} as soon as possible. check out {URL}",
            "Login credentials for {SYSTEM}: {USERNAME} / {PASSWORD}",
            "SSH key: {SSH_KEY} for server {HOST}",
            "AWS access key: {AWS_KEY} secret: {AWS_SECRET}",
            "GITHUB_TOKEN={GITHUB_TOKEN} with access to {REPO}"
        ]
        
        # Encode templates
        template_embeddings = self.model.encode(templates, convert_to_numpy=True)
        
        results = []
        for idx in candidate_indices:
            embedding = self.embeddings[idx:idx+1]
            
            # Find best matching template
            similarities = cosine_similarity(embedding, template_embeddings).flatten()
            best_match = np.argmax(similarities)
            template_sim = similarities[best_match]
            
            # Credential indicators
            # Simulate credential detection with heuristics
            # In practice, this would use regex and entropy analysis
            cred_score = min(1.0, template_sim * 1.2)  # Simplified
            
            combined_score = 0.5 * cred_score + 0.5 * template_sim
            
            results.append({
                'chunk_idx': idx,
                'template_sim': template_sim,
                'cred_score': cred_score,
                'combined_score': combined_score,
                'best_template': templates[best_match]
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Print top 5
        print("    Top 5 recon:")
        for i, r in enumerate(results[:5]):
            print(f"      #{i+1} chunk[{r['chunk_idx']:3d}]  combined={r['combined_score']:.4f}  "
                  f"(cred={r['cred_score']:.3f} inv={r['template_sim']:.4f})")
        
        print(f"    [{time.time()-start_time:.1f}s]\n")
        return results
    
    def fuse_results(self, density_results, pw_results, recon_results):
        """Weighted Reciprocal Rank Fusion"""
        print("-"*60)
        print("  FUSION: Weighted RRF  (density:1.0, pw:1.5, recon:2.0)")
        print("-"*60 + "\n")
        
        # Create rank maps
        density_rank = {idx: i+1 for i, idx in enumerate(density_results)}
        pw_rank = {idx: i+1 for i, idx in enumerate(pw_results)}
        recon_rank = {r['chunk_idx']: i+1 for i, r in enumerate(recon_results)}
        
        all_chunks = set(density_rank.keys()) | set(pw_rank.keys()) | set(recon_rank.keys())
        
        # Compute weighted RRF
        scores = {}
        for idx in all_chunks:
            rrf = 0.0
            if idx in density_rank:
                rrf += 1.0 / (10 + density_rank[idx])
            else:
                rrf += 1.0 / (10 + len(density_results) + 5)  # Penalty
            
            if idx in pw_rank:
                rrf += 1.5 / (10 + pw_rank[idx])
            else:
                rrf += 1.5 / (10 + len(pw_results) + 5)
            
            if idx in recon_rank:
                rrf += 2.0 / (10 + recon_rank[idx])
            else:
                rrf += 2.0 / (10 + len(recon_results) + 5)
            
            scores[idx] = rrf
        
        # Sort by score
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        print("="*72)
        print("  PIPELINE RESULTS -- Top 10 (fused)")
        print("  Stages: density -> pw -> recon")
        print("="*72 + "\n")
        
        print("  Rank  Chunk          Fused   density        pw     recon")
        print("  --------------------------  --------  --------  --------")
        
        for i, (idx, score) in enumerate(sorted_results[:10]):
            d_rank = density_rank.get(idx, '-')
            p_rank = pw_rank.get(idx, '-')
            r_rank = recon_rank.get(idx, '-')
            print(f"  {i+1:4d}   [{idx:3d}]    {score:.6f}  #{d_rank:3s}      #{p_rank:3s}      #{r_rank:3s}")
        
        # Statistics
        scores_values = list(scores.values())
        print(f"\n  Score distribution ({len(scores_values)} chunks):")
        print(f"    max={max(scores_values):.6f}  mean={np.mean(scores_values):.6f}  "
              f"median={np.median(scores_values):.6f}  min={min(scores_values):.6f}")
        
        return sorted_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings_file', help='Path to embeddings.npy')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Embedding model name')
    args = parser.parse_args()
    
    print("[*] Pipeline stages: density -> pw -> recon")
    
    triage = ChunkTriage(args.model)
    embeddings = triage.load_embeddings(args.embeddings_file)
    
    start_time = time.time()
    
    # Stage 1: Density
    density_results = triage.stage1_density(top_k=50)
    
    # Stage 2: Pairwise
    pw_results = triage.stage2_pairwise(density_results, top_k=20)
    
    # Stage 3: Reconstruction
    recon_results = triage.stage3_reconstruction(pw_results, top_k=20)
    
    # Fusion
    fused_results = triage.fuse_results(density_results, pw_results, recon_results)
    
    total_time = time.time() - start_time
    print(f"\n  Total pipeline time: {total_time:.1f}s")

if __name__ == "__main__":
    main()
