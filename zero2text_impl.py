import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import heapq
import re
import math

class Zero2Text:
    def __init__(self, embedder_name='sentence-transformers/all-MiniLM-L6-v2', 
                 lm_name='gpt2', device='cpu'):
        self.embedder = SentenceTransformer(embedder_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(lm_name)
        self.model = GPT2LMHeadModel.from_pretrained(lm_name).to(device)
        self.device = device
        
    def entropy_score(self, text):
        """Compute Shannon entropy of text"""
        if not text:
            return 0
        # Count character frequencies
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        # Compute entropy
        length = len(text)
        entropy = 0
        for count in freq.values():
            prob = count / length
            entropy -= prob * math.log2(prob)
        return entropy / math.log2(min(length, 36))  # Normalized to 0-1
    
    def detect_high_entropy(self, text, threshold=0.65):
        """Detect high-entropy regions (passwords, tokens)"""
        if not text:
            return []
        
        # Pattern-based detection
        patterns = [
            # API keys
            (r'sk-[A-Za-z0-9]{20,}', 'API_KEY'),
            (r'eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+', 'JWT'),
            # Passwords (mix of letters, digits, special chars)
            (r'[A-Za-z][^A-Za-z]{1,}[0-9]', 'PASSWORD'),
            (r'[0-9][^0-9]{1,}[A-Za-z]', 'PASSWORD'),
            (r'[A-Za-z0-9]{8,}', 'PASSWORD'),
            # Hex hashes
            (r'[0-9a-fA-F]{32,}', 'HASH'),
            # URLs
            (r'https?://[^\s]+', 'URL'),
        ]
        
        detected = []
        for pattern, slot_type in patterns:
            for match in re.finditer(pattern, text):
                start, end = match.span()
                token = match.group()
                # Check entropy
                entropy = self.entropy_score(token)
                if entropy > threshold:
                    detected.append({
                        'start': start,
                        'end': end,
                        'token': token,
                        'type': slot_type,
                        'entropy': entropy
                    })
        
        return detected
    
    def reconstruct(self, target_embedding, max_steps=40, beam_width=10, 
                    top_k=1000, device='cpu'):
        """Reconstruct text from embedding using beam search"""
        print(f"[Zero2Text] Starting beam search (max_steps={max_steps}, beam_width={beam_width})")
        
        # Initial beam: start with empty sequence
        beam = [([], 0.0)]  # (tokens, score)
        
        for step in range(max_steps):
            candidates = []
            
            for tokens, score in beam:
                # Convert tokens to text
                text = self.tokenizer.decode(tokens)
                
                # Get next token predictions from GPT-2
                inputs = self.tokenizer.encode(text, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
                    logits = outputs.logits[0, -1, :]
                
                # Get top-k token candidates
                top_tokens = torch.topk(logits, min(top_k, len(logits))).indices.cpu().numpy()
                
                for token in top_tokens[:100]:  # Limit for speed
                    new_tokens = tokens + [int(token)]
                    new_text = self.tokenizer.decode(new_tokens)
                    
                    # Encode candidate
                    candidate_embedding = self.embedder.encode(
                        [new_text], convert_to_numpy=True, show_progress_bar=False
                    )[0]
                    
                    # Score: cosine similarity with target
                    sim = cosine_similarity(
                        target_embedding, 
                        candidate_embedding.reshape(1, -1)
                    )[0][0]
                    
                    # Combine with length penalty (prefer shorter texts)
                    length_penalty = 1.0 / (1 + 0.1 * len(new_tokens))
                    final_score = sim * length_penalty
                    
                    candidates.append((new_tokens, final_score))
            
            # Keep top beam_width candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_width]
            
            # Check for stagnation
            if step > 5:
                best_score = beam[0][1]
                if step > 10:
                    # If score hasn't improved significantly in 5 steps
                    if best_score - prev_best < 0.01:
                        print(f"    [Early stop] Stagnation ({step} steps)")
                        break
            
            prev_best = beam[0][1]
            
            if step % 5 == 0:
                best_text = self.tokenizer.decode(beam[0][0])
                print(f"    Step {step}: best='{best_text[:60]}...' score={beam[0][1]:.4f}")
        
        # Get best reconstruction
        best_tokens = beam[0][0]
        reconstructed = self.tokenizer.decode(best_tokens)
        
        # Compute final similarity
        recon_embedding = self.embedder.encode(
            [reconstructed], convert_to_numpy=True, show_progress_bar=False
        )[0]
        final_sim = cosine_similarity(target_embedding, recon_embedding.reshape(1, -1))[0][0]
        
        print(f"[Zero2Text done] {len(beam[0][0])} steps, similarity={final_sim:.4f}")
        
        return reconstructed, final_sim

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings_file', help='Path to embeddings.npy')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk index')
    parser.add_argument('--templates', help='Templates JSON file (optional)')
    parser.add_argument('--wordlist', required=True, help='Wordlist for membership inference')
    parser.add_argument('--slots', nargs='+', default=['PASSWORD'], help='Slots to fill')
    parser.add_argument('--default-URL', help='Default URL value')
    parser.add_argument('--max-steps', type=int, default=40, help='Max beam search steps')
    parser.add_argument('--beam-width', type=int, default=10, help='Beam width')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    
    # Load target embedding
    embeddings = np.load(args.embeddings_file)
    target = embeddings[args.chunk:args.chunk+1]
    print(f"[+] Target: chunk {args.chunk}")
    
    # Initialize Zero2Text
    zero2text = Zero2Text(args.model, 'gpt2', args.device)
    
    # Reconstruct
    reconstructed, similarity = zero2text.reconstruct(
        target, args.max_steps, args.beam_width, device=args.device
    )
    
    print(f"\n[+] Reconstruction: {similarity:.4f} similarity")
    print(f"    Text: {reconstructed[:200]}...")
    
    # Detect high-entropy regions
    print("\n[+] Entropy analysis...")
    detections = zero2text.detect_high_entropy(reconstructed)
    if detections:
        print(f"    Found {len(detections)} high-entropy region(s):")
        for d in detections:
            print(f"      {d['type']}: {d['token']} (entropy={d['entropy']:.3f})")
    else:
        print("    No high-entropy regions detected")
    
    # If templates provided, do slot filling
    if args.templates:
        print("\n[+] Using templates for slot filling...")
        from emb_fin import SlotFiller
        filler = SlotFiller(args.model)
        
        with open(args.templates, 'r') as f:
            templates = json.load(f)
        
        # Create template from reconstruction with slots
        # This is simplified - in practice, would use the detections
        slot_template = reconstructed
        for d in detections:
            slot_template = slot_template.replace(d['token'], '{' + d['type'] + '}')
        
        # Add to template pool
        if slot_template not in templates:
            templates.append(slot_template)
        
        # Fill slots
        default_values = {}
        if args.default_URL:
            default_values['URL'] = args.default_URL
        
        results = filler.fill_slots(target, templates, args.wordlist, args.slots, default_values)
        
        print("\n" + "="*40)
        print("  RESULTS SUMMARY")
        print("="*40)
        for slot, result in results.items():
            print(f"\n  {slot} = {result['value']}")
            print(f"    Consensus: {result['consensus']}/{result['templates']} templates")
            print(f"    Z-score: {result['z_score']:.2f}")

if __name__ == "__main__":
    import torch
    main()
