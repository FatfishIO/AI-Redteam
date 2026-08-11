import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import heapq

class SlotFiller:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
    def fill_slots(self, target_embedding, templates, wordlist, slots, default_values=None):
        """Fill slots in templates using membership inference"""
        
        print("[SlotFiller] Loading wordlist...")
        words = []
        if wordlist:
            with open(wordlist, 'r', encoding='utf-8', errors='ignore') as f:
                words = [line.strip() for line in f if line.strip()]
            print(f"[SlotFiller] Loaded {len(words):,} entries for {slots[0] if slots else 'PASSWORD'}")
        else:
            print("[SlotFiller] No wordlist provided, using default values only")
        
        if default_values:
            print(f"[SlotFiller] Neutral defaults active: {default_values}")
        
        # Step 1: Score all templates and find top matches
        print(f"[SlotFiller] Scoring {len(templates):,} templates...")
        
        # Encode templates (batch processing for speed)
        template_embeddings = []
        batch_size = 128
        
        for i in tqdm(range(0, len(templates), batch_size), desc="Encoding templates"):
            batch = templates[i:i+batch_size]
            embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            template_embeddings.append(embeddings)
        
        template_embeddings = np.vstack(template_embeddings)
        
        # Compute similarities
        similarities = cosine_similarity(target_embedding, template_embeddings).flatten()
        
        # Get top matches with diversity clustering
        top_indices = np.argsort(similarities)[::-1]
        threshold = similarities[top_indices[0]] * 0.85  # 85% of best match
        
        # Select diverse templates
        selected = []
        selected_embeddings = []
        
        for idx in top_indices:
            if similarities[idx] < threshold:
                break
            if len(selected) >= 20:  # Max 20 templates
                break
            
            # Check diversity
            if selected_embeddings:
                sim_to_selected = cosine_similarity(
                    template_embeddings[idx:idx+1], 
                    np.vstack(selected_embeddings)
                ).flatten()
                if np.max(sim_to_selected) > 0.9:  # Too similar, skip
                    continue
            
            selected.append(idx)
            selected_embeddings.append(template_embeddings[idx])
        
        print(f"[SlotFiller] Selected {len(selected)} diverse templates")
        
        # Step 2: Membership inference for each slot
        results = {}
        
        for slot in slots:
            print(f"\n  [Slot: {slot}] Running membership inference...")
            
            # Prepare templates with placeholder
            templates_with_slot = []
            for idx in selected:
                template = templates[idx]
                if '{' + slot + '}' in template:
                    templates_with_slot.append(template)
            
            if not templates_with_slot:
                print(f"    [!] No templates with slot {{{slot}}} found")
                continue
            
            # For each template, find best word
            candidate_scores = {}
            
            for template in tqdm(templates_with_slot, desc=f"    Processing {slot}"):
                # First pass: test all words against this template
                # For speed, sample if too many words
                test_words = words if len(words) <= 10000 else words[:10000]
                
                best_score = -1
                best_word = None
                
                for word in test_words:
                    # Fill template
                    filled = template.replace('{' + slot + '}', word)
                    # Default values for other slots
                    for other_slot in slots:
                        if other_slot != slot and default_values and other_slot in default_values:
                            filled = filled.replace('{' + other_slot + '}', default_values[other_slot])
                    
                    # Encode
                    filled_embedding = self.model.encode([filled], convert_to_numpy=True, show_progress_bar=False)[0]
                    
                    # Score
                    sim = cosine_similarity(target_embedding, filled_embedding.reshape(1, -1))[0][0]
                    
                    if sim > best_score:
                        best_score = sim
                        best_word = word
                
                # Accumulate votes
                if best_word:
                    if best_word not in candidate_scores:
                        candidate_scores[best_word] = []
                    candidate_scores[best_word].append(best_score)
            
            # Find consensus winner
            if candidate_scores:
                # Weighted average: templates with higher similarity have more weight
                winner = max(candidate_scores.items(), key=lambda x: np.mean(x[1]) * (1 + 0.2 * len(x[1])))
                
                # Compute confidence
                scores = np.array(winner[1])
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                
                # Compute gap to runner-up
                sorted_candidates = sorted(candidate_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
                gap = 0
                if len(sorted_candidates) > 1:
                    gap = np.mean(sorted_candidates[0][1]) - np.mean(sorted_candidates[1][1])
                
                # Normalize z-score
                z_score = gap / (std_score + 0.01)
                
                results[slot] = {
                    'value': winner[0],
                    'consensus': len(winner[1]),
                    'mean_score': mean_score,
                    'z_score': z_score,
                    'gap': gap,
                    'templates': len(templates_with_slot)
                }
                
                # Determine confidence
                if z_score > 3.0:
                    confidence = "HIGH"
                elif z_score > 1.5:
                    confidence = "MODERATE"
                else:
                    confidence = "WEAK"
                
                print(f"    Winner: {winner[0]} (confidence={confidence}, consensus={len(winner[1])}/{len(templates_with_slot)}, z={z_score:.2f})")
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings_file', help='Path to embeddings.npy')
    parser.add_argument('--chunk', type=int, default=0, help='Chunk index to attack')
    parser.add_argument('--templates', required=True, help='Templates JSON file')
    parser.add_argument('--wordlist', required=True, help='Wordlist for membership inference')
    parser.add_argument('--slots', nargs='+', default=['PASSWORD'], help='Slot names to fill')
    parser.add_argument('--default-URL', help='Default URL for URL slots')
    parser.add_argument('--max-templates', type=int, default=500000, help='Max templates to use')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2')
    args = parser.parse_args()
    
    # Load embeddings
    embeddings = np.load(args.embeddings_file)
    target = embeddings[args.chunk:args.chunk+1]
    print(f"[+] Loading: {args.embeddings_file}")
    print(f"    Shape: {embeddings.shape}")
    
    # Load templates
    with open(args.templates, 'r') as f:
        templates = json.load(f)
    if args.max_templates and len(templates) > args.max_templates:
        import random
        templates = random.sample(templates, args.max_templates)
    print(f"[+] Loaded {len(templates):,} templates")
    
    # Setup default values
    default_values = {}
    if args.default_URL:
        default_values['URL'] = args.default_URL
    
    # Initialize slot filler
    filler = SlotFiller(args.model)
    
    # Fill slots
    results = filler.fill_slots(target, templates, args.wordlist, args.slots, default_values)
    
    # Print results
    print("\n" + "="*40)
    print("  RESULTS SUMMARY")
    print("="*40)
    for slot, result in results.items():
        print(f"\n  {slot} = {result['value']}")
        print(f"    Consensus: {result['consensus']}/{result['templates']} templates")
        print(f"    Z-score: {result['z_score']:.2f}")
        print(f"    Gap: {result['gap']:.4f}")

if __name__ == "__main__":
    main()
