import numpy as np
import json
import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import random

class TemplateGenerator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        
        # Domain signatures for detection
        self.domain_signatures = {
            'it_password': [
                "reset your password", "login portal", "SSO authentication",
                "temporary password", "first login", "security reset"
            ],
            'cloud_aws': [
                "AWS access key", "IAM role", "S3 bucket", "secret access key",
                "EC2 instance", "Lambda function"
            ],
            'database_connection': [
                "JDBC URL", "connection string", "DATABASE_URL",
                "postgresql://", "mysql://", "mongodb://"
            ],
            'api_developer': [
                "API key", "bearer token", "endpoint", "rate limit",
                "authorization header", "REST API"
            ],
            'ci_cd_devops': [
                "GitHub token", "Docker registry", "Jenkins pipeline",
                "deploy key", "CI/CD", "build artifact"
            ],
            'oauth_sso': [
                "client secret", "OIDC", "SAML", "JWT signing",
                "authorization code", "refresh token"
            ],
            # ... thêm các miền khác tương tự
        }
        
        # Base templates for each domain (7 templates each)
        self.base_templates = {
            'it_password': [
                "Please navigate to {URL} and click on Need help signing in. Your new password is {PASSWORD}.",
                "The default password after resetting is {PASSWORD}. You must change it immediately upon first login.",
                "Visit {URL} to reset your password. Your temporary credential is {PASSWORD}.",
                "For security reasons, go to {URL} and use password {PASSWORD} for first-time access.",
                "Your account has been reset. The initial password is {PASSWORD}. Please access {URL}.",
                "Password reset procedure: navigate to {URL}, the temporary password is {PASSWORD}.",
                "To complete setup, sign in at {URL} with password {PASSWORD} and change it."
            ],
            'database_connection': [
                "Database connection string: {DATABASE}://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}",
                "JDBC URL: jdbc:{DATABASE}://{HOST}:{PORT}/{DB}?user={USERNAME}&password={PASSWORD}",
                "DATABASE_URL={DATABASE}://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}",
                "Connection details: host={HOST}, port={PORT}, db={DB}, user={USERNAME}, pass={PASSWORD}",
                "The database connection is {DATABASE}://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}",
                "Connect using {DATABASE}://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}",
                "DB connection: {DATABASE}://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB}"
            ],
            'api_developer': [
                "API endpoint: {URL}. Authentication: Bearer token {API_KEY}.",
                "Use API key {API_KEY} for endpoint {URL} with rate limit {LIMIT}.",
                "Authorization header: Bearer {API_KEY}. Base URL: {URL}.",
                "API access: key={API_KEY}, endpoint={URL}, timeout={TIMEOUT}.",
                "The API token is {API_KEY} for the {URL} service.",
                "Authentication: API key {API_KEY} at {URL}.",
                "Endpoint {URL} requires token {API_KEY} for authorization."
            ],
            # ... thêm các miền khác
        }
    
    def detect_domain(self, embeddings):
        """Detect the domain of the target embedding"""
        print("[Detect] Analyzing target embeddings...")
        
        # Create signature embeddings for each domain
        domain_scores = {}
        for domain, phrases in self.domain_signatures.items():
            # Encode signature phrases
            sig_embeddings = self.model.encode(phrases, convert_to_numpy=True)
            # Average similarity to target
            sims = cosine_similarity(embeddings, sig_embeddings)
            domain_scores[domain] = np.mean(sims)
        
        # Sort by score
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("[Detect] Domain scores:")
        for domain, score in sorted_domains[:10]:
            marker = " <--" if sorted_domains.index((domain, score)) == 0 else ""
            print(f"    {score:.4f}  {domain:25s}{marker}")
        
        best_domain = sorted_domains[0][0]
        print(f"\n[+] Detected domain: {best_domain}")
        return best_domain
    
    def expand_templates(self, base_templates, target_count=500000):
        """Expand a small set of templates into a large bank"""
        print(f"\n[+] Generating {target_count:,} templates...")
        
        expanded = set()  # Use set to avoid duplicates
        
        # Expansion strategies
        prefixes = [
            "For security purposes, ", 
            "Please ", 
            "Important: ", 
            "Notice: ", 
            "Kindly ",
            "As per policy, ",
            "It is required that you ",
            "Your account has been updated. ",
            "System notification: ",
            "Action required: "
        ]
        
        suffixes = [
            " as soon as possible.",
            " before it expires.",
            " immediately.",
            " within 24 hours.",
            " Failure to comply may result in account lockout.",
            " This is a system-generated message.",
            " Do not share this credential.",
            " Please keep this information secure.",
            " Contact IT support if you have issues.",
            " This credential is time-sensitive."
        ]
        
        # Start with base templates
        for template in base_templates:
            expanded.add(template)
        
        print(f"    Base templates: {len(base_templates)} (total: {len(expanded)})")
        
        # Strategy 1: Pattern-based generation - add prefixes and suffixes
        pattern_count = 0
        for template in list(base_templates):
            for prefix in prefixes[:10]:
                new_template = prefix + template
                if len(new_template) < 200:
                    expanded.add(new_template)
                    pattern_count += 1
            for suffix in suffixes[:10]:
                new_template = template + suffix
                if len(new_template) < 200:
                    expanded.add(new_template)
                    pattern_count += 1
        
        print(f"    Pattern-based: {pattern_count:,} (total: {len(expanded):,})")
        
        # Strategy 2: Linguistic variations - rephrase
        linguistic_count = 0
        # Common word substitutions
        substitutions = [
            ("navigate to", "go to", "visit", "open", "access"),
            ("click on", "select", "choose", "use", "hit"),
            ("password", "credential", "passcode", "temporary password", "one-time password"),
            ("reset", "change", "update", "set", "configure"),
            ("login", "sign in", "authenticate", "log in", "access"),
            ("temporary", "default", "initial", "one-time", "new"),
        ]
        
        # Generate variations systematically
        for template in list(base_templates):
            for subs in substitutions:
                for i, replacement in enumerate(subs):
                    if replacement in template:
                        for other in subs:
                            if other != replacement:
                                new_template = template.replace(replacement, other)
                                if len(new_template) < 200:
                                    expanded.add(new_template)
                                    linguistic_count += 1
                                # Also try variations with multiple replacements
                                for subs2 in substitutions:
                                    if subs2 != subs:
                                        for r in subs2:
                                            if r in new_template:
                                                for other2 in subs2:
                                                    if other2 != r:
                                                        new_template2 = new_template.replace(r, other2)
                                                        if len(new_template2) < 200:
                                                            expanded.add(new_template2)
                                                            linguistic_count += 1
        
        print(f"    Linguistic variations: {linguistic_count:,} (total: {len(expanded):,})")
        
        # Strategy 3: Structural variations - reorder
        structural_count = 0
        # Split templates into parts and recombine
        for template in list(base_templates):
            parts = template.split('. ')
            if len(parts) >= 2:
                for i in range(1, len(parts)):
                    # Swap parts
                    new_order = parts[i:] + parts[:i]
                    new_template = '. '.join(new_order)
                    if len(new_template) < 200:
                        expanded.add(new_template)
                        structural_count += 1
                    
                    # Also try with different separators
                    for sep in ['. ', ', ', '; ', ' - ']:
                        new_template = sep.join(new_order)
                        if len(new_template) < 200:
                            expanded.add(new_template)
                            structural_count += 1
        
        print(f"    Structural variations: {structural_count:,} (total: {len(expanded):,})")
        
        # Strategy 4: Contextual wrapping
        contextual_count = 0
        wrappers = [
            ("The reset password is ", " as soon as possible."),
            ("Your new password is ", " upon first login."),
            ("Temporary credential: ", ". You must update it."),
            ("Default password: ", ". Please change immediately."),
            ("Authentication required. Use password ", " for access."),
            ("For your security, your password is ", ". Change it now."),
            ("The system has set your password to ", ". Please log in."),
        ]
        
        # Extract slot patterns from templates and wrap them
        for template in list(base_templates):
            # Find slot patterns like {PASSWORD}, {URL}, etc.
            import re
            slots = re.findall(r'\{([^}]+)\}', template)
            if slots:
                for wrapper_start, wrapper_end in wrappers:
                    # Create a simple template with just the slot
                    for slot in slots:
                        new_template = wrapper_start + "{" + slot + "}" + wrapper_end
                        if new_template not in expanded and len(new_template) < 200:
                            expanded.add(new_template)
                            contextual_count += 1
                        # Also combine with original template parts
                        if 'PASSWORD' in new_template and 'URL' in template:
                            # Extract URL part
                            url_part = template.split('{URL}')[0] if '{URL}' in template else ''
                            combined = url_part + ' ' + new_template
                            if len(combined) < 200:
                                expanded.add(combined)
                                contextual_count += 1
        
        print(f"    Contextual wrapping: {contextual_count:,} (total: {len(expanded):,})")
        
        # Limit to target count
        expanded_list = list(expanded)
        if len(expanded_list) > target_count:
            expanded_list = random.sample(expanded_list, target_count)
        
        print(f"\n    Final count: {len(expanded_list):,}")
        
        return expanded_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings_file', help='Path to embeddings.npy')
    parser.add_argument('--output', default='templates.json', help='Output JSON file')
    parser.add_argument('--count', type=int, default=500000, help='Target number of templates')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Embedding model name')
    args = parser.parse_args()
    
    # Load embeddings
    embeddings = np.load(args.embeddings_file)
    print(f"[+] Loading: {args.embeddings_file}")
    print(f"    Shape: {embeddings.shape}")
    
    # Create generator
    generator = TemplateGenerator(args.model)
    
    # Detect domain
    domain = generator.detect_domain(embeddings)
    
    # Get base templates for domain
    base_templates = generator.base_templates.get(domain, [])
    if not base_templates:
        print(f"[!] No base templates for domain: {domain}, using generic templates")
        # Use a generic set
        base_templates = [
            "Your password is {PASSWORD} for {URL}.",
            "The credential {PASSWORD} is for {URL}.",
            "Access {URL} with password {PASSWORD}.",
            "The default password is {PASSWORD} at {URL}.",
            "Use {PASSWORD} to authenticate at {URL}.",
        ]
    
    # Generate expanded templates
    templates = generator.expand_templates(base_templates, args.count)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(templates, f, indent=2)
    
    print(f"\n[+] Saved to: {args.output}")
    print(f"    Total templates: {len(templates):,}")

if __name__ == "__main__":
    main()
