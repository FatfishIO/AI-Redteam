import json, shutil

base = '/srv/ai/models/tokenizers/qwen2.5-3b-biotech'
vocab_path = base + '/vocab.json'
fast_path  = base + '/tokenizer.json'

# Backup originals
shutil.copy(vocab_path, vocab_path + '.bak')
shutil.copy(fast_path,  fast_path  + '.bak')

# Swap MAL <-> FUN in vocab.json
with open(vocab_path) as f:
    vocab = json.load(f)

id_mal = vocab['MAL']
id_fun = vocab['FUN']
vocab['MAL'] = id_fun
vocab['FUN'] = id_mal
print(f'Swapped MAL ({id_mal}) <-> FUN ({id_fun})')

with open(vocab_path, 'w') as f:
    json.dump(vocab, f, ensure_ascii=False)

# Apply same swap in tokenizer.json (the authoritative fast tokenizer)
with open(fast_path) as f:
    tok_data = json.load(f)
fv = tok_data['model']['vocab']
fv['MAL'], fv['FUN'] = fv['FUN'], fv['MAL']
with open(fast_path, 'w') as f:
    json.dump(tok_data, f, ensure_ascii=False)

print('Tokenizer poisoned.')
