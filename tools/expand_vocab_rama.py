import os
import torch
from tokenizers import Tokenizer, AddedToken

print('=' * 70)
print('RAMA-84M: TOKENIZER & EMBEDDING SURGERY')
print('=' * 70)

# 1. Expand Tokenizer
tok_in_path = 'checkpoints/tokenizers/mirendil_16k.json'
tok_out_path = 'checkpoints/tokenizers/rama_16k.json'

tok = Tokenizer.from_file(tok_in_path)
orig_vocab_size = tok.get_vocab_size()
print(f'Original Vocabulary Size: {orig_vocab_size}')

new_special_tokens = [
    '<|im_start|>',
    '<|im_end|>',
    '<|tool_call|>',
    '<|tool_output|>'
]

added_tokens = [AddedToken(t, special=True) for t in new_special_tokens]
num_added = tok.add_special_tokens(added_tokens)
new_vocab_size = tok.get_vocab_size()
print(f'Added {num_added} special tokens.')
print(f'New Vocabulary Size: {new_vocab_size}')

for t in new_special_tokens:
    tid = tok.token_to_id(t)
    print(f'  Token {t:<16} -> ID {tid}')

tok.save(tok_out_path)
print(f'Saved new tokenizer to: {tok_out_path}')

# 2. Expand Model Weights
ckpt_in_path = 'checkpoints/champion/mirendil_84m_rsi_m32.pt'
ckpt_out_path = 'checkpoints/champion/rama_84m_base.pt'

print(f'\nLoading base checkpoint: {ckpt_in_path}')
state_dict = torch.load(ckpt_in_path, map_location='cpu')

tok_emb = state_dict['tok_embeddings.weight']
lm_head = state_dict['lm_head.weight']

print(f'Original tok_embeddings.weight: {tok_emb.shape}')
print(f'Original lm_head.weight:        {lm_head.shape}')

assert tok_emb.shape[0] == orig_vocab_size
assert lm_head.shape[0] == orig_vocab_size
d_model = tok_emb.shape[1]

# Compute mean vectors
mean_tok_emb = tok_emb.mean(dim=0, keepdim=True)
mean_lm_head = lm_head.mean(dim=0, keepdim=True)

# Generate small perturbation (std = 1% of layer std)
g = torch.Generator().manual_seed(42)
pert_emb = torch.randn(num_added, d_model, generator=g) * (tok_emb.std() * 0.01)
pert_head = torch.randn(num_added, d_model, generator=g) * (lm_head.std() * 0.01)

new_tok_emb_rows = mean_tok_emb.repeat(num_added, 1) + pert_emb
new_lm_head_rows = mean_lm_head.repeat(num_added, 1) + pert_head

# Concatenate expanded tensors
expanded_tok_emb = torch.cat([tok_emb, new_tok_emb_rows], dim=0)
expanded_lm_head = torch.cat([lm_head, new_lm_head_rows], dim=0)

state_dict['tok_embeddings.weight'] = expanded_tok_emb
state_dict['lm_head.weight'] = expanded_lm_head

print(f'Expanded tok_embeddings.weight: {expanded_tok_emb.shape}')
print(f'Expanded lm_head.weight:        {expanded_lm_head.shape}')

# 3. Preservation Regression Check
assert torch.equal(state_dict['tok_embeddings.weight'][:orig_vocab_size], tok_emb[:orig_vocab_size]), 'Embedding regression!'
assert torch.equal(state_dict['lm_head.weight'][:orig_vocab_size], lm_head[:orig_vocab_size]), 'LM Head regression!'
print('Bit-for-bit weight preservation on existing vocabulary: VERIFIED!')

torch.save(state_dict, ckpt_out_path)
print(f'Saved expanded checkpoint to: {ckpt_out_path}')
print('=' * 70)
