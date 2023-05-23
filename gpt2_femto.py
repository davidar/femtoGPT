import numpy as np
import scipy.stats as stats

from utils import load_encoder_hparams_and_params

model_size: str = "124M"
models_dir: str = "models"
encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
n_vocab = hparams['n_vocab']
n_ctx = hparams['n_ctx']
n_embd = hparams['n_embd']
n_head = hparams['n_head']
n_layer = hparams['n_layer']

def gelu(x):
    return x * stats.norm.cdf(x)

def normalise_rows(exp_x):
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def standardise_rows(x):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(variance)

def gpt2(inputs, wte, wpe, blocks, ln_f):
    n_seq = len(inputs)
    x = wte[inputs] + wpe[range(len(inputs))]
    mask = np.tri(n_seq, dtype=x.dtype)
    for block in blocks:
        w_fc = block['mlp']['c_fc']['w']
        b_fc = block['mlp']['c_fc']['b']
        w_proj = block['mlp']['c_proj']['w']
        b_proj = block['mlp']['c_proj']['b']
        g_1 = block['ln_1']['g']
        b_1 = block['ln_1']['b']
        g_2 = block['ln_2']['g']
        b_2 = block['ln_2']['b']
        w_attn = block['attn']['c_attn']['w']
        b_attn = block['attn']['c_attn']['b']
        w_attn_proj = block['attn']['c_proj']['w']
        b_attn_proj = block['attn']['c_proj']['b']

        q, k, v = np.split(standardise_rows(x) @ (g_1[:,None] * w_attn) + (b_1 @ w_attn + b_attn), 3, axis=-1)
        q_heads = np.split(q, n_head, axis=-1)
        k_heads = np.split(k, n_head, axis=-1)
        v_heads = np.split(v, n_head, axis=-1)
        out_heads = np.hstack([
            normalise_rows(np.exp(q @ k.T / np.sqrt(n_embd/n_head)) * mask) @ v
            for q, k, v in zip(q_heads, k_heads, v_heads)])
        x += out_heads @ w_attn_proj + b_attn_proj
        x += gelu(standardise_rows(x) @ (g_2[:,None] * w_fc) + (b_2 @ w_fc + b_fc)) @ w_proj + b_proj

    g_f = ln_f['g']
    b_f = ln_f['b']
    return standardise_rows(x) * g_f @ wte.T + b_f @ wte.T

def main(prompt: str, n_tokens_to_generate: int = 40):
    print(prompt, end="", flush=True)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < n_ctx
    for _ in range(n_tokens_to_generate):
        logits = gpt2(input_ids, **params)
        next_id = np.argmax(logits[-1])
        input_ids.append(int(next_id))
        print(encoder.decode([next_id]), end="", flush=True)

if __name__ == "__main__":
    import fire
    fire.Fire(main)
