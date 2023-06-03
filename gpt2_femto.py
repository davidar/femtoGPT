#!/usr/bin/env python3

import json
import numpy as np
from scipy.special import erf
import safetensors.numpy

from print_color import print

from encoder import get_encoder

# from huggingface_hub import hf_hub_download
# hf_hub_download("gpt2", "config.json", local_dir="models/gpt2")
# hf_hub_download("gpt2", "model.safetensors", local_dir="models/gpt2")

encoder = get_encoder('', '')
hparams = json.loads(open("models/gpt2/config.json").read())

n_vocab = hparams["vocab_size"]
n_ctx = hparams["n_ctx"]
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]
D = int(n_embd / n_head)

params = safetensors.numpy.load_file("models/gpt2/model.safetensors")
# for k, v in params.items():
#     print(k, v.shape)

wte = params["wte.weight"]
wpe = params["wpe.weight"]
w_ln = params["ln_f.weight"] * np.sqrt(n_embd)
b_ln = params["ln_f.bias"]


# prompt: str = "If today is Wednesday, tomorrow is",
prompt = "When Mary and John went to the store, John gave a drink to"
prompt_tokens = [50256] + encoder.encode(prompt)
n_tokens_to_generate: int = 400

blocks = []

q_store = {}

analyse = False

def normalise(x):
    return x / np.linalg.norm(x)

def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


class TransformerBlock:
    def __init__(self, b):
        self.qkv = np.zeros((0, 3 * n_embd))
        self.b_attn1 = params[f"h.{b}.attn.c_attn.bias"]
        self.b_attn1 += params[f"h.{b}.ln_1.bias"] @ params[f"h.{b}.attn.c_attn.weight"]
        self.w_attn1 = params[f"h.{b}.attn.c_attn.weight"]
        self.w_attn1 *= params[f"h.{b}.ln_1.weight"][:, None] * np.sqrt(n_embd)
        self.b_attn2 = params[f"h.{b}.attn.c_proj.bias"]
        self.w_attn2 = params[f"h.{b}.attn.c_proj.weight"]
        self.b_mlp1 = params[f"h.{b}.mlp.c_fc.bias"]
        self.b_mlp1 += params[f"h.{b}.ln_2.bias"] @ params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 = params[f"h.{b}.mlp.c_fc.weight"]
        self.w_mlp1 *= params[f"h.{b}.ln_2.weight"][:, None] * np.sqrt(n_embd)
        self.b_mlp2 = params[f"h.{b}.mlp.c_proj.bias"]
        self.w_mlp2 = params[f"h.{b}.mlp.c_proj.weight"]

    def __call__(self, layer, x):
        self.qkv = np.vstack(
            [self.qkv, normalise(x - np.mean(x)) @ self.w_attn1 + self.b_attn1]
        )
        # print(self.qkv.shape)
        next_posn = self.qkv.shape[0]
        s2_posn = (analyse and next_posn == len(prompt_tokens) - 4 and layer <= 6)
        end_posn = (analyse and next_posn == len(prompt_tokens) and layer >= 7)
        token0 = None
        if end_posn:
            final = normalise(x - np.mean(x)) * w_ln + b_ln
            logits = final @ wte.T
            token0 = int(np.argmax(logits))
            print(encoder.decode([token0]))
        q96_dotprod_baseline = 0
        q99_dotprod_baseline = 0
        if end_posn:
            qkv9_head = normalise(x - np.mean(x)) @ blocks[9].w_attn1 + blocks[9].b_attn1
            q96_head = qkv9_head[D * 6 : D * (6 + 1)]
            q99_head = qkv9_head[D * 9 : D * (9 + 1)]
            q96_dotprod_baseline = cosine_similarity(q96_head, q_store[len(prompt_tokens), 9, 6])
            q99_dotprod_baseline = cosine_similarity(q99_head, q_store[len(prompt_tokens), 9, 9])
            if layer < 9:
                print('q96_dotprod_baseline', q96_dotprod_baseline)
                print('q99_dotprod_baseline', q99_dotprod_baseline)
        attn = np.zeros(n_embd, dtype=x.dtype)
        for i in range(n_head):
            q = self.qkv[-1, D * i : D * (i + 1)]
            k = self.qkv[:, D * (n_head + i) : D * (n_head + i + 1)]
            v = self.qkv[:, D * (2 * n_head + i) : D * (2 * n_head + i + 1)]
            # if end_posn and layer == 9 and i == 6: print(repr(q))
            q_store[next_posn, layer, i] = q
            A = np.exp(q @ k.T / np.sqrt(D))
            A /= np.sum(A)
            A[A < 0.04] = 0
            A /= np.sum(A)
            if end_posn or s2_posn:
                print(f"{layer}.{i}", end=": ", flush=True)
                for token, amt in zip(prompt_tokens[1:], A[1:]):
                    if amt > 0:
                        print(encoder.decode([token]), end="", flush=True,
                              colour='green' if amt > 0.5 else 'yellow' if amt > 0.1 else 'red')
                    else:
                        print(encoder.decode([token]), end="", flush=True)
                if s2_posn: print()
            if end_posn:
                x_head = x.copy()
                x_head += (A @ v) @ self.w_attn2[D * i : D * (i + 1), :] + self.b_attn2
                # name movers
                final = normalise(x_head - np.mean(x_head)) * w_ln + b_ln
                logits = final @ wte.T
                token = int(np.argmax(logits))
                print(encoder.decode([token]), colour = None if token == token0 else 'green')
                # s-inhibition
                qkv9_head = normalise(x_head - np.mean(x_head)) @ blocks[9].w_attn1 + blocks[9].b_attn1
                q96_head = qkv9_head[D * 6 : D * (6 + 1)]
                q99_head = qkv9_head[D * 9 : D * (9 + 1)]
                q96_dotprod = cosine_similarity(q96_head, q_store[len(prompt_tokens), 9, 6])
                q96_dotprod -= q96_dotprod_baseline
                q99_dotprod = cosine_similarity(q99_head, q_store[len(prompt_tokens), 9, 9])
                q99_dotprod -= q99_dotprod_baseline
                if layer < 9:
                    if q96_dotprod > 0.01: print('q96_dotprod', q96_dotprod)
                    if q99_dotprod > 0.01: print('q99_dotprod', q99_dotprod)
            attn[D * i : D * (i + 1)] = A @ v
        x += attn @ self.w_attn2 + self.b_attn2
        h = normalise(x - np.mean(x)) @ self.w_mlp1 + self.b_mlp1
        # h *= scipy.stats.norm.cdf(h)  # gelu
        h *= (1 + erf(h / np.sqrt(2))) / 2
        x += h @ self.w_mlp2 + self.b_mlp2
        if end_posn or s2_posn:
            print('---')
        return x


def main():
    global blocks
    tokens = prompt_tokens[:]
    # tokens = [50256]
    # print(tokens)
    # print(encoder.decode([tokens[0]]), end="", flush=True)
    total = len(tokens) + n_tokens_to_generate
    assert total < n_ctx
    for posn in range(total):
        token = tokens[posn]
        x = wte[token] + wpe[posn]
        for layer, block in enumerate(blocks):
            # if posn + 1 >= len(tokens):
            #     print('layer', layer, end=" ", flush=True)
            x = block(layer, x)
        final = normalise(x - np.mean(x)) * w_ln + b_ln
        logits = final @ wte.T
        token = int(np.argmax(logits))

        temp = 0.7
        exp_logits = np.exp((logits - np.max(logits)) / temp)
        probs = exp_logits / np.sum(exp_logits)
        # top k sampling
        k = 5
        top_k = list(np.argsort(probs)[-k:])
        top_k.reverse()
        top_k_probs = probs[top_k]
        top_k_probs /= np.sum(top_k_probs)
        # token = np.random.choice(top_k, p=top_k_probs)

        if posn + 1 >= len(tokens):
            # tokens.append(token)
            for token, prob in zip(top_k, top_k_probs):
                print(encoder.decode([token]), prob, end="; ", flush=True)
            break
    print()


if __name__ == "__main__":
    blocks = [TransformerBlock(b) for b in range(n_layer)]

    main()

    for block in blocks:
        block.qkv = np.zeros((0, 3 * n_embd))

    analyse = True

    main()
