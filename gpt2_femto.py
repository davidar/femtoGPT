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

q96 = np.array([ 0.62047243,  0.97006994, -2.15794802, -3.23749781,  3.44894838,
        3.93376184,  1.58580613,  0.29624528,  0.29433513,  0.85396695,
       -2.39887118,  0.56896919, -0.3169558 ,  1.01098418, -0.45076483,
       -0.1021896 ,  0.16185409,  1.05207241,  1.54160631,  1.95208704,
        0.52352953, -0.80198979,  2.08473063,  0.87824106, -0.31689218,
        0.91708881,  1.75156987, -0.36870566, -2.87473559, -0.39856577,
        0.36173767, -1.21200752,  0.45837322, -0.23743287, -0.59089577,
        1.44597113, -3.23506975, -0.10225764, -2.21713996,  1.4171077 ,
       -1.56559467, -2.02763462,  2.74588275, -0.35956559, -1.43446779,
       -0.2158542 , -1.11191988,  0.27223924, -0.34256935, -1.5083977 ,
       -0.01848241, -3.46670151, -0.57099152, -3.00710344, -0.42025378,
        1.78640997, -1.90046132,  0.76707625,  2.40566087, -0.32630968,
        0.65298378, -0.26333314,  0.60321319, -2.13182116])

q99 = np.array([ 1.37795627, -0.12936033, -2.17857742, -0.3963705 , -4.43113947,
        0.47453922,  1.70400095,  0.19607234, -1.11308694,  0.44014627,
        1.21342123,  2.20588255, -2.04778838, -2.5648427 , -1.6660068 ,
       -2.14663887, -1.00195539,  0.14704289, -3.09080958,  1.29497552,
        0.93130547,  0.13077104,  0.40514302,  2.33377981,  0.70178449,
       -2.14083028,  3.24479437, -2.67292738, -1.3273046 ,  1.43655634,
        0.37366623, -2.45767331, -0.45977798,  2.97876358,  0.52454937,
        1.56182909,  2.47367239, -0.38051525, -0.82510483, -2.41618276,
       -0.47854578,  0.75745118,  2.65333366,  1.4364686 , -0.80404937,
       -0.4776988 , -0.08876944,  0.52805376,  0.58157122,  0.77731681,
       -0.95700943,  0.88351893,  0.43560213, -0.32796392, -1.64268029,
       -1.39666915,  0.01206213, -0.66244853, -4.08392334,  0.27594233,
        0.0979808 ,  1.51392853, -0.14122066, -0.83347476])


def normalise(x):
    return x / np.linalg.norm(x)


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
        end_posn = (self.qkv.shape[0] == len(prompt_tokens))
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
            q96_dotprod_baseline = q96_head.dot(q96) / np.linalg.norm(q96_head) / np.linalg.norm(q96)
            q99_dotprod_baseline = q99_head.dot(q99) / np.linalg.norm(q99_head) / np.linalg.norm(q99)
            if layer < 9:
                print('q96_dotprod_baseline', q96_dotprod_baseline)
                print('q99_dotprod_baseline', q99_dotprod_baseline)
        attn = np.zeros(n_embd, dtype=x.dtype)
        for i in range(n_head):
            q = self.qkv[-1, D * i : D * (i + 1)]
            k = self.qkv[:, D * (n_head + i) : D * (n_head + i + 1)]
            v = self.qkv[:, D * (2 * n_head + i) : D * (2 * n_head + i + 1)]
            # if end_posn and layer == 9 and i == 6: print(repr(q))
            A = np.exp(q @ k.T / np.sqrt(D))
            A /= np.sum(A)
            A[A < 0.04] = 0
            A /= np.sum(A)
            if end_posn:
                print(f"{layer}.{i}", end=": ", flush=True)
                for token, nonzero in zip(prompt_tokens[1:], A[1:] > 0):
                    if nonzero:
                        print(encoder.decode([token]), end="", flush=True, color='red')
                    else:
                        print(encoder.decode([token]), end="", flush=True)
                # print()
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
                q96_dotprod = q96_head.dot(q96) / np.linalg.norm(q96_head) / np.linalg.norm(q96)
                q96_dotprod -= q96_dotprod_baseline
                q99_dotprod = q99_head.dot(q99) / np.linalg.norm(q99_head) / np.linalg.norm(q99)
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
        if end_posn:
            print('---')
        return x


def main():
    global blocks
    tokens = prompt_tokens
    # tokens = [50256]
    # print(tokens)
    # print(encoder.decode([tokens[0]]), end="", flush=True)
    total = len(tokens) + n_tokens_to_generate
    assert total < n_ctx
    blocks = [TransformerBlock(b) for b in range(n_layer)]
    for posn in range(total):
        token = tokens[posn]
        x = wte[token] + wpe[posn]
        for layer, block in enumerate(blocks):
            if posn + 1 >= len(tokens):
                print('layer', layer, end=" ", flush=True)
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
            tokens.append(token)
            for token, prob in zip(top_k, top_k_probs):
                print(encoder.decode([token]), prob, end="; ", flush=True)
            break
        # print(encoder.decode([tokens[posn + 1]]), end="", flush=True)
    # print(tokens)
    print()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
