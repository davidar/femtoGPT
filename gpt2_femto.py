#!/usr/bin/env python3

from collections import namedtuple
import json
import jax
import jax.numpy as np
from jax.scipy.special import erf
import safetensors.numpy
import einops

from print_color import print

from encoder import get_encoder

# from huggingface_hub import hf_hub_download
# hf_hub_download("gpt2", "config.json", local_dir="models/gpt2")
# hf_hub_download("gpt2", "model.safetensors", local_dir="models/gpt2")

# jax.config.update("jax_log_compiles", True)

jax.experimental.compilation_cache.compilation_cache.initialize_cache("jax_cache")

encoder = get_encoder("", "")
hparams = json.loads(open("models/gpt2/config.json").read())

n_vocab = hparams["vocab_size"]
n_ctx = hparams["n_ctx"]
n_embd = hparams["n_embd"]
n_head = hparams["n_head"]
n_layer = hparams["n_layer"]
D = int(n_embd / n_head)

params = safetensors.numpy.load_file("models/gpt2/model.safetensors")
for k, v in params.items():
    params[k] = np.array(v)

wte = params["wte.weight"]
wpe = params["wpe.weight"]
w_ln = params["ln_f.weight"] * np.sqrt(n_embd)
b_ln = params["ln_f.bias"]

# centering
w_unembed = wte.T.copy()
w_unembed -= w_unembed.mean(axis=-1, keepdims=True)
wte -= wte.mean(axis=-1, keepdims=True)
wpe -= wpe.mean(axis=-1, keepdims=True)

force_enable_layers = 0

prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

prompt_tokens = [np.array([50256] + encoder.encode(s), dtype=np.int32) for s in prompts]

n_seq = 15
causal_mask = np.tri(n_seq, dtype=np.float32)

for toks in prompt_tokens:
    assert toks.size == n_seq


def normalise_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def print_top_k(probs, k):
    top_k = np.argsort(probs)[-1 : -k - 1 : -1]
    top_k_probs = probs[top_k]
    top_k_probs /= np.sum(top_k_probs)

    for token, prob in zip(top_k, top_k_probs):
        print(
            f"'{encoder.decode([int(token)])}' {100 * prob:.1f}%",
            end="; ",
            flush=True,
        )
    print()


n_batch = 4


TransformerBlock = namedtuple("TransformerBlock", ["layer", "b_qkv", "w_qkv", "b_out", "w_out", "b_mlp1", "w_mlp1", "b_mlp2", "w_mlp2"])

def new_block(layer):
    b_qkv = params[f"h.{layer}.attn.c_attn.bias"]
    b_qkv += params[f"h.{layer}.ln_1.bias"] @ params[f"h.{layer}.attn.c_attn.weight"]
    w_qkv = params[f"h.{layer}.attn.c_attn.weight"]
    w_qkv *= params[f"h.{layer}.ln_1.weight"][:, None] * np.sqrt(n_embd)
    b_out = params[f"h.{layer}.attn.c_proj.bias"]
    b_out -= b_out.mean(keepdims=True)
    w_out = params[f"h.{layer}.attn.c_proj.weight"]
    w_out -= w_out.mean(axis=-1, keepdims=True)
    b_mlp1 = params[f"h.{layer}.mlp.c_fc.bias"]
    b_mlp1 += params[f"h.{layer}.ln_2.bias"] @ params[f"h.{layer}.mlp.c_fc.weight"]
    w_mlp1 = params[f"h.{layer}.mlp.c_fc.weight"]
    w_mlp1 *= params[f"h.{layer}.ln_2.weight"][:, None] * np.sqrt(n_embd)
    b_mlp2 = params[f"h.{layer}.mlp.c_proj.bias"]
    b_mlp2 -= b_mlp2.mean(keepdims=True)
    w_mlp2 = params[f"h.{layer}.mlp.c_proj.weight"]
    w_mlp2 -= w_mlp2.mean(axis=-1, keepdims=True)
    return TransformerBlock(layer, b_qkv, w_qkv, b_out, w_out, b_mlp1, w_mlp1, b_mlp2, w_mlp2)

def attention(posn, q, k):
    A = np.exp(q @ k.T / np.sqrt(D)) * causal_mask[posn]
    A /= np.sum(A, axis=-1, keepdims=True)
    return A

@jax.jit
def call_block(b, x, head_activations, q_batch, k_batch, v_batch):
    qkv = normalise_rows(x) @ b.w_qkv + b.b_qkv
    Q, K, V = np.split(qkv, 3, axis=-1)
    sig = "batch posn (head D) -> head batch posn D"
    qs = einops.rearrange(Q, sig, batch=n_batch, posn=n_seq, head=n_head, D=D)
    ks = einops.rearrange(K, sig, batch=n_batch, posn=n_seq, head=n_head, D=D)
    vs = einops.rearrange(V, sig, batch=n_batch, posn=n_seq, head=n_head, D=D)
    attn = einops.rearrange(
        jax.vmap(
            lambda batch: jax.vmap(
                lambda posn: jax.vmap(
                    lambda head: attention(
                        posn,
                        qs[head, q_batch[batch, b.layer, head, posn], posn],
                        ks[head, k_batch[batch, b.layer, head, posn]],
                    )
                    @ vs[head, v_batch[batch, b.layer, head, posn]]
                )(np.arange(n_head))
            )(np.arange(n_seq))
        )(np.arange(n_batch)),
        "batch posn head D -> batch posn (head D)",
        batch=n_batch,
        posn=n_seq,
        head=n_head,
        D=D,
    )
    attn *= np.repeat(head_activations[:, :, b.layer, :], D, axis=-1)
    x += attn @ b.w_out + b.b_out

    h = normalise_rows(x) @ b.w_mlp1 + b.b_mlp1
    # h *= scipy.stats.norm.cdf(h)  # gelu
    h *= (1 + erf(h / np.sqrt(2))) / 2
    x += h @ b.w_mlp2 + b.b_mlp2

    return x


blocks = [new_block(b) for b in range(n_layer)]

# print(jax.make_jaxpr(lambda x: blocks[0](x, np.ones((n_batch, n_seq, n_layer, n_head)), 0))(np.stack([wpe[:n_seq]] * n_batch)))

warmup = True


def gpt2(which_prompt, head_activations, q_batch, k_batch, v_batch, output_batch):
    x = wte[prompt_tokens[which_prompt]] + wpe[:n_seq]
    x = np.stack([x] * n_batch)
    for block in blocks:
        x = call_block(block, x, head_activations, q_batch, k_batch, v_batch)
        # assert x.mean() < 1e-5
    final = normalise_rows(x[output_batch]) * w_ln + b_ln
    logits = final @ w_unembed
    return logits


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


@jax.grad
def grad_logit_diff(head_activations, q_batch, k_batch, v_batch, output_batch):
    sum = 0
    for p in range(len(prompts)):
        logits = gpt2(p, head_activations, q_batch, k_batch, v_batch, output_batch)
        logits = logits[-1]
        correct, incorrect = answers[p]
        sum += (
            logits[int(encoder.encode(correct)[0])]
            - logits[int(encoder.encode(incorrect)[0])]
        )
    return sum / len(prompts)


if __name__ == "__main__":
    head_activations = np.ones((n_batch, n_seq, n_layer, n_head), dtype=np.float32)
    q_batch = np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32)
    k_batch = np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32)
    v_batch = np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32)

    print("Name Mover heads", format="bold")

    sensitivity = grad_logit_diff(
        head_activations, q_batch, k_batch, v_batch, 1
    ).__array__()

    for posn in range(1, n_seq):
        print(encoder.decode([int(prompt_tokens[0][posn])]), end=" ")
        absmax = max(np.abs(sensitivity[1, posn, :, :]).max(), 0.5)
        for layer in range(n_layer):
            for head in range(n_head):
                s = np.abs(sensitivity[1, posn, layer, head]) / absmax
                if s > 0.1:
                    print(
                        f"{layer}.{head} -- {sensitivity[1, posn, layer, head]:.2f}",
                        # end=" ",
                        colour="green" if s > 0.5 else "yellow" if s > 0.25 else "red",
                    )
                if s > 0.25 and sensitivity[1, posn, layer, head] > 0:
                    q_batch = q_batch.at[2, layer, head, posn].set(1)
        print()

    print("S-Inhibition heads", format="bold")

    # warmup = False
    sensitivity = grad_logit_diff(
        head_activations, q_batch, k_batch, v_batch, 2
    ).__array__()
    q_batch = np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32)

    for posn in range(1, n_seq):
        print(encoder.decode([int(prompt_tokens[0][posn])]), end=" ")
        absmax = max(np.abs(sensitivity[1, posn, :, :]).max(), 0.2)
        for layer in range(n_layer):
            for head in range(n_head):
                s = np.abs(sensitivity[1, posn, layer, head]) / absmax
                if s > 0.1:
                    print(
                        f"{layer}.{head} -- {sensitivity[1, posn, layer, head]:.2f}",
                        # end=" ",
                        colour="green" if s > 0.5 else "yellow" if s > 0.25 else "red",
                    )
                if s > 0.25:
                    v_batch = v_batch.at[0, layer, head, posn].set(1)
        print()

    print("Induction heads", format="bold")

    sensitivity = grad_logit_diff(
        head_activations, q_batch, k_batch, v_batch, 0
    ).__array__()
    v_batch = np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32)

    for posn in range(1, n_seq):
        print(encoder.decode([int(prompt_tokens[0][posn])]), end=" ")
        absmax = max(np.abs(sensitivity[1, posn, :, :]).max(), 0.2)
        for layer in range(n_layer):
            for head in range(n_head):
                s = np.abs(sensitivity[1, posn, layer, head]) / absmax
                if s > 0.1:
                    print(
                        f"{layer}.{head} -- {sensitivity[1, posn, layer, head]:.2f}",
                        # end=" ",
                        colour="green" if s > 0.5 else "yellow" if s > 0.25 else "red",
                    )
                if s > 0.25:
                    q_batch = q_batch.at[0, layer, head, posn].set(1)
                    k_batch = k_batch.at[0, layer, head, posn].set(1)
        print()

    print("Duplicate token heads", format="bold")

    sensitivity = grad_logit_diff(
        head_activations,
        q_batch,
        np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32),
        v_batch,
        0,
    ).__array__()
    q_batch = np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32)

    for posn in range(1, n_seq):
        print(encoder.decode([int(prompt_tokens[0][posn])]), end=" ")
        absmax = max(np.abs(sensitivity[1, posn, :, :]).max(), 0.2)
        for layer in range(n_layer):
            for head in range(n_head):
                s = np.abs(sensitivity[1, posn, layer, head]) / absmax
                if s > 0.1:
                    print(
                        f"{layer}.{head} -- {sensitivity[1, posn, layer, head]:.2f}",
                        # end=" ",
                        colour="green" if s > 0.5 else "yellow" if s > 0.25 else "red",
                    )
                # if s > 0.25:
                #     q_batch = q_batch.at[0, layer, head, posn].set(1)
        print()

    print("Previous token heads", format="bold")

    sensitivity = grad_logit_diff(
        head_activations, q_batch, k_batch, v_batch, 0
    ).__array__()
    k_batch = np.zeros((n_batch, n_layer, n_head, n_seq), dtype=np.int32)

    for posn in range(1, n_seq):
        print(encoder.decode([int(prompt_tokens[0][posn])]), end=" ")
        absmax = max(np.abs(sensitivity[1, posn, :, :]).max(), 0.2)
        for layer in range(n_layer):
            for head in range(n_head):
                s = np.abs(sensitivity[1, posn, layer, head]) / absmax
                if s > 0.1:
                    print(
                        f"{layer}.{head} -- {sensitivity[1, posn, layer, head]:.2f}",
                        # end=" ",
                        colour="green" if s > 0.5 else "yellow" if s > 0.25 else "red",
                    )
                # if s > 0.25:
                #     q_batch = q_batch.at[0, layer, head, posn].set(1)
        print()
