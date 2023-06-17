from collections import namedtuple
import json
import jax.numpy as np
import safetensors.flax

TransformerBlock = namedtuple(
    "TransformerBlock",
    [
        "layer",
        "b_qkv",
        "w_qkv",
        "b_out",
        "w_out",
        "b_mlp1",
        "w_mlp1",
        "b_mlp2",
        "w_mlp2",
    ],
)

TransformerModel = namedtuple(
    "TransformerModel", ["wte", "wpe", "w_ln", "b_ln", "w_unembed", "blocks"]
)


def load_gpt2():
    model = safetensors.flax.load_file("models/gpt2/model.safetensors")
    config = json.loads(open("models/gpt2/config.json").read())

    wte = model["wte.weight"]
    wpe = model["wpe.weight"]
    w_ln = model["ln_f.weight"] * np.sqrt(config["n_embd"])
    b_ln = model["ln_f.bias"]

    # centering
    w_unembed = wte.T.copy()
    w_unembed -= w_unembed.mean(axis=-1, keepdims=True)
    wte -= wte.mean(axis=-1, keepdims=True)
    wpe -= wpe.mean(axis=-1, keepdims=True)

    blocks = []
    for layer in range(config["n_layer"]):
        b_qkv = model[f"h.{layer}.attn.c_attn.bias"]
        b_qkv += model[f"h.{layer}.ln_1.bias"] @ model[f"h.{layer}.attn.c_attn.weight"]
        w_qkv = model[f"h.{layer}.attn.c_attn.weight"]
        w_qkv *= model[f"h.{layer}.ln_1.weight"][:, None] * np.sqrt(config["n_embd"])
        b_out = model[f"h.{layer}.attn.c_proj.bias"]
        b_out -= b_out.mean(keepdims=True)
        w_out = model[f"h.{layer}.attn.c_proj.weight"]
        w_out -= w_out.mean(axis=-1, keepdims=True)
        b_mlp1 = model[f"h.{layer}.mlp.c_fc.bias"]
        b_mlp1 += model[f"h.{layer}.ln_2.bias"] @ model[f"h.{layer}.mlp.c_fc.weight"]
        w_mlp1 = model[f"h.{layer}.mlp.c_fc.weight"]
        w_mlp1 *= model[f"h.{layer}.ln_2.weight"][:, None] * np.sqrt(config["n_embd"])
        b_mlp2 = model[f"h.{layer}.mlp.c_proj.bias"]
        b_mlp2 -= b_mlp2.mean(keepdims=True)
        w_mlp2 = model[f"h.{layer}.mlp.c_proj.weight"]
        w_mlp2 -= w_mlp2.mean(axis=-1, keepdims=True)
        blocks.append(
            TransformerBlock(
                layer, b_qkv, w_qkv, b_out, w_out, b_mlp1, w_mlp1, b_mlp2, w_mlp2
            )
        )

    return TransformerModel(wte, wpe, w_ln, b_ln, w_unembed, blocks)
