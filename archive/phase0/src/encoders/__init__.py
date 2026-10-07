"""Frozen encoders behind one interface (`src.encoders.base.Encoder`). Registry key = `encoder` in configs.

Each encoder keeps its own pretraining preprocessing (OPERA: 64-mel, 16 kHz; AST: 128-mel fbank; CLAP: 48 kHz,
10 s; HeAR: 2 s PCEN clips), so frame embeddings differ in D and layer count but not in shape (32, D).
Imports are lazy: loading one encoder does not import the others' dependencies.
"""
import importlib

_REGISTRY = {"opera_ct": "opera.OperaCT", "opera_ce": "opera.OperaCE", "ast_audioset": "vit.AstViT",
             "hear": "vit.HeARViT", "clap_htsat": "clap.ClapHtsat"}
ENCODERS = tuple(_REGISTRY)


def get(name):
    """Encoder class `name` (one of `ENCODERS`), without loading weights."""
    module, cls = _REGISTRY[name].split(".")
    return getattr(importlib.import_module(f"src.encoders.{module}"), cls)


def load(name):
    """Instantiate the frozen encoder `name`."""
    return get(name)()
