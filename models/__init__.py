#!/usr/bin/env python3
# coding: utf-8

from .gru import GRU

model_map = {
    'gru' : GRU
}

model_list = list(model_map.keys())


def create_model(model, **kwargs):
    """
    Factory function for creating a new model instance.
    """
    model = model.lower()

    if model not in model_map:
        raise ValueError(f"invalid model '{model}'")
        
    return model_map[model](**kwargs)