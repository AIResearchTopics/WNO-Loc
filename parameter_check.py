# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 01:07:27 2026

@author: anjum
"""
import torch

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Load your models and count
from model import LocalizationModel
from baselines import BiLSTMBaseline, FNOBaseline, TransformerBaseline

wno = LocalizationModel(in_features=3, use_real_data=False, 
                         wno_width=32, wno_levels=3, wno_layers=2,
                         signal_length=100, wavelet='db4')

bilstm = BiLSTMBaseline(...)
fno = FNOBaseline(...)
transformer = TransformerBaseline(...)

for name, model in [('WNO', wno), ('BiLSTM', bilstm), 
                    ('FNO', fno), ('Transformer', transformer)]:
    total, trainable = count_parameters(model)
    print(f"{name}: {total:,} parameters ({total/1e3:.1f}K)")