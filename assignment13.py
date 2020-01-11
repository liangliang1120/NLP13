# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:50:40 2020

@author: guliang
"""

import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from torch.utils.data import DataLoader,Dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
weights = torch.load('E:/my_path/NLP13/homework/small_ft.pkl')
medium_config = GPT2Config(n_embd = 768,n_layer = 12, n_head = 12)
model = GPT2LMHeadModel(medium_config)

weights['lm_head.weight'] = weights['lm_head.decoder.weight']
weights.pop('lm_head.decoder.weight',None)

model.load_state_dict(weights)
model.train()

tokenizer.encode('i like playing basketball')
tokenizer.decode([72,588,2712,9669])



text = 'Does money buy happiness ?'

predicted_text = text


# 每一个只能补一个token出来，补一句话需要多次，3次是我拍脑袋的
for i in range(0,3):
    # 以上次预测结果作为本次的输入，所谓的自回归
    indexed_tokens = tokenizer.encode(predicted_text)
    # 将读出的索引标记转化成PyTorch向量
    tokens_tensor = torch.tensor([indexed_tokens])
    # 进行推理
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]
        # 获取预测的下一个子词
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        # 解码成我们都读懂的文本
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        # 打印输入结果
        print(predicted_text)










