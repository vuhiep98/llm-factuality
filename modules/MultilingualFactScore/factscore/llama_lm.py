# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


from factscore.lm import LM

class LlamaLM(LM):
    def __init__(self, model_name, quantization=True, cache_file=None, return_score=True):
        self.model_name = model_name
        self.quantization = quantization
        self.return_score = return_score
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", 
            torch_dtype="auto", 
            device_map="auto", 
            # attn_implementation="flash_attention_2"
        )

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128):
        message = [{"role": "user", "content": prompts}]
        with torch.no_grad():
            inputs = self.tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                return_tensors="pt", 
                max_length=max_sequence_length
            )
            inputs = inputs.to(self.model.device)
            response = self.model.generate(inputs, max_new_tokens=max_output_length)
            output = self.tokenizer.decode(response[0][len(inputs[0]):], skip_special_tokens=True)
        return output, response[0]

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    lm = LlamaLM(model_name=model_name, quantization=False)
    lm.load_model()
    
    prompts = "What is the capital of France?"
    output, response = lm._generate(prompts)
    
    print("Output:", output)
    print("Response:", response)

