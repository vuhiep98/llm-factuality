import yaml
import torch
import re
import json
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from yaml import Loader
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

logging.set_verbosity_error()
root_dir = Path(__file__).parents[1]

class FactReasoner:
    def __init__(self, model_path):
        self.model_path = model_path
        self.prompts = {}
        self._load_prompts()
        self._load_model()
    
    def _load_prompts(self):
        self.prompts["factuality_reasoning"] = yaml.load(open(root_dir/"prompts/factuality_reasoning_prompt.yaml"), Loader=Loader)["user"]
    
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
    def _read_input(self, input_file):
        inputs = []
        with open(input_file, 'r') as f:
            for line in f:
                inputs.append(json.loads(line.strip()))
        return inputs
        
    def _build_fact_reasoning_message(self, llm_output):
        message = [
            {
                "role": "user", 
                "content": self.prompts["factuality_reasoning"].format(llm_output=llm_output)
            }
        ]
        return message
    
    def _reasoning(self, message):
        for _ in range(5):
            try:
                with torch.no_grad():
                    inputs = self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=True, return_tensors="pt")
                    inputs = inputs.to(self.model.device)
                    outputs = self.model.generate(inputs, max_new_tokens=8192)
                    response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
                atomic_facts = self._extract_predicted_facts(response)
                break
            except Exception as e:
                atomic_facts = []
                print(str(e) + "\nRetrying...")
        return atomic_facts, response
    
    def _extract_predicted_facts(self, response):
        predicted_facts = re.findall(r"```OUTPUT\n([\s\S]*)```", response)[0]
        predicted_facts = json.loads(predicted_facts)
        return predicted_facts
    
    def _calculate_factuality(self, predicted_facts):
        if len(predicted_facts) == 0:
            return 0.0
        else:
            return len([fact for fact in predicted_facts if fact["is_supported"] == True])/len(predicted_facts)
    
    def _fact_reasoning_output(self, llm_output):
        message = self._build_fact_reasoning_message(llm_output)
        predicted_facts, response = self._reasoning(message)
        factscore = self._calculate_factuality(predicted_facts)
        return factscore, predicted_facts, response
    
    def factual_reasoning(self, input_file, verbosity=True, save_output=True):
        inputs = self._read_input(input_file)
        if verbosity:
            inputs = tqdm(inputs)
            
        predicted_facts = []
        scores = []
        responses = []
        for input in inputs:
            score, facts, full_response = self._fact_reasoning_output(input["output"])
            scores.append(score)
            predicted_facts.append({"topic": input["topic"], "atomic-facts": facts, "full-response": full_response})
        
        factscore = np.mean(scores)
        
        with open(root_dir/"outputs/factscore_reasoning"/input_file.split("/")[-1].replace(".jsonl", ".json"), "w") as f:
            json.dump({"score": factscore, "factscore-output": predicted_facts}, f, indent=2)
        
        return factscore
        