from openai import OpenAI
import os
import logging
import json
import yaml
import re
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    
    data_folder = "/home/s2420414/fact-check/adobe/modules/FActScore/data/labeled"
    llm_output_files = [file for file in os.listdir(data_folder) if file.endswith(".jsonl")]
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://spcc-a100g07:8000/v1"
    )
    
    # Factuality Reasoning
    for llm_output_file in llm_output_files:
        logging.info(f"Esimating for {llm_output_file.split('.')[0]}")
        
        with open(os.path.join(data_folder, llm_output_file)) as f:
            inputs = [json.loads(line) for line in f.readlines()]
        
        logging.info(f"Number of inputs: {len(inputs)}")
        
        factscores = []
        for input in tqdm(inputs):
            reasoning_prompt = yaml.load(open("/home/s2420414/fact-check/adobe/llm-factuality/prompts/factuality_reasoning_prompt.yaml"), Loader=yaml.FullLoader)
            
            messages = [
                {"role": "system", "content": reasoning_prompt["system"]},
                {"role": "user", "content": reasoning_prompt["user"].format(llm_output=input["output"])}
            ]
            
            response = client.chat.completions.create(
                messages=messages,
                model="meta-llama/Llama-3.1-8B-Instruct",
                temperature=0,
                max_tokens=8192,
                top_p=0.000001
            )
            
            try:
                factscore_json = re.findall(r"```json\n([\s\S]*)\n```", response.choices[0].message.content)[0]
                factscore_json = json.loads(factscore_json)
            except:
                factscore_json = response.choices[0].message.content
            
            input["factscore_reasoning"] = factscore_json
        
        factscore_output_file = llm_output_file.replace(".jsonl", "_factscore_reasoning.jsonl")
        logging.info(f"Fininsh Reasoning\nWriting to {factscore_output_file}")
        with open(f"/home/s2420414/fact-check/adobe/llm-factuality/outputs/factscore_reasoning/{factscore_output_file}", "w") as f:
            for input in inputs:
                f.write(json.dumps(input) + "\n")