import json
import re
import argparse
import os
import yaml
import ast
from tqdm.auto import tqdm
from openai import AzureOpenAI

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
max_tokens = 2048
temperature = 0.0
model = "gpt-4o-2024-05-13"

def read_jsonl(file):
    with open(file, "r") as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    return data

def build_messages(sentence, prediction, ground_truth):
    
    prompt = yaml.load(open(f"{rootdir}/prompts/estimator_prompt.yaml"), Loader=yaml.FullLoader)
    
    messages = [
        {"role": "system", "content": prompt["system"]},
        {
            "role": "user",
            "content": prompt["user"].format(sentence=sentence, 
                                             ground_truth=ground_truth,
                                             prediction=prediction)
        }
    ]
    return messages

def get_response(client, messages):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content

def estimate(input):
    estimate_output = []
    if "annotations" in input and input["annotations"] is not None:
        for sentence in input["annotations"]:
            if "human-atomic-facts" in sentence:
                sent = sentence["text"]
                ground_truth = ""
                for fact in sentence["human-atomic-facts"]:
                    ground_truth += f"Fact: {fact['text']}\nLabel: {fact['label']}\n"
                
                pred_sentence = [p_sentence for p_sentence in input["factscore_reasoning"]["factscores"] if p_sentence["sentence"] == sent]
                
                if len(pred_sentence) > 0:
                    pred_sentence = pred_sentence[0]
                else:
                    continue
                
                prediction = ""
                for fact in pred_sentence["atomic_facts"]:
                    prediction += f"Fact: {fact['atom']}\nLabel: {fact['is_supported']}\n"
                
                messages = build_messages(sent, prediction, ground_truth)
                response = get_response(client, messages)
                try:
                    response = re.findall(r"```\n(\[([\s\S])*\])\n```", response)[0][0]
                    estimating_result = ast.literal_eval(response)
                except:
                    estimating_result = response
                    print("Error parsing response")
                estimate_output.append({
                    "sentence": sent,
                    "estimate": estimating_result
                })
    return estimate_output

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjective_llm", type=str, required=True)
    args = parser.parse_args()
    
    subjective_llm = args.subjective_llm
    
    # Initialize OpenAI client
    with open("configs.json") as f:
        configs = json.load(f)

    client = AzureOpenAI(
        azure_endpoint = configs["azure_endpoint"], 
        api_key=configs["openai"],  
        api_version=configs["api_version"],
    )

    subjective_llm_file = f"{rootdir}/outputs/factscore_reasoning/{subjective_llm}_factscore_reasoning.jsonl"
    
    outputs = read_jsonl(subjective_llm_file)
    
    subjective_llm_estimate = []
    
    num_pred = 0
    num_extr = 0
    total = 0
    for output in tqdm(outputs[:10]):
        etm = estimate(output)
        if len(etm) > 0:
            for sent in etm:
                if isinstance(sent["estimate"], list):
                    for e in sent["estimate"]:
                        if e["extracted"] == "true":
                            num_extr += 1
                        if e["predicted"] == "true":
                            num_pred += 1
                        total += 1
        
        subjective_llm_estimate.append({
            "topic": output["topic"],
            "estimate": estimate(output)
        })
    
    with open(f"{rootdir}/outputs/estimate/{subjective_llm}_estimate.jsonl", "w") as f:
        for output in subjective_llm_estimate:
            f.write(json.dumps(output) + "\n")
    
    print(f"Accuracy: {num_pred/num_extr*100:.2f}% ({num_pred}/{num_extr})")
    print(f"Coverage: {num_extr/total*100:.2f}% ({num_extr}/{total})")
        