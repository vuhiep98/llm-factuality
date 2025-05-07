import json
import re
import copy
from ast import literal_eval
from openai import AzureOpenAI

with open("../configs/configs.json") as f:
    configs = json.load(f)

client = AzureOpenAI(
  azure_endpoint = configs["azure_endpoint"], 
  api_key=configs["openai"],  
  api_version="2024-10-21"
)

def read_json_file(file):
    data = []
    with open(file) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def format_facts(facts):
    formatted_facts = [f"Fact: {fact['text']}\nLabel: {fact['label']}" for fact in facts]
    return "\n".join(formatted_facts)

def create_messages(sentence):
    evaluate_prompt = copy.deepcopy(estimate_prompt)
    evaluate_prompt[-1]["content"] = evaluate_prompt[-1]["content"].replace("[SENTENCE]", sentence["sentence"]).replace("[GROUND-TRUTH]", format_facts(sentence["ground_truth"])).replace("[PREDICTION]", format_facts(sentence["predicted"]["atomic_facts"]))

    return evaluate_prompt
    
def send_messages(messages):
    response = client.chat.completions.create(
        model="gpt-4o-2024-05-13",
        messages=messages,
        max_tokens=2048,
        temperature=0.,
    )
    return response.choices[0].message.content

true_false_dict = {
    "true": True,
    "false": False
}

def extract_annotated_results(response):
    matches = re.findall(r"\n\[[\s\S]*\]\n", response)
    facts_str = matches[0]
    facts = literal_eval(facts_str)
    for fact in facts:
        fact["extracted"] = fact["extracted"] if fact["extracted"] in [True, False] else true_false_dict[fact["extracted"]]
        fact["predicted"] = fact["predicted"] if fact["predicted"] in [True, False] else true_false_dict[fact["predicted"]]
    return facts