import json
import os
import time
from tqdm import tqdm
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
for lang in ["ar", "bn", "es"]:
    path = f"~/projects/factscore/real_evaluation/en_{lang}wiki_100.jsonl"
    lst_save = []
    with open(path) as f:
        for i, line in tqdm(enumerate(f)):
            dp = json.loads(line)
            to_save = {}
            to_save = {"input": "", "output": dp["text"].replace(SPECIAL_SEPARATOR, "").replace("<s>", "").replace("</s>", ""), "topic": dp["en_title"]}
            lst_save.append(to_save.copy())
    save_path = f"~/FActScore/data/db_to_extract/en_{lang}wiki_100.jsonl"
    with open(save_path, 'w') as jsonl_file:
        for dictionary in lst_save:
            json_line = json.dumps(dictionary, ensure_ascii=False)
            jsonl_file.write(json_line + '\n', )

    
