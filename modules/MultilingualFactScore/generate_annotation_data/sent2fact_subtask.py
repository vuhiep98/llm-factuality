import json
import os
from collections import defaultdict 

no = "2"
lang = "es"

lst_paths = [f"~/FActScore/data/to_evaluate/{lang}/subtask_extract_{no}{lang}_just-extract-by-ChatGPT_factscore_output_gen_facts.json", f"~/FActScore/data/to_evaluate/{lang}/subtask_extract_{no}{lang}_just-extract-by-GPT-4_factscore_output_gen_facts.json"]
sent2facts = defaultdict(list)
for label_path in lst_paths:
    with open(label_path) as f:
        for line in f:
            dp = json.loads(line)
        lst_sent2fact = dp["sent2facts"]
        for inst in lst_sent2fact:
            for sent in inst:
                if "###" in sent[0] or "**" in sent[0] or ":।" in sent[0]:
                    continue
                sent2facts[sent[0]].append(sent[1])
save_path = f"~/FActScore/data/to_annotate_data/{lang}/task/sub_task_2/sent2fact_{no}.json"
with open(save_path, 'w') as jsonl_file:
    json_line = json.dumps(sent2facts, ensure_ascii=False)
    jsonl_file.write(json_line + '\n', )
print(save_path)
