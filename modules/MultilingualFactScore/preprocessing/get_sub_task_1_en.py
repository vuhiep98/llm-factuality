import json
import os
from tqdm import tqdm
evaluated_model = "gemini"
lang = "ar"
for evaluated_model in ["gemini", "gpt4"]:
    for lang in ["ar", "bn", "es"]:
        en_path = f"~/FActScore/data/to_evaluate/{lang}/en_instances/{evaluated_model}.jsonl"
        source_sub_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1/{evaluated_model}.jsonl"
        en_sub_path = f"~/FActScore/data/to_evaluate/{lang}/en_instances/sub_task_1/{evaluated_model}.jsonl"
        query_lst = []
        save_en_lst = []
        with open(source_sub_path) as f:
            for i, line in tqdm(enumerate(f)):
                dp = json.loads(line)
                query_lst.append(dp["input"])
        lst_all_en = []
        with open(en_path) as f:
            for i, line in tqdm(enumerate(f)):
                dp = json.loads(line)
                lst_all_en.append(dp.copy())
        for input in query_lst:
            for dp in lst_all_en:
                if dp["input"] == input:
                    save_en_lst.append(dp)
                    break          
        with open(en_sub_path, 'w') as jsonl_file:
            for dictionary in save_en_lst:
                json_line = json.dumps(dictionary, ensure_ascii=False)
                jsonl_file.write(json_line + '\n', )
        print("Save path:", en_sub_path)