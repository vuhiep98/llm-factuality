import json
import os
from tqdm import tqdm
evaluator = "Gemini-Pro"
lang = "es"
model = "gemini"
for lang in ["es", "ar", "bn"]:
    for model in ["gemini", "gpt4"]:
        print(lang, model)
        # instance_path = ~/FActScore/data/to_evaluate/ar/sub_task_1/gemini.jsonl
        # label_path = ~/FActScore/data/to_evaluate/ar/sub_task_1/geminiar_ChatGPT_factscore_output_provided_facts.json
        instance_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1_filter/{model}.jsonl"
        # instance_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1/{model}.jsonl"
        label_path_en = f""
        #"~/FActScore/data/to_evaluate/es/sub_task_1_filter/geminies_retrieval+Gemini-Pro_factscore_output_provided_facts_gen_ks.json"
        #"~/FActScore/data/to_evaluate/es/sub_task_1_filter/geminies_retrieval+Gemini-Pro_factscore_output_provided_facts_trans_map.json"
        label_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1_filter/{model}{lang}_retrieval+{evaluator}_factscore_output_provided_facts_trans_map.json"
        # label_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1/{model}{lang}_{evaluator}_factscore_output_provided_facts.json"
        # label_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1/{model}{lang}_{evaluator}_factscore_output_provided_facts.json"
        
        
        
        with open(label_path) as f:
            for line in f:
                dp = json.loads(line)

        decisions = dp["decisions"]
        
        # with open(label_path_en) as f:
        #     for line in f:
        #         dp_en = json.loads(line)

        # decisions_en = dp_en["decisions"]
        save_dict = {}
        with open(instance_path) as f:
            for i, line in tqdm(enumerate(f)):
                dp = json.loads(line)
                label = decisions[i]
                for i,e in enumerate(label):
                    # save_dict["#".join([dp["topic"], e["atom"]])]= decisions_en[i]["is_supported"]
                    save_dict["#".join([dp["topic"], e["atom"]])]= e["is_supported"]
        # print(save_dict)

        save_path = f"~/FActScore/data/to_evaluate/label/{lang}_{model}_label_by_{evaluator}_trans_after_retrieval.json"
        with open(save_path, 'w') as jsonl_file:
            
            json_line = json.dumps(save_dict, ensure_ascii=False)
            jsonl_file.write(json_line + '\n', )
        print(save_path)