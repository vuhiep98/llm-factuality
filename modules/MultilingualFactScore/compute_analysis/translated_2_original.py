import json
import os
from tqdm import tqdm
evaluator = "GPT-4"
lang = "es"
model = "gemini"
for lang in ["es", "ar", "bn"]:
    for model in ["gemini", "gpt4"]:
        print(lang, model)
        instance_path = f"~/FActScore/data/to_evaluate/{lang}/{model}.jsonl"
        instance_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1/{model}.jsonl"
        instance_path_en = f"~/FActScore/data/to_evaluate/{lang}/en_instances/sub_task_1/{model}.jsonl"
        # instance_path = f"~/FActScore/data/to_evaluate/{lang}/sub_gpt4/gpt4.jsonl"
        label_path_en = f"~/FActScore/data/to_evaluate/{lang}/en_instances/{model}en_retrieval+{evaluator}_factscore_output_provided_facts.json"
        label_path_en = f"~/FActScore/data/to_evaluate/{lang}/en_instances/sub_task_1/{model}en_retrieval+{evaluator}_factscore_output_provided_facts.json"
        # label_path_en = f"~/FActScore/data/to_evaluate/{lang}/sub_gpt4/en_instances/{model}en_retrieval+{evaluator}+npm_factscore_output_provided_facts.json"
        print(label_path_en)
        label_path = f"~/FActScore/data/to_evaluate/{lang}/{model}{lang}_retrieval+{evaluator}_factscore_output_provided_facts.json"
        label_path = f"~/FActScore/data/to_evaluate/{lang}/sub_task_1/{model}{lang}_retrieval+{evaluator}_factscore_output_provided_facts.json"
        # label_path = f"~/FActScore/data/to_evaluate/{lang}/sub_gpt4/{model}{lang}_retrieval+{evaluator}_factscore_output_provided_facts.json"
        with open(label_path) as f:
            for line in f:
                dp = json.loads(line)

        decisions = dp["decisions"]
        
        with open(label_path_en) as f:
            for line in f:
                dp_en = json.loads(line)

        decisions_en = dp_en["decisions"]
        save_dict = {}
        ori_dp_list = []
        with open(instance_path) as f:
            for i, line in tqdm(enumerate(f)):
                dp = json.loads(line)
                ori_dp_list.append(dp.copy())
        translated_dp_list = []
        with open(instance_path_en) as f:
            for i, line in tqdm(enumerate(f)):
                dp = json.loads(line)
                translated_dp_list.append(dp.copy())
        for i, (ori_dp, translated_dp) in enumerate(zip(ori_dp_list, translated_dp_list)):
            label = decisions[i]
            label_en = decisions_en[i]
            if len(label) != len(label_en):
                print(translated_dp["topic"], label[0], label_en[0])
                continue
            for j, (e, e_en) in enumerate(zip(label, label_en)):
                # save_dict["#".join([ori_dp["topic"], e["atom"]])] = "#".join([translated_dp["topic"], e_en["atom"]])
                save_dict[e["atom"]] = e_en["atom"]
            save_dict[ori_dp["topic"]] = translated_dp["topic"]
        
        # with open(instance_path) as f:
        #     for i, line in tqdm(enumerate(f)):
        #         dp = json.loads(line)
        #         label = decisions[i]
        #         label_en = decisions_en[i]
        #         if len(label) != len(label_en):
        #             print(dp["topic"], label[0], label_en[0])
        #             continue
        #         for j,e in enumerate(label):
        #             # if j == 5:
        #             #     print(e["atom"], label_en[j]["atom"])

        #             save_dict["#".join([dp["topic"], e["atom"]])]= label_en[j]["is_supported"]
        #             #save_dict["#".join([dp["topic"], e["atom"]])]= e["is_supported"]


        save_path_fact = f"~/FActScore/data/to_evaluate/trans2ori/{lang}_{model}_label_by_{evaluator}.json"
        with open(save_path_fact, 'w') as jsonl_file:
            
            json_line = json.dumps(save_dict, ensure_ascii=False)
            jsonl_file.write(json_line + '\n', )
        print("Save path fact:", save_path_fact)

        # save_path_name = f"~/FActScore/data/to_evaluate/trans2ori/name_{lang}_{model}_label_by_{evaluator}.json"
        # with open(save_path_name, 'w') as jsonl_file:
            
        #     json_line = json.dumps(save_dict, ensure_ascii=False)
        #     jsonl_file.write(json_line + '\n', )
        # print("Save path name:", save_path_name)