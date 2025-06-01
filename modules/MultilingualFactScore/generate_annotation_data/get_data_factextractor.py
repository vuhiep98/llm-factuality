import os
import json
# Define the folder path
lang = "es"
folder_path = f"~/FActScore/data/labeled/en_instances/to_train_fact_extract/{lang}_instances"
folder_path = f"~/FActScore/data/labeled/en_instances/to_train_fact_extract/"

# List all files in the folder
files = os.listdir(folder_path)
sent2dict = []
# Iterate through each file
for file_name in files:
    # Check if the item is a file (not a directory)
    if os.path.isfile(os.path.join(folder_path, file_name)) and file_name.endswith("have_facts.jsonl"):
        print(file_name)
        # Open and read the file
        count = 0
        with open(os.path.join(folder_path, file_name), 'r') as file:
            for line in file:
                dp = json.loads(line)
                if dp["annotations"]:
                    count += 1
                    for dict_sent in dp["annotations"]:
                        ts_dict = {}
                        ts_dict["sentence"] = dict_sent["text"]
                        ts_dict["atomic-facts"] = [fact["text"] for fact in dict_sent["model-atomic-facts"]]
                        sent2dict.append(ts_dict)
        print(file_name, count)

output_json_path_train = f"~/FActScore/data/labeled/en_instances/to_train_fact_extract/{lang}_instances/{lang}_sent2fact_train.json"
output_json_path_train = f"~/FActScore/data/labeled/en_instances/to_train_fact_extract/en_sent2fact_train.json"

print("# of training instances:", len(sent2dict))
with open(output_json_path_train, 'w') as jsonl_file:
    # json_line = json.dumps(result, ensure_ascii=False)
    json.dump({"data": sent2dict}, jsonl_file, ensure_ascii=False)

print("Save to:", output_json_path_train)

valid_json_path = f"~/projects/factscore/demos/demons_{lang}.json"
sent2dict_val = []
with open(valid_json_path, 'r') as file:
    for line in file:
        dp = json.loads(line)
        break
    for k, v in dp.items():
        sent2dict_val.append({"sentence": k, "atomic-facts": v})


output_json_path_val = f"~/FActScore/data/labeled/en_instances/to_train_fact_extract/{lang}_instances/{lang}_sent2fact_test.json"
# output_json_path_val = f"~/FActScore/data/labeled/en_instances/to_train_fact_extract/en_sent2fact_test.json"

print("# of validation instances:", len(sent2dict_val))
with open(output_json_path_val, 'w') as jsonl_file:
    # json_line = json.dumps(result, ensure_ascii=False)
    json.dump({"data": sent2dict_val}, jsonl_file, ensure_ascii=False)

print("Save to:", output_json_path_val)

