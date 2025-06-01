import json
lang = "bn"
model = "gpt4"
path_to_name = f"~/projects/factscore/real_evaluation/en_{lang}wiki_100.jsonl"
path_to_file = f"~/FActScore/data/to_evaluate/{lang}/en_instances/{model}.jsonl"
mapping = {}
with open(path_to_name) as f:
    for line in f:
        dp = json.loads(line)
        mapping[dp["title"]] = dp["en_title"]
lst = []
with open(path_to_file) as f:
    for line in f:
        dp = json.loads(line)
        dp["topic"] = mapping[dp["topic"]]
        lst.append(dp.copy())
with open(path_to_file, 'w') as jsonl_file:
    for dictionary in lst:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )
print(path_to_file)
# print(mapping)
# print(lst[0])