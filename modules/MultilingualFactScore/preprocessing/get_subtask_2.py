import json
import os
lang = "bn"

entities_list = {"es":["Lionel Messi", "Albert Einstein", "Nelson Mandela", "Paulo Coelho", "Bill Gates"], "ar": ["إليزابيث الثانية", "ليونيل ميسي", "ألبرت أينشتاين", "رامي مالك", "بيل غيتس"], "bn": ["দ্বিতীয় এলিজাবেথ", "লিওনেল মেসি", "বিল গেটস", "আলবার্ট আইনস্টাইন", "এমা ওয়াটসন"]}
model = "gpt4"
path = f"~/FActScore/data/to_evaluate/{lang}/{model}.jsonl"
lst_gpt4 = []
with open(path) as f:
    for l in f:
        dp = json.loads(l)
        if dp["topic"] in entities_list[lang]:
            lst_gpt4.append(dp)

model = "gemini"
path = f"~/FActScore/data/to_evaluate/{lang}/{model}.jsonl"
lst_gemini = []
with open(path) as f:
    for l in f:
        dp = json.loads(l)
        if dp["topic"] in entities_list[lang]:
            lst_gemini.append(dp)
lst_1 = lst_gpt4[:3] + lst_gemini[3:]
lst_2 = lst_gemini[:3] + lst_gpt4[3:]

output_path_1 = f"~/FActScore/data/to_evaluate/{lang}/subtask_extract_1.jsonl"
output_path_2 = f"~/FActScore/data/to_evaluate/{lang}/subtask_extract_2.jsonl"

with open(output_path_1, 'w') as jsonl_file:
    for dictionary in lst_1:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )
with open(output_path_2, 'w') as jsonl_file:
    for dictionary in lst_2:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )