import json
import os
from tqdm import tqdm
evaluator = "GPT-4"
lang = "es"
model = "gemini"
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

for lang in ["es", "ar", "bn"]:
    db_ori2en_dict = {}
    name_ori2en_dict = {}
    tran_path = f"~/projects/factscore/real_evaluation/en_{lang}wiki_100.jsonl"
    ori_path = f"~/projects/factscore/real_evaluation/{lang}wiki_100.jsonl"
    print(tran_path)
    print(ori_path)
    dict_tran = {}
    with open(tran_path) as f:
        for line in f:
            dp = json.loads(line)
            dict_tran[dp["title"]] = dp.copy()
    dict_ori = {}
    with open(ori_path) as f:
        for line in f:
            dp = json.loads(line)
            dict_ori[dp["title"]] = dp.copy()
    # for s1, s2 in zip(lst_ori, lst_tran):
    for k in dict_ori.keys():
        assert k in dict_tran.keys()
        name_ori2en_dict[k] = dict_tran[k]["en_title"]
        s1 = dict_ori[k]
        s2 = dict_tran[k]
        assert len(s1["text"].split(SPECIAL_SEPARATOR)) == len(s2["text"].split(SPECIAL_SEPARATOR))
        for ori_text, en_text in zip(s1["text"].split(SPECIAL_SEPARATOR), s2["text"].split(SPECIAL_SEPARATOR)):
            db_ori2en_dict[ori_text.replace("<s>", "").replace("</s>", "").strip()] = en_text.replace("<s>", "").replace("</s>", "").strip()

    save_path_name = f"~/FActScore/data/to_evaluate/trans2ori/{lang}_name.json"
    save_path_db = f"~/FActScore/data/to_evaluate/trans2ori/{lang}_db.json"
    with open(save_path_name, 'w') as jsonl_file:
        
        json_line = json.dumps(name_ori2en_dict, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )
    print("Save path name:", save_path_name)
    with open(save_path_db, 'w') as jsonl_file:
        
        json_line = json.dumps(db_ori2en_dict, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )
    print("Save path db:", save_path_db)
