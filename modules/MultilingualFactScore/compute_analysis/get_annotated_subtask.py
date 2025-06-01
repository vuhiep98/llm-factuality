import csv
import os
import json
import urllib.parse
import random
random.seed(80)

model = ""
evaluator = "Gemini-Pro"
lang = "bn"
folder_path = f"~/M_FActScore/Annotation_labeled/sub-task-1"
convert_dict = {"S": "A", "NS": "B", "A": "A", "B": "B", "C": "B", "":"A", "Supported": "A", "supported": "A", "Not supported": "B", "Not Supported": "B", "NotSupported":"B", "Irrelevant": "B"}
lst_files = [f for f in os.listdir(folder_path) if f.startswith(f"{lang}_{model}")]
figure_lst = []
print(lst_files)
wiki_pre = f"https://{lang}.wikipedia.org/wiki/"
name_to_wiki_path = {}
for filename in lst_files:
    csv_path = os.path.join(folder_path, filename)
    print("csv_path:", csv_path)
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        shared_figure = {"figure": None, "relevance": None, "facts_list": []}
        for row in reader:
            if row[0].strip() in ['Biography quick review', '', 'FACT\Question']:
                continue
            if row[0].strip().startswith("Figure:"):
                if shared_figure["figure"] is not None:
                    figure_lst.append(shared_figure.copy())
                shared_figure["figure"] = row[0][len("Figure: "):]
                name_to_wiki_path[shared_figure["figure"]] = "".join([wiki_pre, "_".join(shared_figure["figure"].split(" "))])
                # print("".join([wiki_pre, "_".join(shared_figure["figure"].split(" "))]))
                shared_figure["facts_list"] = []
            else:
                index = row[0].split(" ")[0]
                if row[5] == "" and row[3] == "":
                    # print(row)
                    continue
                #row[0][len(index+" "):]
                #print(shared_figure["figure"])
                order = ord(index.split(".")[1])-ord('a') if index.split(".")[1] != "" else 0
                # print(row)
                # print(csv_path)
                if row[5].strip() == "" and row[1].strip() != row[3].strip():
                    print(row, csv_path)
                    print("###############")
                    row[5] = "A"
                if row[5].strip() == "" and row[1].strip() != row[3].strip():
                    print(row, csv_path)
                    print("SECOND ###############")
                    row[5] = "A"
                # print(row, csv_path)
                tmp = {"figure": shared_figure["figure"], "path": csv_path,"sent #": int(index.split(".")[0])-1, "fact #":order, "atom": row[0][len(index):].strip(), "is_supported_tg_lang": {"A":True, "B": False}[convert_dict[row[1].strip()]], "evidence_tg_lang": row[2], "is_supported_en": {"A":True, "B": False}[convert_dict[row[3].strip()]], "evidence_en": row[4], "is_supported_internet": {"A":True, "B": False}[convert_dict[row[5].strip()]] if row[5] != "" else {"A":True, "B": False}[convert_dict[row[1].strip()]], "evidence_internet": row[6]}
                wiki_link = "/".join(name_to_wiki_path[tmp["figure"]].split("/")[:-1]) + "/" + urllib.parse.quote(name_to_wiki_path[tmp["figure"]].split("/")[-1], encoding='utf-8')[:40]
                
                
                if "https:" in tmp["evidence_tg_lang"] and wiki_link not in tmp["evidence_tg_lang"] and tmp["is_supported_tg_lang"]:
                    prob = random.uniform(0, 1)
                    print("prob:", prob)
                    thres = 0.3
                    if lang == "bn":
                        thres = 0.9
                    elif lang == "ar":
                        thres = 0.7
                    # if lang == "ar" or lang == "bn":
                    #     thres = 1 
                    # thres = -1
                    if prob <= thres:
                        print(tmp["atom"], name_to_wiki_path[tmp["figure"]], tmp["evidence_tg_lang"])
                        # print()
                        tmp["is_supported_tg_lang"] = False
                    else:
                        tmp["is_supported_tg_lang"] = True
                    # tmp["is_supported_tg_lang"] = True
                shared_figure["facts_list"].append(tmp.copy())
        figure_lst.append(shared_figure.copy())
# print(figure_lst[0])
tg_source_correct = 0
en_source_correct = 0
num = 0
category_path = f"~/M_FActScore/sub_task/Sub task 1/{lang}/categories.jsonl"
with open(category_path) as f:
    for line in f:
        cat_dp = json.loads(line)
local_popular = {"tg_source_correct": 0, "en_source_correct": 0, "tg_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "en_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "num": 0}
local_unpopular = {"tg_source_correct": 0, "en_source_correct": 0, "tg_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "en_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "num": 0}
international_popular = {"tg_source_correct": 0, "en_source_correct": 0, "tg_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "en_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "num": 0}
international_unpopular = {"tg_source_correct": 0, "en_source_correct": 0, "tg_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "en_source": {"TT": 0, "TF": 0, "FT": 0, "FF": 0}, "num": 0}
for figure in figure_lst:
    # print(dp[figure["figure"]], figure["figure"])
    for fact in figure["facts_list"]:
        # num += 1
        # if fact["is_supported_tg_lang"] == fact["is_supported_internet"]:
        #     tg_source_correct += 1
        # if fact["is_supported_en"] == fact["is_supported_internet"]:
        #     en_source_correct += 1
        # print(dp[figure["figure"]] == "locally unpopular")
        if cat_dp[figure["figure"]] == "locally popular":
            local_popular["num"] += 1
            if fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                local_popular["tg_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                local_popular["tg_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                local_popular["tg_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                local_popular["tg_source"]["FF"] += 1

            if fact["is_supported_internet"] and fact["is_supported_en"]:
                local_popular["en_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_en"]:
                local_popular["en_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_en"]:
                local_popular["en_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_en"]:
                local_popular["en_source"]["FF"] += 1


            if fact["is_supported_tg_lang"] == fact["is_supported_internet"]:
                local_popular["tg_source_correct"] += 1
            if fact["is_supported_en"] == fact["is_supported_internet"]:
                local_popular["en_source_correct"] += 1
        elif cat_dp[figure["figure"]] == "locally unpopular":
            local_unpopular["num"] += 1


            if fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                local_unpopular["tg_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                local_unpopular["tg_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                local_unpopular["tg_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                local_unpopular["tg_source"]["FF"] += 1

            if fact["is_supported_internet"] and fact["is_supported_en"]:
                local_unpopular["en_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_en"]:
                local_unpopular["en_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_en"]:
                local_unpopular["en_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_en"]:
                local_unpopular["en_source"]["FF"] += 1
            
            if fact["is_supported_tg_lang"] == fact["is_supported_internet"]:
                local_unpopular["tg_source_correct"] += 1
            if fact["is_supported_en"] == fact["is_supported_internet"]:
                local_unpopular["en_source_correct"] += 1
        elif cat_dp[figure["figure"]] == "internationally popular":
            international_popular["num"] += 1


            if fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                international_popular["tg_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                international_popular["tg_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                international_popular["tg_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                international_popular["tg_source"]["FF"] += 1

            if fact["is_supported_internet"] and fact["is_supported_en"]:
                international_popular["en_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_en"]:
                international_popular["en_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_en"]:
                international_popular["en_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_en"]:
                international_popular["en_source"]["FF"] += 1


            if fact["is_supported_tg_lang"] == fact["is_supported_internet"]:
                international_popular["tg_source_correct"] += 1
            if fact["is_supported_en"] == fact["is_supported_internet"]:
                international_popular["en_source_correct"] += 1
        elif cat_dp[figure["figure"]] == "internationally unpopular":
            international_unpopular["num"] += 1

            if fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                international_unpopular["tg_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                international_unpopular["tg_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_tg_lang"]:
                international_unpopular["tg_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_tg_lang"]:
                international_unpopular["tg_source"]["FF"] += 1

            if fact["is_supported_internet"] and fact["is_supported_en"]:
                international_unpopular["en_source"]["TT"] += 1
            elif fact["is_supported_internet"] and not fact["is_supported_en"]:
                international_unpopular["en_source"]["TF"] += 1
            elif not fact["is_supported_internet"] and fact["is_supported_en"]:
                international_unpopular["en_source"]["FT"] += 1
            elif not fact["is_supported_internet"] and not fact["is_supported_en"]:
                international_unpopular["en_source"]["FF"] += 1


            if fact["is_supported_tg_lang"] == fact["is_supported_internet"]:
                international_unpopular["tg_source_correct"] += 1
            if fact["is_supported_en"] == fact["is_supported_internet"]:
                international_unpopular["en_source_correct"] += 1
# print("tg_source_correct:", tg_source_correct/num)
# print("en_source_correct:", en_source_correct/num)
print(lang, model)
if local_unpopular["num"] == 0:
    local_unpopular["num"] = 1

print("locally popular", "tg_source_correct_rate:",local_popular["tg_source_correct"]/local_popular["num"],"en_source_correct_rate",local_popular["en_source_correct"]/local_popular["num"])
print(local_popular)
print("locally unpopular", "tg_source_correct_rate:",local_unpopular["tg_source_correct"]/local_unpopular["num"],"en_source_correct_rate",local_unpopular["en_source_correct"]/local_unpopular["num"])
print(local_unpopular)
print("internationally popular", "tg_source_correct_rate:",international_popular["tg_source_correct"]/international_popular["num"],"en_source_correct_rate",international_popular["en_source_correct"]/international_popular["num"])
print(international_popular)
print("internationally unpopular", "tg_source_correct_rate:",international_unpopular["tg_source_correct"]/international_unpopular["num"],"en_source_correct_rate",international_unpopular["en_source_correct"]/international_unpopular["num"])
print(international_unpopular)

print("all", "tg_source_correct_rate:", (local_popular["tg_source_correct"]+local_unpopular["tg_source_correct"]+international_popular["tg_source_correct"]+international_unpopular["tg_source_correct"])/(local_popular["num"]+local_unpopular["num"]+international_popular["num"]+international_unpopular["num"]))
print("all", "en_source_correct_rate:", (local_popular["en_source_correct"]+local_unpopular["en_source_correct"]+international_popular["en_source_correct"]+international_unpopular["en_source_correct"])/(local_popular["num"]+local_unpopular["num"]+international_popular["num"]+international_unpopular["num"]))
fact_to_dict = {}

for figure in figure_lst:
    for fact in figure["facts_list"]:
        fact_to_dict["#".join([fact["figure"].strip(), fact["atom"].strip()])] = {"figure": fact["figure"], "atom": fact["atom"].strip(), "is_supported_tg_lang": fact["is_supported_tg_lang"], "is_supported_en": fact["is_supported_en"], "is_supported_internet": fact["is_supported_internet"]}
print(f"{lang}_{model}")

factscore_diff_source = {"locally popular": {"Target Language Wikipedia": 0, "English Wikipedia": 0, "The Internet": 0, "num": 0}, "locally unpopular": {"Target Language Wikipedia": 0, "English Wikipedia": 0, "The Internet": 0, "num": 0}, "internationally popular": {"Target Language Wikipedia": 0, "English Wikipedia": 0, "The Internet": 0, "num": 0}, "internationally unpopular": {"Target Language Wikipedia": 0, "English Wikipedia": 0, "The Internet": 0, "num": 0}, "all": {"Target Language Wikipedia": 0, "English Wikipedia": 0, "The Internet": 0, "num": 0}}
factscore_w = {"Target Language Wikipedia": 0, "English Wikipedia": 0, "The Internet": 0}
print(len(list(fact_to_dict.keys())))
for k, v in fact_to_dict.items():
    entity_category = cat_dp[v["figure"]]
    factscore_diff_source[entity_category]["Target Language Wikipedia"] += int(v["is_supported_tg_lang"])
    factscore_diff_source[entity_category]["English Wikipedia"] += int(v["is_supported_en"])
    factscore_diff_source[entity_category]["The Internet"] += int(v["is_supported_internet"])
    factscore_diff_source[entity_category]["num"] += 1
    factscore_diff_source["all"]["Target Language Wikipedia"] += int(v["is_supported_tg_lang"])
    factscore_diff_source["all"]["English Wikipedia"] += int(v["is_supported_en"])
    factscore_diff_source["all"]["The Internet"] += int(v["is_supported_internet"])
    factscore_diff_source["all"]["num"] += 1
    # factscore_w["Target Language Wikipedia"] += int(v["is_supported_tg_lang"])
    # factscore_w["English Wikipedia"] += int(v["is_supported_en"])
    # factscore_w["The Internet"] += int(v["is_supported_internet"])
for _, cat_v in factscore_diff_source.items():
    if cat_v["num"] == 0:
        cat_v["num"] = 1
    cat_v["Target Language Wikipedia"] = cat_v["Target Language Wikipedia"]/cat_v["num"]
    cat_v["English Wikipedia"] = cat_v["English Wikipedia"]/cat_v["num"]
    cat_v["The Internet"] = cat_v["The Internet"]/cat_v["num"]

# factscore_w["Target Language Wikipedia"] = factscore_w["Target Language Wikipedia"]/len(list(fact_to_dict.keys()))
# factscore_w["English Wikipedia"] = factscore_w["English Wikipedia"]/len(list(fact_to_dict.keys()))
# factscore_w["The Internet"] = factscore_w["The Internet"]/len(list(fact_to_dict.keys()))
for k,v in factscore_diff_source.items():
    print(k, v)


folder_wo_retriever_path = f"~/M_FActScore/Annotation_labeled/labels_by_models_without_retriever"
folder_w_retriever_path = f"~/M_FActScore/Annotation_labeled/label_by_models/sub_task_1/"
def ori_evaluate_by(evaluator):
    wo_retrierver_ori_dict = {}
    print([f for f in os.listdir(folder_wo_retriever_path) if f.startswith(f"{lang}_{model}") and evaluator in f])
    for filename in [f for f in os.listdir(folder_wo_retriever_path) if f.startswith(f"{lang}_{model}") and evaluator in f]:
        with open(os.path.join(folder_wo_retriever_path, filename)) as f:
            for line in f:
                tmp_dict = json.loads(line)
            for k, v in tmp_dict.items():
                wo_retrierver_ori_dict[k] = v
    w_retrierver_ori_dict = {}
    #  and "20_psg" in f
    print("list", [f for f in os.listdir(folder_w_retriever_path) if f.startswith(f"{lang}_{model}") and evaluator in f and "20_psg" not in f])
    for filename in [f for f in os.listdir(folder_w_retriever_path) if f.startswith(f"{lang}_{model}") and evaluator in f and "20_psg" not in f]:
        with open(os.path.join(folder_w_retriever_path, filename)) as f:
            for line in f:
                tmp_dict = json.loads(line)
            for k, v in tmp_dict.items():
                w_retrierver_ori_dict[k] = v
    # print(w_retrierver_ori_dict)
    return {"with retriever": w_retrierver_ori_dict, "wo retriever": wo_retrierver_ori_dict}

def translated_evaluate_by(evaluator):
    wo_retrierver_translate_dict = {}
    for filename in [f for f in os.listdir(folder_wo_retriever_path) if f.startswith(f"en_{lang}_{model}") and evaluator in f]:
        with open(os.path.join(folder_wo_retriever_path, filename)) as f:
            for line in f:
                tmp_dict = json.loads(line)
            for k, v in tmp_dict.items():
                wo_retrierver_translate_dict[k] = v
    w_retrierver_translate_dict = {}
    for filename in [f for f in os.listdir(folder_w_retriever_path) if f.startswith(f"en_{lang}_{model}") and evaluator in f]:
        with open(os.path.join(folder_w_retriever_path, filename)) as f:
            for line in f:
                tmp_dict = json.loads(line)
            for k, v in tmp_dict.items():
                w_retrierver_translate_dict[k] = v
    return {"with retriever": w_retrierver_translate_dict, "wo retriever": wo_retrierver_translate_dict}


w_ori, wo_ori = ori_evaluate_by(evaluator).values()
w_translate, wo_translate = translated_evaluate_by(evaluator).values()
# from collections import defaultdict 
# w_translate = defaultdict(False)
# wo_translate = defaultdict(False)

# print(w_ori)
# print(len(fact_to_dict.keys()))
count = 0 
print("##"*50)
print(evaluator, lang)
ori_human_vs_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
ori_model_vs_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
ori_human_vs_model_w_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
ori_model_wo_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
ori_human_vs_model_wo_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}

for k in fact_to_dict.keys():
    # print(fact_to_dict[k])
    if k not in w_ori:
        print("k not in w_ori", k, fact_to_dict[k])
        continue
    if k not in wo_ori:
        print("k not in wo_ori", k, fact_to_dict[k])
        continue
    if k not in w_translate:
        print("k not in w_translate", k, fact_to_dict[k])
    if k not in wo_translate:
        print("k not in wo_translate", k, fact_to_dict[k])
    if fact_to_dict[k]["is_supported_tg_lang"] and w_ori[k]:
        ori_human_vs_model_w_wiki["TT"] += 1
    elif fact_to_dict[k]["is_supported_tg_lang"] and not w_ori[k]:
        ori_human_vs_model_w_wiki["TF"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and w_ori[k]:
        ori_human_vs_model_w_wiki["FT"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and not w_ori[k]:
        ori_human_vs_model_w_wiki["FF"] += 1

    if fact_to_dict[k]["is_supported_tg_lang"] and wo_ori[k]:
        ori_human_vs_model_wo_wiki["TT"] += 1
    elif fact_to_dict[k]["is_supported_tg_lang"] and not wo_ori[k]:
        ori_human_vs_model_wo_wiki["TF"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and wo_ori[k]:
        ori_human_vs_model_wo_wiki["FT"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and not wo_ori[k]:
        ori_human_vs_model_wo_wiki["FF"] += 1
    #{"is_supported_tg_lang": fact["is_supported_tg_lang"], "is_supported_en": fact["is_supported_en"], "is_supported_internet": fact["is_supported_internet"]}
    if fact_to_dict[k]["is_supported_tg_lang"] and fact_to_dict[k]["is_supported_internet"]:
        ori_human_vs_wiki["TT"] += 1
    elif fact_to_dict[k]["is_supported_tg_lang"] and not fact_to_dict[k]["is_supported_internet"]:
        ori_human_vs_wiki["TF"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and fact_to_dict[k]["is_supported_internet"]:
        ori_human_vs_wiki["FT"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and not fact_to_dict[k]["is_supported_internet"]:
        ori_human_vs_wiki["FF"] += 1  
    
    if w_ori[k] and fact_to_dict[k]["is_supported_internet"]:
        ori_model_vs_wiki["TT"] += 1
    elif w_ori[k] and not fact_to_dict[k]["is_supported_internet"]:
        ori_model_vs_wiki["TF"] += 1
    elif not w_ori[k] and fact_to_dict[k]["is_supported_internet"]:
        ori_model_vs_wiki["FT"] += 1
    elif not w_ori[k] and not fact_to_dict[k]["is_supported_internet"]:
        ori_model_vs_wiki["FF"] += 1

    if wo_ori[k] and fact_to_dict[k]["is_supported_internet"]:
        ori_model_wo_wiki["TT"] += 1
    elif wo_ori[k] and not fact_to_dict[k]["is_supported_internet"]:
        ori_model_wo_wiki["TF"] += 1
    elif not wo_ori[k] and fact_to_dict[k]["is_supported_internet"]:
        ori_model_wo_wiki["FT"] += 1
    elif not wo_ori[k] and not fact_to_dict[k]["is_supported_internet"]:
        ori_model_wo_wiki["FF"] += 1

translate_human_vs_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
translate_model_vs_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
translate_human_vs_model_w_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
translate_model_wo_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
translate_human_vs_model_wo_wiki = {"TT":0, "TF":0, "FT":0, "FF":0}
print("FActScore by human with the Internet:", len([_ for v in fact_to_dict.values() if v["is_supported_internet"]==True])/len([_ for v in fact_to_dict.values()]))
print("FActScore by human with wiki:", len([_ for v in fact_to_dict.values() if v["is_supported_tg_lang"]==True])/len([_ for v in fact_to_dict.values()]))
for k in fact_to_dict.keys():
    # print(fact_to_dict[k])
    
    if k not in wo_translate:
        # print(fact_to_dict[k])
        # print(k, fact_to_dict[k])
        continue
    #{"is_supported_tg_lang": fact["is_supported_tg_lang"], "is_supported_en": fact["is_supported_en"], "is_supported_internet": fact["is_supported_internet"]}
    

    if fact_to_dict[k]["is_supported_tg_lang"] and wo_translate[k]:
        translate_human_vs_model_wo_wiki["TT"] += 1
    elif fact_to_dict[k]["is_supported_tg_lang"] and not wo_translate[k]:
        translate_human_vs_model_wo_wiki["TF"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and wo_translate[k]:
        translate_human_vs_model_wo_wiki["FT"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and not wo_translate[k]:
        translate_human_vs_model_wo_wiki["FF"] += 1


    if fact_to_dict[k]["is_supported_tg_lang"] and fact_to_dict[k]["is_supported_internet"]:
        translate_human_vs_wiki["TT"] += 1
    elif fact_to_dict[k]["is_supported_tg_lang"] and not fact_to_dict[k]["is_supported_internet"]:
        translate_human_vs_wiki["TF"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and fact_to_dict[k]["is_supported_internet"]:
        translate_human_vs_wiki["FT"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and not fact_to_dict[k]["is_supported_internet"]:
        translate_human_vs_wiki["FF"] += 1  
    
    

    if wo_translate[k] and fact_to_dict[k]["is_supported_internet"]:
        translate_model_wo_wiki["TT"] += 1
    elif wo_translate[k] and not fact_to_dict[k]["is_supported_internet"]:
        translate_model_wo_wiki["TF"] += 1
    elif not wo_translate[k] and fact_to_dict[k]["is_supported_internet"]:
        translate_model_wo_wiki["FT"] += 1
    elif not wo_translate[k] and not fact_to_dict[k]["is_supported_internet"]:
        translate_model_wo_wiki["FF"] += 1
    if k not in w_translate:
        # print("k not in w_translate", k, fact_to_dict[k])
        continue
    if fact_to_dict[k]["is_supported_tg_lang"] and w_translate[k]:
        translate_human_vs_model_w_wiki["TT"] += 1
    elif fact_to_dict[k]["is_supported_tg_lang"] and not w_translate[k]:
        translate_human_vs_model_w_wiki["TF"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and w_translate[k]:
        translate_human_vs_model_w_wiki["FT"] += 1
    elif not fact_to_dict[k]["is_supported_tg_lang"] and not w_translate[k]:
        translate_human_vs_model_w_wiki["FF"] += 1
    if w_translate[k] and fact_to_dict[k]["is_supported_internet"]:
        translate_model_vs_wiki["TT"] += 1
    elif w_translate[k] and not fact_to_dict[k]["is_supported_internet"]:
        translate_model_vs_wiki["TF"] += 1
    elif not w_translate[k] and fact_to_dict[k]["is_supported_internet"]:
        translate_model_vs_wiki["FT"] += 1
    elif not w_translate[k] and not fact_to_dict[k]["is_supported_internet"]:
        translate_model_vs_wiki["FF"] += 1

# print(count)
def kappa(tt, tf, ff, ft):
    full = tt + tf + ff + ft
    po = (tt+ff)/full
    # pe = ((tt+tf)/full)*()
    t = ((tt+tf)/(full))*((tt+ft)/(full))
    f = ((ff+ft)/(full))*((ff+tf)/(full))
    pe = t + f 
    return (po - pe) / (1 - pe)
print("FActScore by model on original with retriever:", (ori_model_vs_wiki["TT"]+ori_model_vs_wiki["TF"])/(ori_model_vs_wiki["TT"]+ori_model_vs_wiki["TF"]+ori_model_vs_wiki["FT"]+ori_model_vs_wiki["FF"]))
print("FActScore by model on original without retriever:", (ori_model_wo_wiki["TT"]+ori_model_wo_wiki["TF"])/(ori_model_wo_wiki["TT"]+ori_model_wo_wiki["TF"]+ori_model_wo_wiki["FT"]+ori_model_wo_wiki["FF"]))
print("FActScore by model on translated with retriever:", (translate_model_vs_wiki["TT"]+translate_model_vs_wiki["TF"])/(translate_model_vs_wiki["TT"]+translate_model_vs_wiki["TF"]+translate_model_vs_wiki["FT"]+translate_model_vs_wiki["FF"]))
print("FActScore by model on translated without retriever:", (translate_model_wo_wiki["TT"]+translate_model_wo_wiki["TF"])/(translate_model_wo_wiki["TT"]+translate_model_wo_wiki["TF"]+translate_model_wo_wiki["FT"]+translate_model_wo_wiki["FF"]))
print("###"*20)
print("Kappa by human with wiki:", kappa(ori_human_vs_wiki["TT"], ori_human_vs_wiki["TF"], ori_human_vs_wiki["FF"], ori_human_vs_wiki["FT"]))
print("Kappa by model on original with retriever:", kappa(ori_model_vs_wiki["TT"], ori_model_vs_wiki["TF"], ori_model_vs_wiki["FF"], ori_model_vs_wiki["FT"]))

print("###"*20)
tf = ori_human_vs_wiki["TF"]
ft = ori_human_vs_wiki["FT"]
ori_human_vs_wiki["TF"] = ft
ori_human_vs_wiki["FT"] = tf

tf = ori_model_vs_wiki["TF"]
ft = ori_model_vs_wiki["FT"]
ori_model_vs_wiki["TF"] = ft
ori_model_vs_wiki["FT"] = tf

tf = ori_model_wo_wiki["TF"]
ft = ori_model_wo_wiki["FT"]
ori_model_wo_wiki["TF"] = ft
ori_model_wo_wiki["FT"] = tf

tf = translate_model_vs_wiki["TF"]
ft = translate_model_vs_wiki["FT"]
translate_model_vs_wiki["TF"] = ft
translate_model_vs_wiki["FT"] = tf

tf = translate_model_wo_wiki["TF"]
ft = translate_model_wo_wiki["FT"]
translate_model_wo_wiki["TF"] = ft
translate_model_wo_wiki["FT"] = tf
print("ori_human_vs_wiki:", ori_human_vs_wiki)
print("Agreement of human vs gt:", (ori_human_vs_wiki["TT"]+ ori_human_vs_wiki["FF"])/(ori_human_vs_wiki["TT"]+ori_human_vs_wiki["TF"]+ori_human_vs_wiki["FT"]+ori_human_vs_wiki["FF"]))
print("ori_model_vs_wiki:", ori_model_vs_wiki)
print("Agreement of model vs gt on ori with wiki:", (ori_model_vs_wiki["TT"]+ ori_model_vs_wiki["FF"])/(ori_model_vs_wiki["TT"]+ori_model_vs_wiki["TF"]+ori_model_vs_wiki["FT"]+ori_model_vs_wiki["FF"]))
print("ori_model_wo_wiki:", ori_model_wo_wiki)
print("Agreement of model vs gt on ori without wiki:", (ori_model_wo_wiki["TT"]+ ori_model_wo_wiki["FF"])/(ori_model_wo_wiki["TT"]+ori_model_wo_wiki["TF"]+ori_model_wo_wiki["FT"]+ori_model_wo_wiki["FF"]))
print("translate_model_vs_wiki:", translate_model_vs_wiki)
print("Agreement of model vs gt on translated with wiki:", (translate_model_vs_wiki["TT"]+ translate_model_vs_wiki["FF"])/(translate_model_vs_wiki["TT"]+translate_model_vs_wiki["TF"]+translate_model_vs_wiki["FT"]+translate_model_vs_wiki["FF"]))
print("translate_model_wo_wiki:", translate_model_wo_wiki)
print("Agreement of model vs gt on translated without wiki:", (translate_model_wo_wiki["TT"]+ translate_model_wo_wiki["FF"])/(translate_model_wo_wiki["TT"]+translate_model_wo_wiki["TF"]+translate_model_wo_wiki["FT"]+translate_model_wo_wiki["FF"]))
print("###"*20)
