import csv
import os
import json
import urllib.parse
import random
random.seed(80)
def custom_quote(s):
    encoded_chars = []
    for char in s:
        if char == "(" or char == ")":
            encoded_chars.append(char)
        else:
            encoded_chars.append(urllib.parse.quote(char, encoding='utf-8'))
    return ''.join(encoded_chars)


model = "gpt4"
lang = "bn"
folder_path = f"~/M_FActScore/Annotation_labeled/{lang}_full"
convert_dict = {"a": "A", "S": "A", "NS": "B", "A": "A", "B": "B", "C": "B", "":"A", "Supported": "A", "supported": "A", "Not supported": "B", "Not Supported": "B", "NotSupported":"B", "Irrelevant": "B"}
lst_files = [f for f in os.listdir(folder_path) if f.startswith(model)]
figure_lst = []
for filename in lst_files:
    csv_path = os.path.join(folder_path, filename)
    print(csv_path)
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        shared_figure = {"figure": None, "relevance": None, "facts_list": []}
        for row in reader:
            if row[0].strip() in ['Biography quick review', '', 'FACT\Question']:
                continue
            if row[0].strip().startswith("Figure:"):
                if shared_figure["figure"] is not None:
                    figure_lst.append(shared_figure.copy())
                shared_figure["figure"] = row[0][len("Figure: "):].strip()
                shared_figure["facts_list"] = []
            elif row[0].strip() == "Question #0":
                assert row[1] != "", (csv_path, shared_figure["figure"])
                if row[1].strip().lower() in ["", "a", "relevant"]:
                    assert row[1].strip().lower() != "", csv_path
                    shared_figure["relevance"] = "relevant"
                elif row[1].strip().lower() in ["irrelevant", "irrelavant", "irrerlevant","b", "irelevant"]:
                    shared_figure["relevance"] = "irrelevant"
                elif row[1].strip().lower() in ["abstain", "c", "x"]:
                    shared_figure["relevance"] = "abstain"
                else:
                    # print(csv_path)
                    assert False, print(row)
                
            else:
                index = row[0].split(" ")[0]
                if row[1] == "" and row[3] == "":
                    # print(row)
                    continue
                #row[0][len(index+" "):]
                #print(shared_figure["figure"])
                order = ord(index.split(".")[1])-ord('a') if index.split(".")[1] != "" else 0
                tmp = {"csv_path": csv_path, "figure": shared_figure["figure"], "path": csv_path,"sent #": int(index.split(".")[0])-1, "fact #":order, "atom": row[0][len(index):].strip(), "extract": row[1], "fix extract": row[2], "is_supported": {"A":True, "B": False}[convert_dict[row[3].strip()]], "evidence": row[4]}
                # if len(tmp["evidence"]) != 0 and len(tmp["factual"]) == 0:
                #    print(csv_path, tmp, shared_figure["figure"])
                # if tmp["factual"] == "B" and tmp["evidence"] == "":
                #     check.append(csv_path)
                #     print(csv_path, tmp, shared_figure["figure"])
                if tmp["extract"] != "Yes" and tmp["extract"] != "" and shared_figure["relevance"] == "A":
                    print(csv_path, shared_figure["figure"], tmp)
                    # check.append((csv_path, shared_figure["figure"], tmp))
                # if tmp["fact #"] > 0:
                #     assert shared_figure["facts_list"][-1]["fact #"] == tmp["fact #"] - 1, (shared_figure["figure"], row, shared_figure["facts_list"])
                
                shared_figure["facts_list"].append(tmp.copy())
        figure_lst.append(shared_figure.copy())

figure_dict = {}
print("len(figure_lst)", len(figure_lst))
for e in figure_lst:
    if e["figure"] in figure_dict:
        print("e[\"figure\"]", e["figure"])
    figure_dict[e["figure"]] = e
path_to_save = f"~/M_FActScore/Annotation_labeled/run_ai_labeled/{lang}_{model}_labeled.jsonl"
lst_to_save = []
count_all = 0
count_out = 0
count_true = 0
answers = []
name2obj = {}
abstain_num = 0
relevant_num = 0
irrelavant_num = 0
print("len(figure_dict.keys())", len(figure_dict.keys()))
with open(path_to_save) as f:
    for line in f:
        dp = json.loads(line)
        
        if dp["topic"] in figure_dict.keys():
            if figure_dict[dp["topic"]]["relevance"] != "relevant":
                if figure_dict[dp["topic"]]["relevance"] == "abstain":
                    abstain_num += 1
                elif figure_dict[dp["topic"]]["relevance"] == "irrelevant":
                    irrelavant_num += 1
                continue
            relevant_num += 1
            transformed_link = custom_quote(dp["link"].split("/")[-1])
            wiki_link_tmp = "/".join(dp["link"].split("/")[:-1]) + "/" + transformed_link
            wiki_link = "/".join(dp["link"].split("/")[:-1]) + "/" + transformed_link[:40]
            # wiki_link_en = wiki_link.replace("//es.wikipedia", "//en.wikipedia")
            for fact in figure_dict[dp["topic"]]["facts_list"]:
                if fact["extract"] == "" and fact["is_supported"] == "":
                    continue
                count_all += 1
                #and wiki_link not in fact["evidence"].replace("//en.wikipedia", "//es.wikipedia") 
                if "https:" in fact["evidence"] and wiki_link not in fact["evidence"] and fact["is_supported"]:
                    prob = random.uniform(0, 1)
                    print("prob:", prob)
                    thres = 0.3
                    if lang == "bn":
                        thres = 0.9
                    elif lang == "ar":
                        thres = 0.7
                    # if lang == "ar" or lang == "bn":
                    thres = -1
                    if prob <= thres:
                        fact["is_supported"] = False
                        fact["real_label"] = True
                        
                        print("File:", fact["csv_path"])
                        print("EVIDENCE", fact["figure"], fact["atom"],wiki_link_tmp, fact["evidence"])
                        print("====")
                        count_out += 1
                if fact["is_supported"]:
                    count_true += 1
            dp["relevance"] = figure_dict[dp["topic"]]["relevance"]
            dp["human_labeling"] = figure_dict[dp["topic"]]["facts_list"]
            
            
        else:
            print("not yet", dp["topic"])
            dp["relevance"] = "not yet"
            dp["human_labeling"] = "not yet"
        name2obj[dp["topic"]] = dp
        lst_to_save.append(dp.copy())
print("COUNT OUT:", count_out)
print((count_all-count_out)/count_all)
# with open(path_to_save, 'w') as jsonl_file:
#     for dictionary in lst_to_save:
#         json_line = json.dumps(dictionary, ensure_ascii=False)
#         jsonl_file.write(json_line + '\n', )
print(path_to_save)
dict_chatgpt = {}
dict_gpt4 = {}
dict_mistral = {}
dict_human = {}
num_bios = 0
gemini_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_Gemini/{lang}_{model}_label_by_Gemini-Pro.json"
chatgpt_en_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/en_{lang}_{model}_label_by_ChatGPT.json"
gemini_en_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/en_{lang}_{model}_label_by_Gemini-Pro.json"
mistral_en_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/en_{lang}_{model}_label_by_mistral.json"
gpt4_en_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/en_{lang}_{model}_label_by_GPT-4.json"
gpt4_npm_en_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/en_{lang}_{model}_label_by_GPT-4+npm.json"
mistral_npm_en_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/en_{lang}_{model}_label_by_mistral+npm.json"
chatgpt_npm_en_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/en_{lang}_{model}_label_by_ChatGPT+npm.json"
gemini_20_psg_labeling_path = f"~/M_FActScore/Annotation_labeled/label_by_models/{lang}_{model}_label_by_Gemini-Pro_20_psg.json"

with open(gemini_labeling_path) as f:
    for line in f:
        gemini_labeling_dict = json.loads(line)
with open(chatgpt_en_labeling_path) as f:
    for line in f:
        chatgpt_en_labeling_dict = json.loads(line)
with open(gemini_en_labeling_path) as f:
    for line in f:
        gemini_en_labeling_dict = json.loads(line)
with open(gpt4_en_labeling_path) as f:
    for line in f:
        gpt4_en_labeling_dict = json.loads(line)
with open(gpt4_npm_en_labeling_path) as f:
    for line in f:
        gpt4_npm_en_labeling_dict = json.loads(line)
with open(chatgpt_npm_en_labeling_path) as f:
    for line in f:
        chatgpt_npm_en_labeling_dict = json.loads(line)
with open(gemini_20_psg_labeling_path) as f:
    for line in f:
        gemini_20_psg_labeling_dict = json.loads(line)
with open(mistral_npm_en_labeling_path) as f:
    for line in f:
        mistral_npm_en_labeling_dict = json.loads(line)
with open(mistral_en_labeling_path) as f:
    for line in f:
        mistral_en_labeling_dict = json.loads(line)
fact2entity = {}
for dp in lst_to_save:
    if dp["human_labeling"] == "not yet" or dp["relevance"] != "relevant":
        continue
    
    chatgpt_labeling = dp["ChatGPT_labeling"]
    gpt4_labeling = dp["GPT-4_labeling"]
    mistral_labeling = dp["mistral_labeling"]
    human_labeling = dp["human_labeling"]
    num_bios += 1
    for fact in chatgpt_labeling:
        dict_chatgpt["#".join([dp["topic"].strip(), fact["atom"].strip()])] = fact["is_supported"]
    for fact in gpt4_labeling:
        dict_gpt4["#".join([dp["topic"].strip(), fact["atom"].strip()])] = fact["is_supported"]
    for fact in mistral_labeling:
        dict_mistral["#".join([dp["topic"].strip(), fact["atom"].strip()])] = fact["is_supported"]
    for fact in human_labeling:
        dict_human["#".join([dp["topic"].strip(), fact["atom"].strip()])] = (fact["is_supported"], fact)
        fact2entity["#".join([dp["topic"].strip(), fact["atom"].strip()])] = dp["topic"].strip()

dict_gemini = gemini_labeling_dict.copy()
dict_chatgpt_en = chatgpt_en_labeling_dict
dict_gemini_en = gemini_en_labeling_dict
dict_gpt4_en = gpt4_en_labeling_dict
dict_gpt4_npm_en = gpt4_npm_en_labeling_dict
dict_chatgpt_npm_en = chatgpt_npm_en_labeling_dict
dict_mistral_npm_en = mistral_npm_en_labeling_dict
dict_gemini_20_psg = gemini_20_psg_labeling_dict
dict_mistral_en = mistral_en_labeling_dict

num_fact_gpt4 = 0
num_fact_mistral = 0
num_fact_chatgpt = 0
human_vs_chatgpt = 0
human_vs_gpt4 = 0
human_vs_mistral = 0
different_all = 0
all_agree = 0
stat_human_vs_gpt4 = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}

stat_human_vs_mistral = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}

stat_human_vs_chatgpt = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}

stat_human_vs_gemini = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}


stat_human_vs_chatgpt_en = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}
stat_human_vs_gemini_en = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}
stat_human_vs_gpt4_en = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}
stat_human_vs_gpt4_npm_en = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}
stat_human_vs_mistral_npm_en = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}
stat_human_vs_chatgpt_npm_en = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}
stat_human_vs_gemini_20_psg = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}

stat_human_vs_mistral_en = {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0, "very freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "freq": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "medium": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}, "very rare": {"TT": 0, "TF": 0, "FF": 0, "FT": 0, "agree": 0, "disagree": 0, "num_facts": 0}}

import random
def reset(percent=20):
    return random.randrange(100) < percent
disagree_recheck_gpt4 = []
disagree_recheck_gemini = []
disagree_recheck_chatgpt = []
disagree_recheck_mistral = []
disagree_recheck_gpt4_en = []
disagree_recheck_gemini_en = []
disagree_recheck_chatgpt_en = []
disagree_recheck_mistral_en = []

for k in dict_human.keys():
    rarity = name2obj[dict_human[k][1]["figure"]]["cat"][0]
    if "human_label" not in figure_dict[fact2entity[k]]:
        figure_dict[fact2entity[k]]["human_label"] = {"T": 0, "F": 0}

    if dict_human[k][0]:
        figure_dict[fact2entity[k]]["human_label"]["T"] += 1
    else:
        figure_dict[fact2entity[k]]["human_label"]["F"] += 1 
    if k in dict_mistral_en:
        stat_human_vs_mistral_en["num_facts"] += 1
        if "human_mistral_en" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_mistral_en"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "mistral_en_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["mistral_en_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_mistral_en[k]:
            stat_human_vs_mistral_en["agree"] += 1
            
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_mistral_en"]["TT"] += 1
                figure_dict[fact2entity[k]]["mistral_en_label"]["T"] += 1
                stat_human_vs_mistral_en["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_mistral_en"]["FF"] += 1
                figure_dict[fact2entity[k]]["mistral_en_label"]["F"] += 1
                stat_human_vs_mistral_en["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_mistral_en.append(dict_human[k][1])
            stat_human_vs_mistral_en["disagree"] += 1
            
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_mistral_en"]["TF"] += 1
                figure_dict[fact2entity[k]]["mistral_en_label"]["F"] += 1
                stat_human_vs_mistral_en["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_mistral_en"]["FT"] += 1
                figure_dict[fact2entity[k]]["mistral_en_label"]["T"] += 1
                stat_human_vs_mistral_en["FT"] += 1
    else:
        print("no in mistral_en:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        try:
            print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        except:
            print("RECHECK FIGURE:", dict_human[k][1]["figure"])
    if k in dict_gpt4_npm_en:
        stat_human_vs_gpt4_npm_en["num_facts"] += 1
        if "human_gpt4_npm_en" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_gpt4_npm_en"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "gpt4_npm_en_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["gpt4_npm_en_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_gpt4_npm_en[k]:
            stat_human_vs_gpt4_npm_en["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gpt4_npm_en"]["TT"] += 1
                figure_dict[fact2entity[k]]["gpt4_npm_en_label"]["T"] += 1
                stat_human_vs_gpt4_npm_en["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gpt4_npm_en"]["FF"] += 1
                figure_dict[fact2entity[k]]["gpt4_npm_en_label"]["F"] += 1
                stat_human_vs_gpt4_npm_en["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_gpt4_en.append(dict_human[k][1])
            stat_human_vs_gpt4_npm_en["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gpt4_npm_en"]["TF"] += 1
                figure_dict[fact2entity[k]]["gpt4_npm_en_label"]["F"] += 1
                stat_human_vs_gpt4_npm_en["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gpt4_npm_en"]["FT"] += 1
                figure_dict[fact2entity[k]]["gpt4_npm_en_label"]["T"] += 1
                stat_human_vs_gpt4_npm_en["FT"] += 1
    else:
        print("no in gpt4_npm_en:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        # try:
        #     print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        # except:
        #     print("RECHECK FIGURE:", dict_human[k][1]["figure"])

    if k in dict_mistral_npm_en:
        stat_human_vs_mistral_npm_en["num_facts"] += 1
        if "human_mistral_npm_en" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_mistral_npm_en"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "mistral_npm_en_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["mistral_npm_en_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_mistral_npm_en[k]:
            stat_human_vs_mistral_npm_en["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_mistral_npm_en"]["TT"] += 1
                figure_dict[fact2entity[k]]["mistral_npm_en_label"]["T"] += 1
                stat_human_vs_mistral_npm_en["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_mistral_npm_en"]["FF"] += 1
                figure_dict[fact2entity[k]]["mistral_npm_en_label"]["F"] += 1
                stat_human_vs_mistral_npm_en["FF"] += 1
        else:
            # if reset(100):
            #     disagree_recheck_gpt4_en.append(dict_human[k][1])
            stat_human_vs_mistral_npm_en["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_mistral_npm_en"]["TF"] += 1
                figure_dict[fact2entity[k]]["mistral_npm_en_label"]["F"] += 1
                stat_human_vs_mistral_npm_en["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_mistral_npm_en"]["FT"] += 1
                figure_dict[fact2entity[k]]["mistral_npm_en_label"]["T"] += 1
                stat_human_vs_mistral_npm_en["FT"] += 1
    else:
        print("no in mistral_npm_en:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        # try:
        #     print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        # except:
        #     print("RECHECK FIGURE:", dict_human[k][1]["figure"])
    #dict_gemini_20_psg

    if k in dict_gemini_20_psg:
        stat_human_vs_gemini_20_psg["num_facts"] += 1
        if "human_gemini_20_psg" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_gemini_20_psg"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "gemini_20_psg_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["gemini_20_psg_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_gemini_20_psg[k]:
            stat_human_vs_gemini_20_psg["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gemini_20_psg"]["TT"] += 1
                figure_dict[fact2entity[k]]["gemini_20_psg_label"]["T"] += 1
                stat_human_vs_gemini_20_psg["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gemini_20_psg"]["FF"] += 1
                figure_dict[fact2entity[k]]["gemini_20_psg_label"]["F"] += 1
                stat_human_vs_gemini_20_psg["FF"] += 1
        else:
            # if reset(100):
            #     disagree_recheck_gpt4__en.append(dict_human[k][1])
            stat_human_vs_gemini_20_psg["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gemini_20_psg"]["TF"] += 1
                figure_dict[fact2entity[k]]["gemini_20_psg_label"]["F"] += 1
                stat_human_vs_gemini_20_psg["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gemini_20_psg"]["FT"] += 1
                figure_dict[fact2entity[k]]["gemini_20_psg_label"]["T"] += 1
                stat_human_vs_gemini_20_psg["FT"] += 1
    else:
        print("no in dict_gemini_20_psg:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        # try:
        #     print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        # except:
        #     print("RECHECK FIGURE:", dict_human[k][1]["figure"])
    if k in dict_chatgpt_npm_en:
        stat_human_vs_chatgpt_npm_en["num_facts"] += 1
        if "human_chatgpt_npm_en" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_chatgpt_npm_en"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "chatgpt_npm_en_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["chatgpt_npm_en_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_chatgpt_npm_en[k]:
            stat_human_vs_chatgpt_npm_en["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_chatgpt_npm_en"]["TT"] += 1
                figure_dict[fact2entity[k]]["chatgpt_npm_en_label"]["T"] += 1
                stat_human_vs_chatgpt_npm_en["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_chatgpt_npm_en"]["FF"] += 1
                figure_dict[fact2entity[k]]["chatgpt_npm_en_label"]["F"] += 1
                stat_human_vs_chatgpt_npm_en["FF"] += 1
        else:
            # if reset(100):
            #     disagree_recheck_gpt4__en.append(dict_human[k][1])
            stat_human_vs_chatgpt_npm_en["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_chatgpt_npm_en"]["TF"] += 1
                figure_dict[fact2entity[k]]["chatgpt_npm_en_label"]["F"] += 1
                stat_human_vs_chatgpt_npm_en["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_chatgpt_npm_en"]["FT"] += 1
                figure_dict[fact2entity[k]]["chatgpt_npm_en_label"]["T"] += 1
                stat_human_vs_chatgpt_npm_en["FT"] += 1
    else:
        print("no in chatgpt_npm_en:", k ,dict_human[k][1])
    if k in dict_gpt4_en:
        stat_human_vs_gpt4_en["num_facts"] += 1
        if "human_gpt4_en" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_gpt4_en"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "gpt4_en_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["gpt4_en_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_gpt4_en[k]:
            stat_human_vs_gpt4_en["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gpt4_en"]["TT"] += 1
                figure_dict[fact2entity[k]]["gpt4_en_label"]["T"] += 1
                stat_human_vs_gpt4_en["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gpt4_en"]["FF"] += 1
                figure_dict[fact2entity[k]]["gpt4_en_label"]["F"] += 1
                stat_human_vs_gpt4_en["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_gpt4_en.append(dict_human[k][1])
            stat_human_vs_gpt4_en["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gpt4_en"]["TF"] += 1
                figure_dict[fact2entity[k]]["gpt4_en_label"]["F"] += 1
                stat_human_vs_gpt4_en["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gpt4_en"]["FT"] += 1
                figure_dict[fact2entity[k]]["gpt4_en_label"]["T"] += 1
                stat_human_vs_gpt4_en["FT"] += 1
    else:
        print("no in gpt4_en:", k ,dict_human[k][1])
    
    if k in dict_gemini_en:
        stat_human_vs_gemini_en["num_facts"] += 1
        if "human_gemini_en" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_gemini_en"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "gemini_en_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["gemini_en_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_gemini_en[k]:
            stat_human_vs_gemini_en["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gemini_en"]["TT"] += 1
                figure_dict[fact2entity[k]]["gemini_en_label"]["T"] += 1
                stat_human_vs_gemini_en["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gemini_en"]["FF"] += 1
                figure_dict[fact2entity[k]]["gemini_en_label"]["F"] += 1
                stat_human_vs_gemini_en["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_gemini_en.append(dict_human[k][1])
            stat_human_vs_gemini_en["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gemini_en"]["TF"] += 1
                figure_dict[fact2entity[k]]["gemini_en_label"]["F"] += 1
                stat_human_vs_gemini_en["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gemini_en"]["FT"] += 1
                figure_dict[fact2entity[k]]["gemini_en_label"]["T"] += 1
                stat_human_vs_gemini_en["FT"] += 1
    else:
        print("no in gemini_en:", k ,dict_human[k][1])
        try:
            print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        except:
            print("RECHECK FIGURE:", dict_human[k][1]["figure"])
    if k in dict_chatgpt_en:
        stat_human_vs_chatgpt_en["num_facts"] += 1
        if "human_chatgpt_en" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_chatgpt_en"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "chatgpt_en_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["chatgpt_en_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_chatgpt_en[k]:
            stat_human_vs_chatgpt_en["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_chatgpt_en"]["TT"] += 1
                figure_dict[fact2entity[k]]["chatgpt_en_label"]["T"] += 1
                stat_human_vs_chatgpt_en["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_chatgpt_en"]["FF"] += 1
                figure_dict[fact2entity[k]]["chatgpt_en_label"]["F"] += 1
                stat_human_vs_chatgpt_en["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_chatgpt_en.append(dict_human[k][1])
            stat_human_vs_chatgpt_en["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_chatgpt_en"]["TF"] += 1
                figure_dict[fact2entity[k]]["chatgpt_en_label"]["F"] += 1
                stat_human_vs_chatgpt_en["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_chatgpt_en"]["FT"] += 1
                figure_dict[fact2entity[k]]["chatgpt_en_label"]["T"] += 1
                stat_human_vs_chatgpt_en["FT"] += 1
    else:
        print("no in chatgpt_en:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        try:
            print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        except:
            print("RECHECK FIGURE:", dict_human[k][1]["figure"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]]['model-atomic-facts'][dict_human[k][1]["fact #"]])




    if k in dict_gemini:
        stat_human_vs_gemini["num_facts"] += 1
        if "human_gemini" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_gemini"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "gemini_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["gemini_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_gemini[k]:
            stat_human_vs_gemini["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gemini"]["TT"] += 1
                figure_dict[fact2entity[k]]["gemini_label"]["T"] += 1
                stat_human_vs_gemini["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gemini"]["FF"] += 1
                figure_dict[fact2entity[k]]["gemini_label"]["F"] += 1
                stat_human_vs_gemini["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_gemini.append(dict_human[k][1])
            stat_human_vs_gemini["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gemini"]["TF"] += 1
                figure_dict[fact2entity[k]]["gemini_label"]["F"] += 1
                stat_human_vs_gemini["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gemini"]["FT"] += 1
                figure_dict[fact2entity[k]]["gemini_label"]["T"] += 1
                stat_human_vs_gemini["FT"] += 1
    else:
        print("no in gemini:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        try:
            print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        except:
            print("RECHECK FIGURE:", dict_human[k][1]["figure"])
    if k in dict_chatgpt:
        stat_human_vs_chatgpt["num_facts"] += 1
        if "human_chatgpt" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_chatgpt"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "chatgpt_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["chatgpt_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_chatgpt[k]:
            stat_human_vs_chatgpt["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_chatgpt"]["TT"] += 1
                figure_dict[fact2entity[k]]["chatgpt_label"]["T"] += 1
                stat_human_vs_chatgpt["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_chatgpt"]["FF"] += 1
                figure_dict[fact2entity[k]]["chatgpt_label"]["F"] += 1
                stat_human_vs_chatgpt["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_chatgpt.append(dict_human[k][1])
            stat_human_vs_chatgpt["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_chatgpt"]["TF"] += 1
                figure_dict[fact2entity[k]]["chatgpt_label"]["F"] += 1
                stat_human_vs_chatgpt["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_chatgpt"]["FT"] += 1
                figure_dict[fact2entity[k]]["chatgpt_label"]["T"] += 1
                stat_human_vs_chatgpt["FT"] += 1
    else:
        print("no in chatgpt:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        # print(len(name2obj[dict_human[k][1]["figure"]]["annotations"]))
        # print(dict_human[k][1]["sent #"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        try:
            print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        except:
            print("RECHECK FIGURE:", dict_human[k][1]["figure"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]]['model-atomic-facts'][dict_human[k][1]["fact #"]])
    if k in dict_gpt4:
        stat_human_vs_gpt4["num_facts"] += 1
        if "human_gpt4" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_gpt4"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "gpt4_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["gpt4_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_gpt4[k]:
            stat_human_vs_gpt4["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gpt4"]["TT"] += 1
                figure_dict[fact2entity[k]]["gpt4_label"]["T"] += 1
                stat_human_vs_gpt4["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gpt4"]["FF"] += 1
                figure_dict[fact2entity[k]]["gpt4_label"]["F"] += 1
                stat_human_vs_gpt4["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_gpt4.append(dict_human[k][1])
            stat_human_vs_gpt4["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_gpt4"]["TF"] += 1
                figure_dict[fact2entity[k]]["gpt4_label"]["F"] += 1
                stat_human_vs_gpt4["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_gpt4"]["FT"] += 1
                figure_dict[fact2entity[k]]["gpt4_label"]["T"] += 1
                stat_human_vs_gpt4["FT"] += 1
    else:
        print("no in gpt4:", k ,dict_human[k][1])
        try:
            print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        except:
            print("RECHECK FIGURE:", dict_human[k][1]["figure"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]]['model-atomic-facts'][dict_human[k][1]["fact #"]])
    if k in dict_mistral:
        stat_human_vs_mistral["num_facts"] += 1
        if "human_mistral" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["human_mistral"] = {"TT": 0, "FF": 0, "TF": 0, "FT": 0}
        if "mistral_label" not in figure_dict[fact2entity[k]]:
            figure_dict[fact2entity[k]]["mistral_label"] = {"T": 0, "F": 0}
        if dict_human[k][0] == dict_mistral[k]:
            stat_human_vs_mistral["agree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_mistral"]["TT"] += 1
                figure_dict[fact2entity[k]]["mistral_label"]["T"] += 1
                stat_human_vs_mistral["TT"] += 1
            else:
                figure_dict[fact2entity[k]]["human_mistral"]["FF"] += 1
                figure_dict[fact2entity[k]]["mistral_label"]["F"] += 1
                stat_human_vs_mistral["FF"] += 1
        else:
            if reset(100):
                disagree_recheck_mistral.append(dict_human[k][1])
            stat_human_vs_mistral["disagree"] += 1
            if dict_human[k][0]:
                figure_dict[fact2entity[k]]["human_mistral"]["TF"] += 1
                figure_dict[fact2entity[k]]["mistral_label"]["F"] += 1
                stat_human_vs_mistral["TF"] += 1
            else:
                figure_dict[fact2entity[k]]["human_mistral"]["FT"] += 1
                figure_dict[fact2entity[k]]["mistral_label"]["T"] += 1
                stat_human_vs_mistral["FT"] += 1
    else:
        print("no in mistral:", k ,dict_human[k][1])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"])
        try:
            print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]])
        except:
            print("RECHECK FIGURE:", dict_human[k][1]["figure"])
        # print(name2obj[dict_human[k][1]["figure"]]["annotations"][dict_human[k][1]["sent #"]]['model-atomic-facts'][dict_human[k][1]["fact #"]])
    # if dict_chatgpt[k] == dict_gpt4[k] == dict_mistral[k]:
    #     all_agree += 1
    # if dict_gpt4[k] != dict_human[k][0]:
    #     different_all += 1
        # print(dict_human[k][1])
# print()
print("len(dict_human.keys()):",len(dict_human.keys()))
print("len(dict_gemini.keys()):",len(dict_gemini.keys()))
print("len(dict_gpt4.keys()):",len(dict_gpt4.keys()))
print("len(dict_chatgpt.keys()):",len(dict_chatgpt.keys()))
print("len(dict_mistral.keys()):",len(dict_mistral.keys()))

print("FActScore by human:", sum([e["human_label"]["T"]/(e["human_label"]["T"]+e["human_label"]["F"]) for e in figure_dict.values() if "human_label" in e])/len([e["human_label"]["T"]/(e["human_label"]["T"]+e["human_label"]["F"]) for e in figure_dict.values() if "human_label" in e]))
print("FActScore by chatgpt_en:", sum([e["chatgpt_en_label"]["T"]/(e["chatgpt_en_label"]["T"]+e["chatgpt_en_label"]["F"]) for e in figure_dict.values() if "chatgpt_en_label" in e])/len([e["chatgpt_en_label"]["T"]/(e["chatgpt_en_label"]["T"]+e["chatgpt_en_label"]["F"]) for e in figure_dict.values() if "chatgpt_en_label" in e]))
print("FActScore by gemini_en:", sum([e["gemini_en_label"]["T"]/(e["gemini_en_label"]["T"]+e["gemini_en_label"]["F"]) for e in figure_dict.values() if "gemini_en_label" in e])/len([e["gemini_en_label"]["T"]/(e["gemini_en_label"]["T"]+e["gemini_en_label"]["F"]) for e in figure_dict.values() if "gemini_en_label" in e]))
print("FActScore by gpt4_en:", sum([e["gpt4_en_label"]["T"]/(e["gpt4_en_label"]["T"]+e["gpt4_en_label"]["F"]) for e in figure_dict.values() if "gpt4_en_label" in e])/len([e["gpt4_en_label"]["T"]/(e["gpt4_en_label"]["T"]+e["gpt4_en_label"]["F"]) for e in figure_dict.values() if "gpt4_en_label" in e]))
print("FActScore by gpt4_npm_en:", sum([e["gpt4_npm_en_label"]["T"]/(e["gpt4_npm_en_label"]["T"]+e["gpt4_npm_en_label"]["F"]) for e in figure_dict.values() if "gpt4_npm_en_label" in e])/len([e["gpt4_npm_en_label"]["T"]/(e["gpt4_npm_en_label"]["T"]+e["gpt4_npm_en_label"]["F"]) for e in figure_dict.values() if "gpt4_npm_en_label" in e]))
print("FActScore by chatgpt_npm_en:", sum([e["chatgpt_npm_en_label"]["T"]/(e["chatgpt_npm_en_label"]["T"]+e["chatgpt_npm_en_label"]["F"]) for e in figure_dict.values() if "chatgpt_npm_en_label" in e])/len([e["chatgpt_npm_en_label"]["T"]/(e["chatgpt_npm_en_label"]["T"]+e["chatgpt_npm_en_label"]["F"]) for e in figure_dict.values() if "chatgpt_npm_en_label" in e]))
print("FActScore by gemini_20_psg:", sum([e["gemini_20_psg_label"]["T"]/(e["gemini_20_psg_label"]["T"]+e["gemini_20_psg_label"]["F"]) for e in figure_dict.values() if "gemini_20_psg_label" in e])/len([e["gemini_20_psg_label"]["T"]/(e["gemini_20_psg_label"]["T"]+e["gemini_20_psg_label"]["F"]) for e in figure_dict.values() if "gemini_20_psg_label" in e]))

print("FActScore by mistral_npm_en:", sum([e["mistral_npm_en_label"]["T"]/(e["mistral_npm_en_label"]["T"]+e["mistral_npm_en_label"]["F"]) for e in figure_dict.values() if "mistral_npm_en_label" in e])/len([e["mistral_npm_en_label"]["T"]/(e["mistral_npm_en_label"]["T"]+e["mistral_npm_en_label"]["F"]) for e in figure_dict.values() if "mistral_npm_en_label" in e]))
print("FActScore by mistral_en:", sum([e["mistral_en_label"]["T"]/(e["mistral_en_label"]["T"]+e["mistral_en_label"]["F"]) for e in figure_dict.values() if "mistral_en_label" in e])/len([e["mistral_en_label"]["T"]/(e["mistral_en_label"]["T"]+e["mistral_en_label"]["F"]) for e in figure_dict.values() if "mistral_en_label" in e]))

print("FActScore by gemini:", sum([e["gemini_label"]["T"]/(e["gemini_label"]["T"]+e["gemini_label"]["F"]) for e in figure_dict.values() if "gemini_label" in e])/len([e["gemini_label"]["T"]/(e["gemini_label"]["T"]+e["gemini_label"]["F"]) for e in figure_dict.values() if "gemini_label" in e]))
print("FActScore by gpt4:", sum([e["gpt4_label"]["T"]/(e["gpt4_label"]["T"]+e["gpt4_label"]["F"]) for e in figure_dict.values() if "gpt4_label" in e])/len([e["gpt4_label"]["T"]/(e["gpt4_label"]["T"]+e["gpt4_label"]["F"]) for e in figure_dict.values() if "gpt4_label" in e]))
print("FActScore by chatgpt:", sum([e["chatgpt_label"]["T"]/(e["chatgpt_label"]["T"]+e["chatgpt_label"]["F"]) for e in figure_dict.values() if "chatgpt_label" in e])/len([e["chatgpt_label"]["T"]/(e["chatgpt_label"]["T"]+e["chatgpt_label"]["F"]) for e in figure_dict.values() if "chatgpt_label" in e]))
print("FActScore by mistral:", sum([e["mistral_label"]["T"]/(e["mistral_label"]["T"]+e["mistral_label"]["F"]) for e in figure_dict.values() if "mistral_label" in e])/len([e["mistral_label"]["T"]/(e["mistral_label"]["T"]+e["mistral_label"]["F"]) for e in figure_dict.values() if "mistral_label" in e]))








