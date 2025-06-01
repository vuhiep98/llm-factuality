import os
from tqdm import tqdm
import json
import numpy as np
from scipy import stats
lang = "bn"
models = ["gemini", "gpt4"]
dir = f"~/long-form-factuality/third_party/entities_with_human_annotated/{lang}"
disagreement_dict = {"gpt4": [], "gemini": []}
wiki_label_both = []
gt_label_both = []
stat_both = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
for model in models:
    file_path = os.path.join(dir, f"{model}.jsonl")
    with open(file_path) as f:
        wiki_one_label_lst = []
        gt_label_lst = []
        stat = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
        for i, line in tqdm(enumerate(f)):
            dp = json.loads(line)
            tmp_one_wiki = {"supported": 0, "irrelevant": 0, "not_supported": 0, "num_claims": 0}
            tmp_gt = {"supported": 0, "irrelevant": 0, "not_supported": 0, "num_claims": 0}
            for sent in dp["annotations"]:
                assert len(sent["human-atomic-facts-gt"]) == len(sent["human-atomic-facts"]), (sent["human-atomic-facts-gt"][0], sent["human-atomic-facts"][0])
                for gt_label, wiki_label in zip(sent["human-atomic-facts-gt"], sent["human-atomic-facts"]):
                    assert gt_label["text"] == wiki_label["text"], (gt_label["text"], wiki_label["text"])
                # for gt_label in dp["annotations"]["human-atomic-facts-gt"]:
                    tmp_one_wiki["num_claims"] += 1
                    tmp_gt["num_claims"] += 1
                    if gt_label["label"] == "S" and wiki_label["label"] == "S":
                        tmp_gt["supported"] += 1
                        tmp_one_wiki["supported"] += 1
                        stat["TT"] += 1
                    elif gt_label["label"] == "S" and wiki_label["label"] == "NS":
                        tmp_gt["supported"] += 1
                        tmp_one_wiki["not_supported"] += 1
                        stat["TF"] += 1
                        disagreement_dict[model].append({"from": model, "atom": gt_label["text"], "sent": sent["text"], "topic": dp["topic"],"link": dp["link"],"item": gt_label, "Wiki_label": wiki_label["label"],"reason": "", "true/false": "", "comment": ""})
                    elif gt_label["label"] == "NS" and wiki_label["label"] == "S":
                        tmp_gt["not_supported"] += 1
                        tmp_one_wiki["supported"] += 1
                        stat["FT"] += 1
                        disagreement_dict[model].append({"from": model, "atom": gt_label["text"], "sent": sent["text"], "topic": dp["topic"],"link": dp["link"],"item": gt_label, "Wiki_label": wiki_label["label"],"reason": "", "true/false": "", "comment": ""})
                    elif gt_label["label"] == "NS" and wiki_label["label"] == "NS":
                        tmp_gt["not_supported"] += 1
                        tmp_one_wiki["not_supported"] += 1
                        stat["FF"] += 1
                    wiki_one_label_lst.append(tmp_one_wiki.copy())
                    gt_label_lst.append(tmp_gt.copy())
        wiki_label_both.extend(wiki_one_label_lst)
        gt_label_both.extend(gt_label_lst)
        for k in stat.keys():
            stat_both[k] += stat[k]
        print("Model:", model)
        print(stat)
        print("FActScore by human vs the Internet:", (stat["TT"] + stat["TF"])/(stat["TT"] + stat["FF"] + stat["TF"] + stat["FT"]))
        print("FActScore by human vs wiki:", (stat["TT"] + stat["FT"])/(stat["TT"] + stat["FF"] + stat["TF"] + stat["FT"]))
        print("Agreement with human:", (stat["TT"] + stat["FF"])/(stat["TT"] + stat["FF"] + stat["TF"] + stat["FT"]))
        #stats.pearsonr(x, y)
        print("Support Pearson:", stats.pearsonr([e["supported"] for e in gt_label_lst], [e["supported"] for e in wiki_one_label_lst]))
        print("Not Support Pearson:", stats.pearsonr([e["not_supported"] for e in gt_label_lst], [e["not_supported"] for e in wiki_one_label_lst]))
        #stats.spearmanr
        print("Support Spearman:", stats.spearmanr([e["supported"] for e in gt_label_lst], [e["supported"] for e in wiki_one_label_lst]))
        print("Not Support Spearman:", stats.spearmanr([e["not_supported"] for e in gt_label_lst], [e["not_supported"] for e in wiki_one_label_lst]))
        print("#############"*10)
print("ALL")
print(stat_both)
print("Agreement with human:", (stat_both["TT"] + stat_both["FF"])/(stat_both["TT"] + stat_both["FF"] + stat_both["TF"] + stat_both["FT"]))
#stats.pearsonr(x, y)
print("Support Pearson:", stats.pearsonr([e["supported"] for e in gt_label_both], [e["supported"] for e in wiki_label_both]))
print("Not Support Pearson:", stats.pearsonr([e["not_supported"] for e in gt_label_both], [e["not_supported"] for e in wiki_label_both]))
                
print("Support Spearman:", stats.spearmanr([e["supported"] for e in gt_label_both], [e["supported"] for e in wiki_label_both]))
print("Not Support Spearman:", stats.spearmanr([e["not_supported"] for e in gt_label_both], [e["not_supported"] for e in wiki_label_both]))
print("#############"*10)








models = ["gemini", "gpt4"]
dir = f"~/long-form-factuality/third_party/entities_with_human_annotated/{lang}"
modelling_label_both = []
gt_label_both = []
stat_both = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
agreement_dict = {"gpt4": [], "gemini": []}
for model in models:
    file_path = os.path.join(dir, f"{model}.jsonl")
    # _trans_after_retrieval
    label_path = f"~/FActScore/data/to_evaluate/label/{lang}_{model}_label_by_Gemini-Pro_gen_ks.json"
    with open(label_path) as f:
        for line in f:
            label_dict = json.loads(line)
    with open(file_path) as f:
        modelling_label_lst = []
        gt_label_lst = []
        stat = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
        for i, line in tqdm(enumerate(f)):
            dp = json.loads(line)
            tmp_modelling = {"supported": 0, "irrelevant": 0, "not_supported": 0, "num_claims": 0}
            tmp_gt = {"supported": 0, "irrelevant": 0, "not_supported": 0, "num_claims": 0}
            for sent in dp["annotations"]:
                for gt_label in sent["human-atomic-facts-gt"]:
                    if not gt_label["label"]:
                        print(gt_label)
                # for gt_label in dp["annotations"]["human-atomic-facts-gt"]:
                    k = "#".join([dp["topic"], gt_label["text"]])
                    if k in label_dict:
                        tmp_modelling["num_claims"] += 1
                        tmp_gt["num_claims"] += 1
                        if gt_label["label"] == "S" and label_dict[k]:
                            tmp_gt["supported"] += 1
                            tmp_modelling["supported"] += 1
                            stat["TT"] += 1
                            agreement_dict[model].append({"from": model, "sent": sent["text"], "topic": dp["topic"],"link": dp["link"], "ks_label": label_dict[k], "item": gt_label, "reason": "", "true/false": "", "comment": ""})
                        elif gt_label["label"] == "S" and not label_dict[k]:
                            tmp_gt["supported"] += 1
                            tmp_modelling["not_supported"] += 1
                            stat["TF"] += 1                            
                            
                        elif gt_label["label"] == "NS" and label_dict[k]:
                            tmp_gt["not_supported"] += 1
                            tmp_modelling["supported"] += 1
                            stat["FT"] += 1
                            
                        elif gt_label["label"] == "NS" and not label_dict[k]:
                            tmp_gt["not_supported"] += 1
                            tmp_modelling["not_supported"] += 1
                            stat["FF"] += 1
                            agreement_dict[model].append({"from": model, "sent": sent["text"], "topic": dp["topic"],"link": dp["link"], "ks_label": label_dict[k], "item": gt_label, "reason": "", "true/false": "", "comment": ""})
                        modelling_label_lst.append(tmp_modelling.copy())
                        gt_label_lst.append(tmp_gt.copy())
                    else:
                        tmp_modelling["num_claims"] += 1
                        tmp_gt["num_claims"] += 1
                        if gt_label["label"] == "S":
                            tmp_gt["supported"] += 1
                            tmp_modelling["not_supported"] += 1
                            stat["TF"] += 1
                        elif gt_label["label"] == "NS":
                            tmp_gt["not_supported"] += 1
                            tmp_modelling["not_supported"] += 1
                            stat["FF"] += 1
                        modelling_label_lst.append(tmp_modelling.copy())
                        gt_label_lst.append(tmp_gt.copy())
        modelling_label_both.extend(modelling_label_lst)
        gt_label_both.extend(gt_label_lst)
        for k in stat.keys():
            stat_both[k] += stat[k]
        print("Model:", model)
        summ = stat["TT"] + stat["FF"] + stat["FT"] + stat["TF"]
        # print({"TT": (), "TF": (), "FT": (), "TF": ()})
        print(stat)
        print("FActScore by human vs model with wiki: %.2f" % (100*(stat["FT"] + stat["TT"])/(stat["TT"] + stat["FF"] + stat["TF"] + stat["FT"])))
        print("Agreement with human: %.2f" % ((stat["TT"] + stat["FF"])/(stat["TT"] + stat["FF"] + stat["TF"] + stat["FT"])*100))
        #stats.pearsonr(x, y)
        print("Support Pearson:", stats.pearsonr([e["supported"] for e in gt_label_lst], [e["supported"] for e in modelling_label_lst]))
        print("Not Support Pearson:", stats.pearsonr([e["not_supported"] for e in gt_label_lst], [e["not_supported"] for e in modelling_label_lst]))
        #stats.spearmanr
        print("Support Spearman:", stats.spearmanr([e["supported"] for e in gt_label_lst], [e["supported"] for e in modelling_label_lst]))
        print("Not Support Spearman:", stats.spearmanr([e["not_supported"] for e in gt_label_lst], [e["not_supported"] for e in modelling_label_lst]))
        print("#############"*10)
print("ALL")
print(stat_both)
print("Agreement with human: %.2f" % ((stat_both["TT"] + stat_both["FF"])/(stat_both["TT"] + stat_both["FF"] + stat_both["TF"] + stat_both["FT"])*100))
#stats.pearsonr(x, y)
print("Support Pearson:", stats.pearsonr([e["supported"] for e in gt_label_both], [e["supported"] for e in modelling_label_both]))
print("Not Support Pearson:", stats.pearsonr([e["not_supported"] for e in gt_label_both], [e["not_supported"] for e in modelling_label_both]))
                
print("Support Spearman:", stats.spearmanr([e["supported"] for e in gt_label_both], [e["supported"] for e in modelling_label_both]))
print("Not Support Spearman:", stats.spearmanr([e["not_supported"] for e in gt_label_both], [e["not_supported"] for e in modelling_label_both]))
print("#############"*10)


disagree_wiki = {}
agree_safe = {}

for model in disagreement_dict:
    for e in disagreement_dict[model]:

        disagree_wiki["#".join([e["topic"], e["item"]["text"]])] = e.copy() 


for model in agreement_dict:
    for e in agreement_dict[model]:

        agree_safe["#".join([e["topic"], e["item"]["text"]])] = e.copy() 


for k in disagree_wiki:
    if k in agree_safe:
        print(disagree_wiki[k])
