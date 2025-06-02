import os
from tqdm import tqdm
import json
import numpy as np
from scipy import stats
lang = "bn"
models = ["gemini", "gpt4"]
dir = f"~/long-form-factuality/third_party/entities_with_human_annotated/{lang}"
modelling_label_both = []
gt_label_both = []
stat_both = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
disagreement_dict = {"gpt4": [], "gemini": []}
for model in models:
    file_path = os.path.join(dir, f"{model}.jsonl")
    # _trans_after_retrieval
    label_path = f"~/FActScore/data/to_evaluate/label/{lang}_{model}_label_by_Gemini-Pro_trans_after_retrieval.json"
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
                for gt_label in sent["human-atomic-facts"]:
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
                        elif gt_label["label"] == "S" and not label_dict[k]:
                            tmp_gt["supported"] += 1
                            tmp_modelling["not_supported"] += 1
                            stat["TF"] += 1                            
                            disagreement_dict[model].append({"from": model, "sent": sent["text"], "topic": dp["topic"],"link": dp["link"],"item": gt_label, "reason": "", "true/false": "", "comment": ""})
                        elif gt_label["label"] == "NS" and label_dict[k]:
                            tmp_gt["not_supported"] += 1
                            tmp_modelling["supported"] += 1
                            stat["FT"] += 1
                            disagreement_dict[model].append({"from": model, "sent": sent["text"], "topic": dp["topic"],"link": dp["link"],"item": gt_label, "reason": "", "true/false": "", "comment": ""})
                        elif gt_label["label"] == "NS" and not label_dict[k]:
                            tmp_gt["not_supported"] += 1
                            tmp_modelling["not_supported"] += 1
                            stat["FF"] += 1
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
