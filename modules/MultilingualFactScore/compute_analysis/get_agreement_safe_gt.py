import os
from tqdm import tqdm
import json
import numpy as np
from scipy import stats

lang = "es"
safe_result_path = f"~/long-form-factuality/results/real_full_third_party_entities_with_human_annotated_{lang}_-correlation_vs_factscore.json"
print(lang)
disagreement_dict = {"gpt4": [], "gemini": []}
with open(safe_result_path) as f:
    for line in f:
        safe_dict = json.loads(line)
safe_label = {}
for instance in safe_dict["rate_facts"]["per_prompt_data"]:
    model_name = instance["model_name"]
    prompt = instance["prompt"]
    for fact in instance["checked_statements"]:
        text_fact = fact["atomic_fact"]
        if fact["annotation"] == "Supported":
            label = True
        elif fact["annotation"] == "Not Supported":
            label = False
        else:
            assert False, fact
        safe_label["#".join([model_name, prompt, text_fact])] = label


models = ["gemini", "gpt4"]
dir = f"~/long-form-factuality/third_party/entities_with_human_annotated/{lang}/"
safe_label_both = []
gt_label_both = []
stat_both = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
for model in models:
    file_path = os.path.join(dir, f"{model}.jsonl")
    with open(file_path) as f:
        safe_label_lst = []
        gt_label_lst = []
        stat = {"TT": 0, "TF": 0, "FT": 0, "FF": 0}
        print(file_path)
        for i, line in tqdm(enumerate(f)):
            dp = json.loads(line)
            tmp_safe = {"supported": 0, "irrelevant": 0, "not_supported": 0, "num_claims": 0}
            tmp_gt = {"supported": 0, "irrelevant": 0, "not_supported": 0, "num_claims": 0}
            for sent in dp["annotations"]:
                assert len(sent["human-atomic-facts-gt"]) == len(sent["human-atomic-facts"]), (sent["human-atomic-facts-gt"][0], sent["human-atomic-facts"][0])
                for gt_label in sent["human-atomic-facts-gt"]:
                    #assert gt_label["text"] == wiki_label["text"], (gt_label["text"], wiki_label["text"])
                # for gt_label in dp["annotations"]["human-atomic-facts-gt"]:
                    tmp_safe["num_claims"] += 1
                    tmp_gt["num_claims"] += 1
                    key = "#".join([model, dp["input"], gt_label["text"]])
                    if gt_label["label"] == "S" and safe_label[key]:
                        tmp_gt["supported"] += 1
                        tmp_safe["supported"] += 1
                        stat["TT"] += 1
                    elif gt_label["label"] == "S" and not safe_label[key]:
                        tmp_gt["supported"] += 1
                        tmp_safe["not_supported"] += 1
                        stat["TF"] += 1
                        disagreement_dict[model].append({"from": model, "sent": sent["text"], "topic": dp["topic"],"link": dp["link"],"item": gt_label, "reason": "", "true/false": "", "comment": ""})
                    elif gt_label["label"] == "NS" and safe_label[key]:
                        tmp_gt["not_supported"] += 1
                        tmp_safe["supported"] += 1
                        stat["FT"] += 1
                        disagreement_dict[model].append({"from": model, "sent": sent["text"], "topic": dp["topic"],"link": dp["link"],"item": gt_label, "reason": "", "true/false": "", "comment": ""})
                    elif gt_label["label"] == "NS" and not safe_label[key]:
                        tmp_gt["not_supported"] += 1
                        tmp_safe["not_supported"] += 1
                        stat["FF"] += 1
                    safe_label_lst.append(tmp_safe.copy())
                    gt_label_lst.append(tmp_gt.copy())
        safe_label_both.extend(safe_label_lst)
        gt_label_both.extend(gt_label_lst)
        for k in stat.keys():
            stat_both[k] += stat[k]
        print("Model:", model)
        print(stat)
        print("FActScore by human vs safe:", (stat["FT"] + stat["TT"])/(stat["TT"] + stat["FF"] + stat["TF"] + stat["FT"]))
        print("Agreement with human:", (stat["TT"] + stat["FF"])/(stat["TT"] + stat["FF"] + stat["TF"] + stat["FT"]))
        #stats.pearsonr(x, y)
        print("Support Pearson:", stats.pearsonr([e["supported"] for e in gt_label_lst], [e["supported"] for e in safe_label_lst]))
        print("Not Support Pearson:", stats.pearsonr([e["not_supported"] for e in gt_label_lst], [e["not_supported"] for e in safe_label_lst]))
        #stats.spearmanr
        print("Support Spearman:", stats.spearmanr([e["supported"] for e in gt_label_lst], [e["supported"] for e in safe_label_lst]))
        print("Not Support Spearman:", stats.spearmanr([e["not_supported"] for e in gt_label_lst], [e["not_supported"] for e in safe_label_lst]))
        print("#############"*10)
print("ALL")
print(stat_both)
print("Agreement with human:", (stat_both["TT"] + stat_both["FF"])/(stat_both["TT"] + stat_both["FF"] + stat_both["TF"] + stat_both["FT"]))
#stats.pearsonr(x, y)
print("Support Pearson:", stats.pearsonr([e["supported"] for e in gt_label_both], [e["supported"] for e in safe_label_both]))
print("Not Support Pearson:", stats.pearsonr([e["not_supported"] for e in gt_label_both], [e["not_supported"] for e in safe_label_both]))
                
print("Support Spearman:", stats.spearmanr([e["supported"] for e in gt_label_both], [e["supported"] for e in safe_label_both]))
print("Not Support Spearman:", stats.spearmanr([e["not_supported"] for e in gt_label_both], [e["not_supported"] for e in safe_label_both]))
print("#############"*10)