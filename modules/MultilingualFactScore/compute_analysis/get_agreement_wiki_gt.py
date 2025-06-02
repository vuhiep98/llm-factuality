import os
from tqdm import tqdm
import json
import numpy as np
from scipy import stats
lang = "bn"
models = ["gemini", "gpt4"]
dir = f"~/long-form-factuality/third_party/entities_with_human_annotated/{lang}"
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
                    elif gt_label["label"] == "NS" and wiki_label["label"] == "S":
                        tmp_gt["not_supported"] += 1
                        tmp_one_wiki["supported"] += 1
                        stat["FT"] += 1
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
