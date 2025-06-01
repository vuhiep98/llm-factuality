import sys
import os
import json
from pathlib import Path
from tqdm.auto import tqdm

root = Path(__file__).resolve().parents[2]
print(root)
sys.path.insert(0, str(root))
from modules.FActScore.factscore.retrieval import Retrieval, DocDB

if __name__ == "__main__":
    
    db = DocDB(
        db_path="/home/s2420414/fact-check/adobe/modules/FActScore/.cache/factscore/enwiki-20230401.db", 
        data_path="/home/s2420414/fact-check/adobe/modules/FActScore/.cache/factscore/enwiki-20230401.json"
    )
    retrieval = Retrieval(
        db=db, 
        cache_path="/home/s2420414/fact-check/adobe/modules/FActScore/.cache/factscore/retrieval-enwiki-20230401.json",
        embed_cache_path="/home/s2420414/fact-check/adobe/modules/FActScore/.cache/factscore/retrieval-enwiki-20230401.pkl",
        batch_size=256
    )
    
    fs_folder = root/"modules/FActScore/data/train/"
    
    for file in os.listdir(fs_folder):
        if file.endswith(".json"):
            fs_file = fs_folder/file
            fs_data = json.load(open(fs_file))
            for input in tqdm(fs_data, desc=file):
                if "factscores" in input:
                    for sentence, facts in input["factscores"].items():
                        for fact in facts:
                            passages = retrieval.get_passages(topic=input["topic"], question=fact["atom"], k=5)
                            for psg in passages:
                                psg["text"] = psg["text"].replace("<s>", "").replace("</s>", "")
                            fact["passages"] = passages
    
        with open(fs_file, "w") as f:
            json.dump(fs_data, f, indent=4)