# create db from jsonl file
from factscore.retrieval import DocDB, Retrieval
print("AAAAAAAAAAAAAA")
db_path = "~/projects/factscore/filter_mt5.db"
data_path = "~/FActScore/data/filter.jsonl"
DocDB(db_path=db_path, data_path=data_path)


