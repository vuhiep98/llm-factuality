from factscore.retrieval import DocDB, Retrieval


langs = ["es", "ar", "bn"]

for lang in langs:
    print(lang)
    # "~/projects/factscore/real_evaluation/en_arwiki_100.jsonl"
    data_path = f"~/projects/factscore/real_evaluation/en_{lang}wiki_100.jsonl"
    db_path = f"~/projects/factscore/real_evaluation/en_{lang}wiki_100.db"
    print(data_path)
    db = DocDB(db_path=db_path, data_path=data_path)