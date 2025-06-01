import subprocess
import os
from pathlib import Path

unlabeled_dir = Path(__file__).parents[2]/"modules/FActScore/factscore_data/unlabeled"
llms = [file.split(".")[0] for file in os.listdir(unlabeled_dir) if file.endswith(".jsonl")]
for llm in llms:
    if llm != "InstructGPT":
        subprocess.run(["python", "derive_training_data_from_factscore_output.py", "--llm", llm, "--threshold", "0.2"])