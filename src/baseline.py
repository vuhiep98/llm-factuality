import argparse

from src.fact_reasoning import FactReasoner

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str)
    parser.add_argument("--model_path", type=str)

    args = parser.parse_args()

    fact_reasoner = FactReasoner(model_path=args.model_path)
    factscore = fact_reasoner.factual_reasoning(args.input_file)

    print(f"FActSCORE for {args.input_file.split('/')[-1]}: {factscore*100:.2f} %")