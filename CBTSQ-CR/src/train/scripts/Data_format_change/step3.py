import json
from tqdm import tqdm


def format_sample(example):

    history = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}"
                         for turn in example["dialogue_history"]])
    input_prompt = f"{history}\nStrategy: {example['strategy']}\nTherapist:"
    return {
        "input": input_prompt,
        "output": example["response"]
    }


def process_dataset(input_file, output_file):

    with open(input_file, 'r', encoding='utf-8') as fin, \
            open(output_file, 'w', encoding='utf-8') as fout:

        for line in tqdm(fin, desc="Processing samples"):
            try:
                data = json.loads(line.strip())


                assert "dialogue_history" in data, "Missing dialogue_history"
                assert "strategy" in data, "Missing strategy"
                assert "response" in data, "Missing response"


                formatted = format_sample(data)


                fout.write(json.dumps(formatted, ensure_ascii=False) + "\n")

            except Exception as e:
                print(f"Error processing line: {line.strip()}\nError: {str(e)}")
                continue


if __name__ == "__main__":

    input_file = r"...\sft_data_val.jsonl"
    output_file = r"...\processed_data_val.jsonl"


    print("Starting data processing...")
    process_dataset(input_file, output_file)

    print(f"Processing complete! Saved to {output_file}")
    print("Sample preview:")


    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            print(json.loads(line.strip()))