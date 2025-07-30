import json

def convert_dialogues_to_sft_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    sft_samples = []

    for example in raw_data:
        dialogue = example["dialogues"]
        history = []
        for i, turn in enumerate(dialogue):
            role = turn["speaker"]
            content = turn["content"]


            if role == "therapist":
                strategy = turn.get("strategy", "")
                sample = {
                    "dialogue_history": history.copy(),  # deep copy
                    "strategy": strategy,
                    "response": content
                }
                sft_samples.append(sample)


            history.append({
                "role": role,
                "content": content
            })


    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in sft_samples:
            json.dump(item, f_out, ensure_ascii=False)
            f_out.write('\n')

    print(f"✅ Done! Converted {len(sft_samples)} samples saved to {output_file}")



if __name__ == "__main__":
    convert_dialogues_to_sft_format(r"...\val.json", r"...\sft_data_val.jsonl")
