import json


input_file = r'...\processed_data_val.jsonl'

output_file = r'...\key_converted_dataset_val.json'


data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        entry['human'] = entry.pop('input')
        entry['assistant'] = entry.pop('output')
        data.append(entry)


with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
