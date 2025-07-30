import json


def convert_data(input_file_path, output_file_path):
    try:

        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            original_data = json.load(input_file)


        converted_data = []
        for sample in original_data:
            if "human" in sample and "assistant" in sample:
                new_sample = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": sample["human"]
                        },
                        {
                            "from": "assistant",
                            "value": sample["assistant"]
                        }
                    ]
                }
                converted_data.append(new_sample)
            else:
                print(f"skip：{sample}")


        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            json.dump(converted_data, output_file, indent=4, ensure_ascii=False)

        print(f"saved to {output_file_path}")

    except FileNotFoundError:
        print(f"error：file {input_file_path} not find.")
    except json.JSONDecodeError:
        print(f"error：file {input_file_path} not JSON")
    except Exception as e:
        print(f"error：{e}")



input_file_path = r'...\key_converted_dataset_val.json'
output_file_path = r'...\changed_val.json'


convert_data(input_file_path, output_file_path)