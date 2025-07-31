import json
import random
from sklearn.model_selection import train_test_split


input_file = r'CBTSQ-CR/Data/SoCBTtalk_2000.json'


train_file = r'CBTSQ-CR/Data/train.json'
val_file = r'CBTSQ-CR/Data/val.json'


with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)


random.shuffle(data)


train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)


def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

save_to_file(train_data, train_file)
save_to_file(val_data, val_file)

