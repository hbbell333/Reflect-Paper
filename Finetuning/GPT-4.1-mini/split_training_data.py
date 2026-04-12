import random
import json

random.seed(123)

input_file = "/Users/carolinezhang/Desktop/Reflect/GPT-SFT/pref_pairs.jsonl"

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

data = []
for line in lines:
    line = line.strip()
    if line: 
        data.append(json.loads(line))

print(f"Total samples: {len(data)}")

# Shuffle the data
random.shuffle(data)

split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
valid_data = data[split_idx:]

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(valid_data)}")

# Write training data
train_file = "pref_pairs_training.jsonl"
with open(train_file, 'w', encoding='utf-8') as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Write validation data
valid_file = "pref_pairs_valid.jsonl"
with open(valid_file, 'w', encoding='utf-8') as f:
    for item in valid_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

