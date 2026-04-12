import json
from openai import OpenAI
from tqdm import tqdm
import os 
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

with open('/Users/carolinezhang/Desktop/Reflect/Size Ablation/safeRLHF_ICL.txt', 'r') as infile:
    safeRLHF_ICL = infile.read()

with open('/Users/carolinezhang/Desktop/Reflect/safeRLHF_500.json', 'r') as infile:
    safeRLHF_500 = json.load(infile)

outputs = []

for item in tqdm(safeRLHF_500):
    prompt = item["prompt"]
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "developer", "content": safeRLHF_ICL},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    outputs.append([[{'role':'user', 'content': prompt}, {'role':'assistant', 'content':response.choices[0].message.content}]])

with open('/Users/carolinezhang/Desktop/Reflect/Size Ablation/GPT_nano_ICL_base_responses.json', 'w', encoding='utf-8') as outfile:
    json.dump(outputs, outfile, ensure_ascii=False, indent=4)
    
    