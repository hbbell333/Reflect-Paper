import json
from openai import OpenAI
import os
from tqdm import tqdm

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ICL instructions
with open('/Users/carolinezhang/Desktop/Reflect/GPT-SFT/safeRLHF_ICL.txt', 'r') as f:
    icl_instructions = f.read().strip()

with open('/Users/carolinezhang/Desktop/Reflect/safeRLHF_500.json', 'r') as f:
    safeRLHF_500 = json.load(f)

# Config
GENERATE_WITH_ICL = True
GENERATE_NO_ICL = True

# Open output files and initialize JSON arrays
output_file_icl = '/Users/carolinezhang/Desktop/Reflect/GPT-SFT/New_DPO_GPT_results_ICL.json'
output_file_no_icl = '/Users/carolinezhang/Desktop/Reflect/GPT-SFT/New_DPO_GPT_results_noICL.json'

if GENERATE_WITH_ICL:
    with open(output_file_icl, 'w', encoding='utf-8') as outfile:
        outfile.write('[\n')

if GENERATE_NO_ICL:
    with open(output_file_no_icl, 'w', encoding='utf-8') as outfile:
        outfile.write('[\n')

for i, prompt in enumerate(tqdm(safeRLHF_500, desc="Processing prompts")):
    user_prompt = prompt["prompt"]

    messages_with_ICL = [
        {"role": "developer", "content": icl_instructions},
        {"role": "user", "content": user_prompt}
    ]

    messages_no_ICL = [{"role": "user", "content": user_prompt}]

    # Generate response with ICL if enabled
    if GENERATE_WITH_ICL:
        response_icl = client.chat.completions.create(
            model="ft:gpt-4.1-mini-2025-04-14:duke-alignment:new-dpo:CoNSoWDC",
            messages=messages_with_ICL,
            temperature=0.7,
            max_tokens=1000
        )
        
        conversation_icl = [[{'role':'user', 'content': user_prompt}, {'role':'assistant', 'content':response_icl.choices[0].message.content}]]
        with open(output_file_icl, 'a', encoding='utf-8') as outfile:
            if i > 0:
                outfile.write(',\n')
            conversation_json = json.dumps(conversation_icl, ensure_ascii=False, indent=4)
            lines = conversation_json.split('\n')
            indented_lines = ['    ' + line for line in lines]
            outfile.write('\n'.join(indented_lines))

    # Generate response without ICL if enabled
    if GENERATE_NO_ICL:
        response_no_icl = client.chat.completions.create(
            model="ft:gpt-4.1-mini-2025-04-14:duke-alignment:new-dpo:CoNSoWDC",
            messages=messages_no_ICL,
            temperature=0.7,
            max_tokens=1000
        )
        
        conversation_no_icl = [[{'role':'user', 'content': user_prompt}, {'role':'assistant', 'content':response_no_icl.choices[0].message.content}]]
        with open(output_file_no_icl, 'a', encoding='utf-8') as outfile:
            if i > 0:
                outfile.write(',\n')
            conversation_json = json.dumps(conversation_no_icl, ensure_ascii=False, indent=4)
            lines = conversation_json.split('\n')
            indented_lines = ['    ' + line for line in lines]
            outfile.write('\n'.join(indented_lines))
    
    status_parts = []
    if GENERATE_WITH_ICL:
        status_parts.append("with ICL")
    if GENERATE_NO_ICL:
        status_parts.append("no ICL")
    print(f"Saved conversation {i+1} ({', '.join(status_parts)})")

if GENERATE_WITH_ICL:
    with open(output_file_icl, 'a', encoding='utf-8') as outfile:
        outfile.write('\n]')
    print(f"All conversations with ICL saved to {output_file_icl}")

if GENERATE_NO_ICL:
    with open(output_file_no_icl, 'a', encoding='utf-8') as outfile:
        outfile.write('\n]')
    print(f"All conversations without ICL saved to {output_file_no_icl}")
    