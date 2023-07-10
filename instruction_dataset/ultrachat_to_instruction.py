import json
import tqdm
import os
import matplotlib.pyplot as plt
from typing import List

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor


filtered_keywords = [
    'image', 'picture', 'visual', 'photo', "i'm sorry", 'i am sorry', 'sorry, ', 'ai language model', '<no input>', '<noinput>', '<nooutput>', 'gpt model', 'provide more context or details', 'content policies', 'content policy', 'have enough information to', 'i do not have access to', 'no information provided to', 'cannot answer this question', 'without additional', 'please provide more', 'do not have the ability', 'cannot generate'
]

class InstructionInputOutputProcessor(DataProcessor):
    def __init__(self, min_length=0, max_length=float('inf')):
        super().__init__()
        self.labels = None
        self.min_length = min_length
        self.max_length = max_length

    def get_examples(self, data_path: str) -> List[InputExample]:
        examples = []
        j = 0
        
        all_sample_lengths = []
        all_sample_turns = []

        with open(data_path) as f:
            for line in tqdm.tqdm(f.readlines(), desc=f"Processing {data_path}"):
                if line.strip():
                    data = json.loads(line)
                    id_ = data["id"]
                    dialogue = data["data"]
                    dialogue_with_tags = []
                    instruction = ''
                    user_tag = '\n### Human: '
                    ai_tag = '\n### Assistant: '
                    for i, text in enumerate(dialogue):
                        tag = user_tag if i % 2 == 0 else ai_tag
                        if i==0:
                            tag = ''
                        dialogue_with_tags.append(tag + text)

                    final_user_response = dialogue_with_tags[-2]
                    final_ai_response = dialogue_with_tags[-1]

                    contains_keywords = False
                    for keyword in filtered_keywords:
                        if keyword in final_ai_response.replace('\n### Assistant: ', '').lower():
                            contains_keywords = True

                    for k in range(2, len(dialogue_with_tags) + 1, 2):
                        input_text = '\n'.join(dialogue_with_tags[:k])
                        all_sample_lengths.append(len(input_text))
                        all_sample_turns.append(input_text.count('\n') + 1)

                        if self.min_length <= len(input_text + '\n' + final_user_response) <= self.max_length and not contains_keywords:
                            example = InputExample(guid=str(j), text_a=instruction, text_b=input_text + '\n' + final_user_response, tgt_text=final_ai_response[len(ai_tag):])
                            examples.append(example)
                            j += 1

        return examples, all_sample_lengths, all_sample_turns



input_jsons = [
    'ultrachat_existent_material_release_230420_Assistance on Existent Materials [Part I].json',
    'ultrachat_material_release_230412_Writing and Creation [Part I].json',
    'ultrachat_material_release_230417_Writing and Creation [Part II].json',
    'ultrachat_release_230407_Questions about the World [Part I + Part II].json',
]
max_length=1536
# max_length=1024
output_data = []

total_samples = 0
sample_lengths = []
sample_turns = []

for input_json in input_jsons:
    input_json = os.path.join('/cpfs/user/chendelong/instruction_tuning_dataset/ultra_chat', input_json)
    processor = InstructionInputOutputProcessor(min_length=2, max_length=max_length)
    examples, all_lengths, all_turns = processor.get_examples(input_json)
    total_samples += len(all_lengths)

    sample_lengths.extend(all_lengths)
    sample_turns.extend(all_turns)

    for example in examples:
        output_data.append({
            "input": example.text_a + ' ' + example.text_b,
            "output": example.tgt_text
        })

os.makedirs('converted_datasets/ultra_chat/', exist_ok=True)
with open(f'converted_datasets/ultra_chat/ultra_chat_max{max_length}.json', "w") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
