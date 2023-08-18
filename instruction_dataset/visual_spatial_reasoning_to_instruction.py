import json
import os
import random

question_answer_templates = [
    ("Is the statement '{}' correct?", ("Yes, it's correct.", "No, it's incorrect.")),
    ("Does the description '{}' hold true?", ("Yes, it holds true.", "No, it doesn't hold true.")),
    ("Can we confirm that '{}' is accurate?", ("Yes, we can confirm it's accurate.", "No, we cannot confirm it's accurate.")),
    ("Is it true to say that '{}'", ("Yes, it's true.", "No, it's false.")),
    ("Is the assertion '{}' valid?", ("Yes, it's valid.", "No, it's invalid.")),
    ("Is the given information '{}' reliable?", ("Yes, it's reliable.", "No, it's unreliable.")),
    ("Can we verify that '{}' is true?", ("Yes, we can verify it's true.", "No, we cannot verify it's true.")),
    ("Is it correct to assume that '{}'", ("Yes, it's correct to assume that.", "No, it's incorrect to assume that.")),
    ("Is the claim '{}' accurate?", ("Yes, the claim is accurate.", "No, the claim is inaccurate.")),
    ("Is the following statement true: '{}'?", ("Yes, the statement is true.", "No, the statement is false.")),
    ("Regarding '{}', is this true or false?", ("It's true.", "It's false.")),
    ("Is '{}' a true statement?", ("Yes, it's a true statement.", "No, it's a false statement.")),
    ("Is the sentence '{}' correct?", ("Yes, the sentence is correct.", "No, the sentence is incorrect.")),
    ("Is it factual that '{}'", ("Yes, it's factual.", "No, it's not factual.")),
    ("Can we accept '{}' as true?", ("Yes, we can accept it as true.", "No, we cannot accept it as true.")),
    ("Is there truth in the statement '{}'", ("Yes, there is truth in the statement.", "No, there is no truth in the statement.")),
    ("Is the declaration '{}' true?", ("Yes, the declaration is true.", "No, the declaration is false.")),
    ("Would you say '{}' is true?", ("Yes, I would say it's true.", "No, I would say it's false.")),
    ("Is it the case that '{}'", ("Yes, it's the case.", "No, it's not the case.")),
    ("Is the following accurate: '{}'?", ("Yes, it's accurate.", "No, it's not accurate.")),
    # Variations that start with the statement
    ("{}. Is that true?", ("Yes, that's true.", "No, that's not true.")),
    ("{}. Can we confirm this?", ("Yes, we can confirm this.", "No, we cannot confirm this.")),
    ("{}. Is this accurate?", ("Yes, this is accurate.", "No, this is not accurate.")),
    ("{}. Is this statement correct?", ("Yes, this statement is correct.", "No, this statement is incorrect.")),
    ("{}. Is this true?", ("Yes, this is true.", "No, this is not true.")),
    ("{}. Is this valid?", ("Yes, this is valid.", "No, this is not valid.")),
    ("{}. Is this reliable?", ("Yes, this is reliable.", "No, this is not reliable.")),
    ("{}. Can we verify the truth?", ("Yes, we can verify the truth.", "No, we cannot verify the truth.")),
    ("{}. Can we assume this?", ("Yes, we can assume this.", "No, we cannot assume this.")),
    ("{}. Is this claim accurate?", ("Yes, this claim is accurate.", "No, this claim is not accurate.")),
    ("{}. True or false?", ("True.", "False.")),
    ("{}. Is this a true statement?", ("Yes, this is a true statement.", "No, this is a false statement.")),
    ("{}. Is this sentence correct?", ("Yes, this sentence is correct.", "No, this sentence is not correct.")),
    ("{}. Is it factual?", ("Yes, it's factual.", "No, it's not factual.")),
    ("{}. Can we accept this as true?", ("Yes, we can accept this as true.", "No, we cannot accept this as true.")),
    ("{}. Is there truth in this statement?", ("Yes, there is truth in this statement.", "No, there is no truth in this statement.")),
    ("{}. Is the declaration true?", ("Yes, the declaration is true.", "No, the declaration is not true.")),
    ("{}. Would you say this is true?", ("Yes, I would say this is true.", "No, I would say this is not true.")),
    ("{}. Is it the case?", ("Yes, it's the case.", "No, it's not the case.")),
    ("{}. Is the following accurate?", ("Yes, the following is accurate.", "No, the following is not accurate."))
]

def visual_spatial_reasoning_to_instruction(input_dir, output_file):
    def process_file(input_file):
        processed_data = []

        with open(input_file, 'r') as f:
            for line in f:
                data_point = json.loads(line)
                caption = data_point["caption"]
                label = data_point["label"]
                if label==0:
                    label=1
                elif label==1:
                    label=0
                else:
                    raise ValueError("Label is not 0 or 1")

                template = random.choice(question_answer_templates)
                question = template[0].format(caption[:-1])
                answer = template[1][label]

                processed_data.append({
                    "input": f"<img_path>/cpfs/user/chendelong/instruction_tuning_dataset/visual-spatial-reasoning/data/images/{data_point['image']}<img_path> {question}",
                    "output": answer
                })

        return processed_data

    train_data = process_file(os.path.join(input_dir, "train.jsonl"))
    dev_data = process_file(os.path.join(input_dir, "dev.jsonl"))
    test_data = process_file(os.path.join(input_dir, "test.jsonl"))

    # all_data = train_data + dev_data + test_data

    with open(output_file + 'train.json', "w") as f:
        json.dump(train_data, f, indent=2)
    with open(output_file + 'dev.json', "w") as f:
        json.dump(dev_data, f, indent=2)
    with open(output_file + 'test.json', "w") as f:
        json.dump(test_data, f, indent=2)

# Example usage:
os.makedirs('converted_datasets/visual-spatial-reasoning', exist_ok=True)
input_dir = "/cpfs/user/chendelong/instruction_tuning_dataset/visual-spatial-reasoning/data/splits/random"
output_file = "converted_datasets/visual-spatial-reasoning/visual_spatial_reasoning_instruction_"
visual_spatial_reasoning_to_instruction(input_dir, output_file)
