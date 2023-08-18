import random
import csv
import json
import argparse
import os.path as osp
import os
from operator import itemgetter

convert_html_instructions = [
    "Convert the image to an HTML representation.",
    "Create an HTML version of the image.",
    "Please provide the HTML code for the image.",
    "Generate HTML code that represents the image.",
    "Produce the HTML equivalent of the image.",
    "Transform the image into an HTML format.",
    "Recreate the image as an HTML file.",
    "Convert the given image to HTML format.",
    "Please generate an HTML version of the image.",
    "Create an HTML representation of the image.",
    "I want to create an HTML page that looks like this draft, can you help me?",
    "I hope to generate a website that resembles this image, how can I do that?",
    "Please help me turn this image into an HTML page.",
    "Can you assist me in creating a website based on this picture?",
    "I'd like to convert this picture into a website, what's the best way to do it?",
    "Could you guide me on how to transform this image into a web page?",
    "I'm trying to build a website that matches this design, any tips?",
    "How can I use this picture as inspiration for my website's layout?",
    "I want my website to look like this sketch, where should I start?"

]



direct_answer_instructions = [
    "Provide a concise answer to this question without any explanation or analysis.",
    "Give a brief answer without discussing your thought process.",
    "Offer a short and direct response without elaboration.",
    "Respond with a simple answer, skipping any reasoning.",
    "Share a straightforward answer, no explanation needed.",
    "Submit a quick answer without detailing your thoughts.",
    "Present a clear and concise answer without any background information.",
    "Furnish a short response, leaving out any analysis.",
    "State a brief answer, bypassing any detailed explanation.",
    "Reveal a succinct answer, forgoing any thought process.",
    "Deliver a short and simple answer without further elaboration.",
    "Produce a terse response, eliminating any reasoning.",
    "Give an unadorned answer, without discussing the thought process.",
    "Provide a to-the-point response without any detailed explanation.",
    "Offer an undecorated answer, skipping the reasoning part.",
    "(Return a minimal answer, without any discussion or analysis).",
    "(Submit a plain response, leaving out any thought process).",
    "(Communicate a brief and direct answer, avoiding any explanation).",
    "(Share a neat answer, without delving into your thoughts).",
    "(State a clear-cut answer, excluding any additional information).",
]

question_prompts = [
    "Based on the provided image, answer the following question. ",
    "Refer to the given image and respond to the question below: ",
    "Using the image as a reference, please address the question: ",
    "Considering the image, please provide an answer to the question. ",
    "With the image in mind, reply to the subsequent question: ",
    "Taking the supplied image into account, answer the question: ",
    "Examine the image and then respond to the inquiry. ",
    "After observing the picture, please address the question: ",
    "In light of the image, answer the question that follows: ",
    "Keeping the image in consideration, reply to the question: ",
    "View the image and proceed to answer the question: ",
    "Study the given picture and respond to the question. ",
    "Upon analyzing the image, provide an answer to the question: ",
    "In reference to the image, address the question below: ",
    "While looking at the provided image, answer the following: ",
    "Regarding the presented image, please answer the question: ",
    "Assess the image and proceed to respond to the question. ",
    "Using the image for context, answer the following question: ",
    "Reflect on the image and then reply to the question. ",
    "After inspecting the image, provide a response to the question: ",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=None, type=str, required=True,
                        help="The root directory of the raw WebSRC dataset; The output SQuAD-style json file will also"
                             "be placed here.")
    parser.add_argument("--version", default=None, type=str, required=True,
                        help="The version of the generating dataset, which will also be the name of the json file.")
    parser.add_argument("--suffix", default="", type=str,
                        help="Other suffix to distinguish different dataset.")
    return parser.parse_args()

def convert_csv_to_dict(args, mode):
    dir_list = os.walk(args.root_dir)
    print('Start Converting')

    data = []

    for d, _, fs in dir_list:
        for f in fs:
            if f != 'dataset.csv':
                continue
            print('Now converting', d + '/' + f)
            raw_data = list(csv.DictReader(open(osp.join(d, f))))

            raw_data.sort(key=itemgetter('id'))

            last = raw_data[0]
            for i in range(len(raw_data)):
                current = raw_data[i]
                if i != 0:
                    if mode=='qa':
                        question_prompt = random.choice(question_prompts) if random.random() < 0.5 else ''
                        # Update this part to the new format
                        if random.random() < 0.5:
                            instruction = question_prompt + last['question'] + ' ' + random.choice(direct_answer_instructions)
                        else:
                            instruction = question_prompt + random.choice(direct_answer_instructions) + ' ' + last['question']
                        data_point = {
                                    'input': f"{instruction}<img_path>/{d}/processed_data/{last['id'][2:9]}.png<img_path>".replace('/../','') ,
                                    'output': last['answer'].capitalize() + '.' if last['answer'][-1] != '.' else ''}
                        data.append(data_point)
                    elif mode=='html':
                        if random.random() < 0.1:
                            with open(f"{d}/processed_data/{last['id'][2:9]}.html", "r") as html_file:
                                html_content = html_file.read()
                            html_instruction = random.choice(convert_html_instructions)
                            html_data_point = {
                                            'input': f"{html_instruction}<img_path>{d}/processed_data/{last['id'][2:9]}.png<img_path>".replace('/../',''),
                                            'output': html_content}
                            data.append(html_data_point)

                last = current

    print('Converting Finished\n')
    return data

def dataset_split(args, data, mode):
    split = json.load(open(osp.join(args.root_dir, 'dataset_split_release.json')))
    train_data, test_data, dev_data = [], [], []
    print(split)
    for data_point in data:
        # <img_path>data/auto/08/08/0800001.png<img_path>
        # 'input': '<img_path>//cpfs/user/chendelong/instruction_tuning_dataset/WebSRC/WebSRC-Baseline/data/auto/08/processed_data/0800001.png<img_path>'
        # print(data_point['input'])
        domain_website = data_point['input'].split('/')[-4][:2] + data_point['input'].split('/')[-3]

        if domain_website in split['test']:
            test_data.append(data_point)
        elif domain_website in split['dev']:
            dev_data.append(data_point)
        elif domain_website in split['train']:
            train_data.append(data_point)
        else:
            raise ValueError(f'The domain_website {domain_website} is not in the split dict {split}. data_point: {data_point}')

    # Write the train, test, and dev datasets to JSON files
    with open(osp.join('converted_datasets/websrc', f'websrc_{mode}_train' + args.suffix + '.json'), 'w') as f:
        f.write(json.dumps(train_data))
    with open(osp.join('converted_datasets/websrc', f'websrc_{mode}_test' + args.suffix + '.json'), 'w') as f:
        f.write(json.dumps(test_data))
    with open(osp.join('converted_datasets/websrc', f'websrc_{mode}_dev' + args.suffix + '.json'), 'w') as f:
        f.write(json.dumps(dev_data))

    return
# cd /cpfs/user/chendelong/instruction_tuning_dataset/WebSRC/WebSRC-Baseline/src
# python websrc_to_instruction.py --root_dir '/cpfs/user/chendelong/instruction_tuning_dataset/WebSRC/WebSRC-Baseline/data' --version websrc1.0
if __name__ == '__main__':
    args = parse_args()
    os.makedirs('converted_datasets/websrc', exist_ok=True)
    for mode in ['qa', 'html']:
        data = convert_csv_to_dict(args, mode)
        dataset_split(args, data, mode)



