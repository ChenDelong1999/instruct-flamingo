import random


short_caption_prompts = [
    "A short image caption:",
    "A short image description:",
    "A photo of",
    "An image that shows",
    "Write a short description for the image.",
    "Write a description for the photo.",
    "Provide a description of what is presented in the photo.",
    "Briefly describe the content of the image.",
    "Can you briefly explain what you see in the image?",
    "Could you use a few words to describe what you perceive in the photo?",
    "Please provide a short depiction of the picture.",
    "Using language, provide a short account of the image.",
    "Use a few words to illustrate what is happening in the picture.",
]

long_caption_prompts = [
    "Describe the following image in detail",
    "Provide a detailed description of the given image",
    "Give an elaborate explanation of the image you see",
    "Share a comprehensive rundown of the presented image",
    "Offer a thorough analysis of the image",
    "Explain the various aspects of the image before you",
    "Clarify the contents of the displayed image with great detail",
    "Characterize the image using a well-detailed description",
    "Break down the elements of the image in a detailed manner",
    "Walk through the important details of the image",
    "Portray the image with a rich, descriptive narrative",
    "Narrate the contents of the image with precision",
    "Analyze the image in a comprehensive and detailed manner",
    "Illustrate the image through a descriptive explanation",
    "Examine the image closely and share its details",
    "Write an exhaustive depiction of the given image"
]


direct_answer_prompts = [
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


equation_ocr_prompt = [
    'Please write out the expression of the formula in the image using LaTeX format.',
    'Now perform OCR',
    'recognize the characters in the image',
    'write out this expression in markdown',
    'this is a equation of:',
    'the textual content is:',
    'what does it says?',
    'write out this',
    'this is:',
    'text recognition result is:',
    'a mathmatical expression of'
]


'''
# earlier_image_identifier = 'Earlier image'
# later_image_identifier = 'Later image'
# spliter = ', '
# input_text_segments = [
#     f"{earlier_image_identifier}: <img_path>{left_image_path}<img_path>",
#     f"{later_image_identifier}: <img_path>{right_image_path}<img_path>",
# ]
# input_text = f'{spliter}'.join(input_text_segments)
'''
remote_change_caption_templates = [
    "What are the differences between this pair of remote sensing images? Note that these two images could possibly be the same.",
    "Can you spot the potential differences between these two remote sensing images?",
    "Identify the (possible) differences between the two remote sensing images.",
    "List the differences between these two satellite images. It is also possible that these two images are the same.",
    "Describe the differences between the satellite images if these images are not same.",
    "What changes can you observe between the two satellite pictures? Please note that these two pictures could be the same.",
    "Please point out all the (potential) differences between the two satellite photos.",
    "Find the differences between the given satellite images there are some changes.",
    "Are there some changes? Tell me the differences between these two image of earth observations.",
    "Explain the variations between the two remote sensing images if changes appears.",
    "Detect the possible differences between these two images captured by satellite.",
    "If these two images are same? What are the dissimilarities between the pictures from satellite?",
    "Are there something changed? Note the differences in the given satellite images.",
    "Please enumerate the differences between the remote sensing pictures, if there are some changes.",
    "Whether it is changed? State the differences between the provided remote sensing images.",
]



nlvr2_question_answer_templates = [
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

refcoco_instructions = [
    "Given this image, generate the referring expression for the object inside this <REGION>.",
    "Describe the object located within this area: <REGION>.",
    "What content is contained within this region <REGION>?",
    "Can you provide a referring expression for the object here <REGION>?",
    "What is the object found inside <REGION>?",
    "Can you describe this object <REGION>?",
    "In the image, what is the content within this patch <REGION>?",
    "Describe the object within: <REGION>.",
    "What is inside the <REGION> in the image?",
    "Please generate a description for this object <REGION>.",
    "What does the <REGION> in the image correspond to?",
    "Can you describe the content within the area of <REGION>?",
    "Please provide a referring expression for this object <REGION>.",
    "What is the content inside <REGION>?",
    "Describe the object inside <REGION>.",
    "What is the object inside <REGION>?",
    "Please provide a description of the object inside <REGION>.",
    "What is the referring expression for the object <REGION>?",
]

def get_chain_of_thought_instruction(instruction):
    # Alpaca-COT 
    chain_of_thought_prompts = [
        # AQUA (2690)
        '<INSTRUCTION> Some random reasoning:',
        '<INSTRUCTION> Hmmm, my stream of consciousness:',
        '<INSTRUCTION> Stream of consciousness:',
        'Give stream of consciousness and then the final answer. <INSTRUCTION> ',
        'Student: <INSTRUCTION> \nAnother student: Let\'s say, hmmm...\n',
        'Give a quick stream of consciousness before answering the following question. <INSTRUCTION> ',   
        # CREAK (6839)
        '<INSTRUCTION> Step-by-step reasoning process:',
        '<INSTRUCTION> Your chain-of-thought:',
        'Given the following question, let\'s solve step-by-step. <INSTRUCTION> ',
        '<INSTRUCTION> Chain of thought and solution for this question is:',
        '<INSTRUCTION> The thought process:',
        # ECQA (7027)
        '<INSTRUCTION> Let\'s give stream of consciousness first:',
        '<INSTRUCTION> Please answer and provide answer explanation.',
        '<INSTRUCTION> \nreasoning:',
        'I\'ll give you a question, please answer with step-by-step reasoning process. <INSTRUCTION>',
        # GSM8k (7267)
        'Question: <INSTRUCTION> Think carefully first, then make a decision:',
        '<INSTRUCTION> Let\'s be accurate as possible.',
        # QASC (1038)
        '<INSTRUCTION> Let\'s solve this problem gradually.\n',
        '<INSTRUCTION> Hmmm, let me think. I don\'t want to be wrong, so I got to be careful.',
        '<INSTRUCTION> \nLet\'s reason step-by-step:',
        '<INSTRUCTION> OK. Let\'s think hard:',
        '<INSTRUCTION> \nThe thinking starts now:',
        # QED (1809)
        'Answer the following Q with stream of consciousness. <INSTRUCTION> ',
        'Give a stream of consciousness and then the final answer. <INSTRUCTION> ',
        '<INSTRUCTION> Let\'s have some stream of consciousness first.',
        '<INSTRUCTION> Okie... think carefully first, then make a decision: ',
        '<INSTRUCTION> Let\'s give stream of consciousness first:',
        '<INSTRUCTION> Steam of consciousness below: ',
        '<INSTRUCTION> OK. Let\'s think. My stream of consciousness: ',
    ]

    return random.choice(chain_of_thought_prompts).replace('<INSTRUCTION>', instruction)


def get_nli_cot_instruction(premise, hypothesis):

    # Alpaca-CoT-ESNLI (34932)
    chain_of_thought_prompts = [
        'Please answer the following question by reasoning step-by-step. Premise: \"<PREMISE>\"\\nHypothesis: \"<HYPOTHESIS>\"\\nDo we know that the hypothesis entailed by the premise?. Step-by-step reasoning:',
        'Can we conclude from \"<HYPOTHESIS>\" that \"<PREMISE>\"?\\nOptions:\\n- yes\\n- no\\n- it is not possible to tell A step-by-step solution is:',
        'If \"<PREMISE>\" does that mean that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- it is not possible to tell\\n- no Let\'s be accurate as possible and think first.',
        'Given the sentence \"<PREMISE>\" is it true that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- it is not possible to tell\\n- no Stream of thoughts:',
        'Premise: \"<PREMISE>\"\\nBased on this premise, can we conclude that the hypothesis \"<HYPOTHESIS>\" is true?\\nOptions:\\n- yes\\n- it is not possible to tell\\n- no A step-by-step solution is:',
        'Can we conclude from \"<PREMISE>\" that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- no\\n- it is not possible to tell Stream of thoughts:',
        'Given the sentence \"<PREMISE>\" can we conclude that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- no\\n- it is not possible to tell Now, let\'s be accurate as possible. Some thinking first:',
        'Premise: \"<PREMISE>\"\\nHypothesis: \"<HYPOTHESIS>\"\\nIs the hypothesis entailed by the premise?\\nOptions:\\n- yes\\n- it is not possible to tell\\n- no A step-by-step solution is:',
        'Student: Premise: \"<PREMISE>\"\\nHypothesis: \"<HYPOTHESIS>\"\\nIs the hypothesis entailed by the premise?\\nOptions:\\n- yes\\n- it is not possible to tell\\n- no.\nTeacher: Let\'s think:',
        'If \"<PREMISE>\" does that mean that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- it is not possible to tell\\n- no Let\'s solve step-by-step:',
        'Student: Can we conclude from \"<PREMISE>\" that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- no\\n- it is not possible to tell.\nTeacher: Let\'s think:',
        'Given the sentence \"<PREMISE>\" can we conclude that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- it is not possible to tell\\n- no A step-by-step solution is:',
        'Test for natural language inference.\\nPremise: \"<PREMISE>\"\\nHypothesis: \"<HYPOTHESIS>\"\\nIs the hypothesis entailed by the premise?\\nOptions:\\n- yes\\n- no\\n- it is not possible to tell Stream of thoughts:',
        'Given the sentence \"<PREMISE>\" is it true that \"<HYPOTHESIS>\"?\\nOptions:\\n- yes\\n- no\\n- it is not possible to tell A step-by-step solution is:',
    ]

    return random.choice(chain_of_thought_prompts).replace('<PREMISE>', premise).replace('<HYPOTHESIS>', hypothesis)

def get_options_prompt(instruction, options):
    # Alpaca-COT QASC
    prompt = f'{instruction} Options:'
    for i, option in enumerate(options, start=1):
        prompt += f'\n- ({chr(65 + i)}) {option}'
    
    return prompt


def get_toolformer_prompt(instruction, tools):
    """
    learned_tools = {
        'wikipedia': 'A wrapper around Wikipedia. Useful for when you need to display general information about people, places, companies, historical events, or other subjects found in an encyclopedia, displays a snippet summary of the topic. Input should be a search query.\nwikipedia(query)',
        'weather': 'Useful for when you want to get weather information from the OpenMeteo API. The input should be a question in natural language that this API can answer.\nweather(querywithlocation)',
        'wolfram': 'A Wolfram Alpha search engine. Useful for when you need to answer questions about Math, Science, Technology, Culture, Society, and Everyday Life. Input should be a search query.\nwolfram(query)',
        'news': 'Use this when you want to get information about the top headlines of current news stories. The input should be a question in natural language that this API can answer.\nnews(query)',
        'python': 'A Python shell. Use this to execute Python commands. Input should be a valid Python command or script. If you expect output it should be printed out. Useful for all code, as well as math calculations.\npython(codetoexecute)',
        'request': 'A portal to the internet. Use this when you need to get specific content from a site. Input should be a specific URL, and the output will be all the text on that page.\nrequest(url)',
        'search': 'A search engine. Useful for when you need to answer questions about current events. Input should be a search query.\nsearch(query)',
        'shell': 'Executes commands in a terminal. Input should be valid commands, and the output will be any output from running that command.\nshell(shellcommand)',
    }
    """

    tool_names = ', '.join(list(tools.keys()))
    tool_documentations = '\n'.join(list(tools.values()))
    prompt = f'toolformer: enabled\ntoolformer access: {tool_names}\n{tool_documentations}\n{instruction}'

    return prompt

# def get_toolbench_prompt(instruction, tools):

#     "Toolbench System Message -- You have access of the following tools:\n1.memeados: Generate custom image, gif and video memes.\n\nSpecifically, you have access to the following APIs: [{'name': 'drakelikehate_for_memeados', 'description': 'This is the subfunction for tool \"memeados\", you can use this tool.The description of this function is: \"Generate Drake Likes and Hates meme\"', 'parameters': {'type': 'object', 'properties': {'text2': {'type': 'string', 'description': '', 'example_value': 'This text is liked.'}, 'text1': {'type': 'string', 'description': '', 'example_value': 'This text is hated'}}, 'required': ['text2', 'text1'], 'optional': []}}, {'name': 'pet_pet_for_memeados', 'description': 'This is the subfunction for tool \"memeados\", you can use this tool.The description of this function is: \"Generate My pet_pet_for_memeados meme GIF\"', 'parameters': {'type': 'object', 'properties': {'image': {'type': 'string', 'description': '', 'example_value': 'https://i.pravatar.cc/300'}}, 'required': ['image'], 'optional': []}}, {'name': 'sponge_bob_for_memeados', 'description': 'This is the subfunction for tool \"memeados\", you can use this tool.The description of this function is: \"Generate Sponge Bob meme\"', 'parameters': {'type': 'object', 'properties': {'text': {'type': 'string', 'description': '', 'example_value': 'Example test'}}, 'required': ['text'], 'optional': []}}, {'name': 'google_fake_autocomplete_for_memeados', 'description': 'This is the subfunction for tool \"memeados\", you can use this tool.The description of this function is: \"Generate Fake google autocomplete\"', 'parameters': {'type': 'object', 'properties': {'text1': {'type': 'string', 'description': '', 'example_value': 'Search'}, 'text3': {'type': 'string', 'description': '', 'example_value': 'search autocomplete 2'}, 'text2': {'type': 'string', 'description': '', 'example_value': 'search autocomplete 1'}}, 'required': ['text1', 'text3', 'text2'], 'optional': []}}, {'name': 'Finish', 'description': \"If you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information.\", 'parameters': {'type': 'object', 'properties': {'return_type': {'type': 'string', 'enum': ['give_answer', 'give_up_and_restart']}, 'final_answer': {'type': 'string', 'description': 'The final answer you want to give the user. You should have this field if \"return_type\"==\"give_answer\"'}}, 'required': ['return_type']}}]",
        