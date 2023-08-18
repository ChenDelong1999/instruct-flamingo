from langchain.prompts.prompt import PromptTemplate


caption_decomposition_prompt=PromptTemplate(
    input_variables=["caption", "output_formating_prompt"],
    template="""\
Given a caption related to an image, decompose it into several independent, atomized, complete, yet succinct sentences of image captions. \
Each captions should be as simple as possible, and cannot be divided further. \
They corresponds to different aspects of the original sentence. \
\nPlease make sure that all of the key information is comprehensively retained. \
However, do not add any additional information that is not present in the original caption. \
\nThe caption to process is: "{caption}"{output_formating_prompt}""")


caption_completion_prompt=PromptTemplate(
    input_variables=["caption", "img_path"],
    template="""<img_path>{img_path}<img_path>Convert this sentence about the image '{caption}' into a complete caption, but keep it as simple as possible.""")


caption_verification_prompt=PromptTemplate(
    input_variables=["caption", "img_path"],
    template="""<img_path>{img_path}<img_path>Given this image, Is the statement '{caption}' correct?""")


yes_no_prompt=PromptTemplate(
    input_variables=["text"],
    template="""Classify the statement '{text}' into one of the following categories: 'yes', 'no'.  Short answer:""")


rationale_explaination_prompt=PromptTemplate(
    input_variables=["caption", "img_path"],
    template="""<img_path>{img_path}<img_path>The following caption of this image is incorrect:\n'{caption}'\nPlease explain the reason in detail according to your observation.""")


caption_revision_prompt=PromptTemplate(
    input_variables=["caption", "img_path", "correct_captions", "incorrect_captions", "output_formating_prompt"],
    template="""\
Your task is to revise a partially correct caption of this image<img_path>{img_path}<img_path>.\n\nThe original caption is: '{caption}'\n\n\
The following aspects of this caption are correct and the information should be retained:\n{correct_captions}\n\n\
However, some of them are incorrect and should be revised:\n{incorrect_captions}\n\n\
Please revise the caption to make it reliable according to your observation and the above analysis.{output_formating_prompt}""")
