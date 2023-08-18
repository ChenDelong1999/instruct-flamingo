"""Chain for question-answering with self-verification."""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from pydantic import Extra, root_validator

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from chains.caption_verification.prompt import (
    caption_decomposition_prompt,
    caption_completion_prompt,
    caption_verification_prompt,
    yes_no_prompt,
    rationale_explaination_prompt,
    caption_revision_prompt
)
from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel


def _load_question_to_checked_assertions_chain(
    llm: BaseLanguageModel,
    caption_decomposition_prompt: PromptTemplate,
    caption_completion_prompt: PromptTemplate,
    caption_verification_prompt: PromptTemplate,
    yes_no_prompt: PromptTemplate,
    rationale_explaination_prompt: PromptTemplate,
    caption_revision_prompt: PromptTemplate,
    verbose: bool = False,
):
    
    caption_decomposition_chain = LLMChain(llm=llm, prompt=caption_decomposition_prompt, verbose=verbose)
    caption_completion_chain = LLMChain(llm=llm, prompt=caption_completion_prompt, verbose=verbose)
    caption_verification_chain = LLMChain(llm=llm, prompt=caption_verification_prompt, verbose=verbose)
    yes_no_chain = LLMChain(llm=llm, prompt=yes_no_prompt, verbose=verbose)
    rationale_explaination_chain = LLMChain(llm=llm, prompt=rationale_explaination_prompt, verbose=verbose)
    caption_revision_chain = LLMChain(llm=llm, prompt=caption_revision_prompt, verbose=verbose)

    chains = [
        caption_decomposition_chain, 
        caption_completion_chain, 
        caption_verification_chain, 
        yes_no_chain, 
        rationale_explaination_chain, 
        caption_revision_chain
        ]
    
    return chains


class CaptionVerificationChain(Chain):
    """Chain for question-answering with self-verification.

    Example:
        .. code-block:: python

            from langchain import OpenAI, LLMCheckerChain
            llm = OpenAI(temperature=0.7)
            checker_chain = LLMCheckerChain.from_llm(llm)
    """

    chains: List[Chain]

    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    caption_decomposition_prompt: PromptTemplate = caption_decomposition_prompt
    caption_completion_prompt: PromptTemplate = caption_completion_prompt
    caption_verification_prompt: PromptTemplate = caption_verification_prompt
    yes_no_prompt: PromptTemplate = yes_no_prompt
    rationale_explaination_prompt: PromptTemplate = rationale_explaination_prompt
    caption_revision_prompt: PromptTemplate = caption_revision_prompt
    
    input_key: List = ["caption", "img_path"]  #: :meta private:
    output_key: List = ["revised_caption", "verification_results"]  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return self.input_key

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return self.output_key

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        caption, img_path = inputs['caption'], inputs['img_path']

        caption_decomposition_chain, caption_completion_chain, caption_verification_chain, yes_no_chain, rationale_explaination_chain, caption_revision_chain = self.chains

        
        output_formating_prompt = 'The followings are some image captions decomposed from the original one:\n'
        decomposed_captions = caption_decomposition_chain(inputs={"caption": caption, 'output_formating_prompt':f'\n### Assistant: {output_formating_prompt}1.'})
        decomposed_captions = decomposed_captions['text'].replace(output_formating_prompt, '').split('\n')
        decomposed_captions = [sentence[3:] for sentence in decomposed_captions if sentence != '']
        # print(decomposed_captions)

        input_list = [{"caption": decomposed_caption, "img_path": img_path} for decomposed_caption in decomposed_captions]
        decomposed_captions = caption_completion_chain.apply(input_list)
        # print(decomposed_captions)

        input_list = [{"caption": decomposed_caption['text'], "img_path": img_path} for decomposed_caption in decomposed_captions]
        verification_results = caption_verification_chain.apply(input_list)
        # print(verification_results)

        verification_results = yes_no_chain.apply(verification_results)
        verification_results = [verification_result['text'].replace('.', '').lower()=='yes' for verification_result in verification_results]
        # print(verification_results)

        if False in verification_results:
            
            input_list = [{"caption": decomposed_caption['text'], "img_path": img_path} for decomposed_caption, verification_result in zip(decomposed_captions, verification_results) if verification_result is not True]
            rationale_explaination_results = rationale_explaination_chain.apply(input_list)
            # print(rationale_explaination_results)

            correct_captions = '- ' + '\n- '.join([f"{caption['text']}" for caption, verification_result in zip(decomposed_captions, verification_results) if verification_result])
            rationale_explainatios = '- ' + '\n- '.join([f"\"{inputs['caption'].lower().replace('.', '')}\": {rationale['text']}" for rationale, inputs in zip(rationale_explaination_results, input_list)])

            output_formating_prompt = 'The revised caption would be:\n'
            revised_caption = caption_revision_chain(inputs={
                "caption": caption, 
                'img_path': img_path,
                "correct_captions": correct_captions, 
                "incorrect_captions": rationale_explainatios,
                "output_formating_prompt": f'\n### Assistant: {output_formating_prompt}"'
                })
            revised_caption = revised_caption['text'].replace(output_formating_prompt, '')

            if revised_caption.endswith('"'):
                revised_caption = revised_caption[:-1]
            if revised_caption.endswith('".'):
                revised_caption = revised_caption[:-2]

        else:
            revised_caption = caption

        verification_results = [{'decomposed_caption': decomposed_caption['text'], 'verification_result': verification_result} for decomposed_caption, verification_result in zip(decomposed_captions, verification_results)]            
        
        return {"revised_caption": revised_caption, "verification_results": verification_results}

    @property
    def _chain_type(self) -> str:
        return "llm_checker_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        caption_decomposition_prompt: PromptTemplate = caption_decomposition_prompt,
        caption_completion_prompt: PromptTemplate = caption_completion_prompt,
        caption_verification_prompt: PromptTemplate = caption_verification_prompt,
        yes_no_prompt: PromptTemplate = yes_no_prompt,
        rationale_explaination_prompt: PromptTemplate = rationale_explaination_prompt,
        caption_revision_prompt: PromptTemplate = caption_revision_prompt,
        verbose: bool = False,
        **kwargs: Any,
    ) -> CaptionVerificationChain:
        chains = (
            _load_question_to_checked_assertions_chain(
                llm,
                caption_decomposition_prompt,
                caption_completion_prompt,
                caption_verification_prompt,
                yes_no_prompt,
                rationale_explaination_prompt,
                caption_revision_prompt,
                verbose
            )
        )
        return cls(
            chains=chains,
            **kwargs,
        )