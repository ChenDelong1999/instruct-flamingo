from fastapi import FastAPI
from pydantic import BaseModel
import json
import time
import pprint
from open_flamingo.instruction_tuning.inferencer import Inferencer

app = FastAPI()

class Input(BaseModel):
    content_lst: dict
    typ: str

class Response(BaseModel):
    result: dict

# # Clever Flamingo V1
# inferencer = Inferencer(
#     lm_path="decapoda-research/llama-7b-hf",
#     checkpoint_paths='/cpfs/user/chendelong/.cache/openflamingo/checkpoint.pt,/cpfs/user/chendelong/open_flamingo/runs/0609-clever_flamingo_s1-[lora]-[1ep]-lr1e-4_b256-context1024-336/final_weights.pt,/cpfs/user/chendelong/open_flamingo/runs/0610-clever_flamingo_s2_0608-[perciver+xattn]-[3ep]-lr1e-4_b1024-context196/final_weights.pt,/cpfs/user/chendelong/open_flamingo/runs/0613-clever_flamingo_s3-[lora]-[1ep]-lr1e-5_b256-context1024-336/final_weights.pt',
#     clip_vision_encoder_path="ViT-L-14-336",
#     clip_vision_encoder_pretrained='openai',
#     tuning_config='/cpfs/user/chendelong/.cache/guanaco-7b',
#     v1=True
#     )
# log_file = '/cpfs/user/chendelong/open_flamingo_v2/serving/api_log.json'

# Clever Flamingo V2 (MPT-7B)
inferencer = Inferencer(
    lm_path="/cpfs/user/chendelong/cache/mpt-7b",
    checkpoint_paths='runs/clever_flamingo_v2/0813_9b-2k_context-80G-lora[lm]+xattn+perciver-countinue2/clever_flamingo_v2_9b_0818.pt',
    clip_vision_encoder_path="ViT-L-14-336",
    clip_vision_encoder_pretrained='openai',
    tuning_config='open_flamingo/instruction_tuning/tuning_config/clever_flamingo_v2.json',
    cross_attn_every_n_layers=4,
    v1=False
    )
log_file = '/cpfs/user/chendelong/open_flamingo_v2/serving/api_log_v2.json'

@app.post("/clever_flamingo",response_model=Response)        
async def clever_flamingo(request: Input):
    time_start = time.time()
    response, full_text = inferencer(
        prompt=                 request.content_lst['prompt'],
        images=                 request.content_lst['imgpaths'],
        max_new_token=          request.content_lst['args']['max_new_token'],
        num_beams=              request.content_lst['args']['num_beams'],
        temperature=            request.content_lst['args']['temperature'],
        top_k=                  request.content_lst['args']['top_k'],
        top_p=                  request.content_lst['args']['top_p'],
        do_sample=              request.content_lst['args']['do_sample'],
        length_penalty=         request.content_lst['args']['length_penalty'],
        no_repeat_ngram_size=   request.content_lst['args']['no_repeat_ngram_size'],
        response_split="### Assistant:"
    )
    time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_elapsed = time.time() - time_start
    sample_log = {
        'time': time_stamp,
        'time_elapsed': time_elapsed,
        'request': str(request),
        'response': str(response),
    }
    
    print(f'Time: {time_stamp} (time_elapsed: {time_elapsed:.2f}s)\n{request.content_lst["imgpaths"]}\n{request.content_lst["prompt"]}\n{response}\n')
    with open(log_file, 'a') as f:
        f.write(json.dumps(sample_log, indent=4) + '\n')

    return Response(result={"response": response})

# Usage:
# CUDA_VISIBLE_DEVICES=7 uvicorn api:app --host=0.0.0.0 --port=44400 --log-level=info
