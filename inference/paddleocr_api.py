from paddleocr import PaddleOCR
from fastapi import FastAPI
from pydantic import BaseModel
import json
import pprint
app = FastAPI()

class Input(BaseModel):
    content_lst: dict
    typ: str

class Response(BaseModel):
    result: dict

ocr = PaddleOCR(use_angle_cls=True, lang="en")
ocr_cache = {}

@app.post("/paddle_ocr",response_model=Response)        
async def clever_flamingo(request: Input):
    imgpath = request.content_lst['imgpath']
    print('='*128)
    print(imgpath)
    if imgpath in ocr_cache:
        print("cache hit")
        result = ocr_cache[imgpath]
    else:
        result = ocr.ocr(imgpath, cls=True)
        ocr_cache[imgpath] = result
    pprint.pprint(result, width=128)

    return Response(result={"response": result})


'''Usage
cd /cpfs/user/chendelong/open_flamingo_v2/inference
conda activate paddle_env
CUDA_VISIBLE_DEVICES=7 uvicorn paddleocr_api:app --host=0.0.0.0 --port=44410 --log-level=info

'''