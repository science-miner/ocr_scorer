from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
import time 
import ocr_scorer
from enum import Enum

router = APIRouter()

scorer = None

def set_scorer(global_scorer):
    global scorer
    scorer = global_scorer

@router.get("/alive", response_class=PlainTextResponse, tags=["generic"], 
    description="Return true if service is up and running.")
def is_alive_status():
    return "true"

@router.get("/version", response_class=PlainTextResponse, tags=["generic"], 
    description="Return the version tag of the service.")
def get_version():
    api_settings = scorer.config['api']
    return api_settings['version']

'''
Estimate the OCR quality of a text segment
'''
@router.post("/score/text", tags=["score"], 
    description="Estimate the OCR quality of a text segment. Return a quality score in [0:1].")
async def post_score_text(text: str, lang: str = 'en'):
    start_time = time.time()

    print(lang)

    result = {}
    result['score'] = scorer.score_text(text, lang)
    result['runtime'] = round(time.time() - start_time, 3)
    
    return result

