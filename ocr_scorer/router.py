from fastapi import APIRouter, HTTPException, Request, Response
from fastapi import File, Form, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
import time 
import ocr_scorer
from enum import Enum
from lxml import etree

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
    description="Estimate the OCR quality of a text segment. Return a quality score in [0,1].")
async def post_score_text(text: str, lang: str = 'en'):
    start_time = time.time()

    print(lang)

    result = {}
    result['score'] = scorer.score_text(text, lang)
    result['runtime'] = round(time.time() - start_time, 3)
    
    return result


'''
Estimate the OCR quality of a text file
'''
@router.post("/score/file/text", tags=["score"], 
    description="Estimate the OCR quality of a text file. Return a quality score in [0,1].")
async def post_score_file_text(file: UploadFile = File(...), lang: str = Form(...)):
    start_time = time.time()

    if file is None:
        raise HTTPException(status_code=404, detail="Invalid empty file")

    if file.content_type != 'text/plain':
        raise HTTPException(status_code=404, detail="Invalid content type file, must be text: " + file.content_type)

    text = await file.read()
    text = text.decode()

    result = {}
    result['score'] = scorer.score_text(text, lang)
    result['runtime'] = round(time.time() - start_time, 3)
    
    return result


'''
Estimate the OCR quality of an XML file
'''
@router.post("/score/file/xml", tags=["score"], 
    description="Estimate the OCR quality of an XML file. Return a quality score in [0,1].")
async def post_score_file_xml(file: UploadFile = File(...), lang: str = Form(...)):
    start_time = time.time()

    if file is None:
        raise HTTPException(status_code=404, detail="Invalid empty file")

    if file.content_type not in [ 'text/xml', 'application/xml', 'application/vnd.pdm.v3+xml' ]:
        raise HTTPException(status_code=404, detail="Invalid content type file, must be XML: " + file.content_type)

    xml_string = await file.read()
    #xml_string = xml_string.decode()
    root = etree.fromstring(xml_string)
    text = etree.tostring(root, encoding='utf-8', method='text')
    text = text.decode()

    result = {}
    result['score'] = scorer.score_text(text, lang)
    result['runtime'] = round(time.time() - start_time, 3)
    
    return result




