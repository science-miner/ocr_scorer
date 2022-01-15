from fastapi import APIRouter, HTTPException, Request, Response
from fastapi import File, Form, UploadFile
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
import time 
import ocr_scorer
from enum import Enum
from lxml import etree
from pdfalto.alto_parser import filter_text
from pdfalto.wrapper import PdfAltoWrapper

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
    the_score = scorer.score_text(text, lang)
    print(the_score)
    result['score'] = the_score
    result['runtime'] = round(time.time() - start_time, 3)

    print(result)
    
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
    root = etree.fromstring(xml_string)
    text = etree.tostring(root, encoding='utf-8', method='text')
    text = text.decode()

    result = {}
    result['score'] = scorer.score_text(text, lang)
    result['runtime'] = round(time.time() - start_time, 3)
    
    return result


'''
Estimate the OCR quality of a PDF file
'''
@router.post("/score/file/pdf", tags=["score"], 
    description="Estimate the OCR quality of an XML file. Return a quality score in [0,1].")
async def post_score_file_pdf(file: UploadFile = File(...), lang: str = Form(...)):
    start_time = time.time()

    if file is None:
        raise HTTPException(status_code=404, detail="Invalid empty file")

    if file.content_type not in [ 'application/pdf' ]:
        raise HTTPException(status_code=404, detail="Invalid content type file, must be PDF: " + file.content_type)

    pdf_content = await file.read()

    # write tmp file on disk
    input_file = os.path.join('./data/pdfalto/tmp/', binascii.b2a_hex(os.urandom(7)).decode() + ".pdf")

    pdfalto = PdfAltoWrapper('./data/pdfalto/lin64/pdfalto')
    output_path = input_file.replace(".pdf", ".xml")
    pdfalto.convert(input_file, output_path)
    logging.info("pdfalto conversion: " + output_path)

    local_text = filter_text(output_path)

    #print(local_text)
    
    result = {}
    result['score'] = scorer.score_text(text, lang)
    result['runtime'] = round(time.time() - start_time, 3)

    # cleaning tmp PDF and ALTO file(s)
    if os.path.isfile(input_path):
        os.remove(input_path)
    if os.path.isfile(output_path):
        os.remove(output_path)
    output_path = output_path.replace(".xml", "_metadata.xml")
    if os.path.isfile(output_path):
        os.remove(output_path)
    
    return result

