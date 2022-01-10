import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import random
import argparse
import time

from model_scorer import ModelScorer
from utils import _load_config
from unicode_utils import normalise_text

import logging
import logging.handlers
# default logging settings, will be override by config file
logging.basicConfig(filename='client.log', filemode='w', level=logging.DEBUG)


class OCRScorer(object):

    # models is a map of models with language (two letters ISO 639-1) as key
    models = {}
    config = None

    # to do: make this list dynamic by exploring the data/models repository
    supported_languages = ['en', 'de', 'fr']

    def __init__(self, config_path="./config.yml"):
        self.config = _load_config(config_path)

        logs_filename = "client.log"
        if "log_file" in self.config: 
            logs_filename = self.config['log_file']

        logs_level = logging.DEBUG
        if "log_level" in self.config:
            if self.config["log_level"] == 'INFO':
                logs_level = logging.INFO
            elif self.config["log_level"] == 'ERROR':
                logs_level = logging.ERROR
            elif self.config["log_level"] == 'WARNING':
                logs_level = logging.WARNING
            elif self.config["log_level"] == 'CRITICAL':
                logs_level = logging.CRITICAL
            else:
                logs_level = logging.NOTSET

        logging.basicConfig(filename=logs_filename, filemode='w', level=logs_level)
        print("logs are written in " + logs_filename)

    def load_model(self, lang):
        print("load_model for ", lang)
        local_model = ModelScorer(lang, self.config)
        self.models[lang] = local_model

    def get_model(self, lang):
        print("get_model for ", lang)

        local_model = None
        
        if not lang in self.models:
            self.load_model(lang)
        if not lang in self.models:
            raise Exception("No model available for the language " + lang)
        local_model = self.models[lang]    

        if local_model == None:
            raise Exception("Failed to identify the language")

        return local_model

    def score_text(self, text, lang="en"):
        '''
        If no language is provided, use a language detector
        '''
        local_model = None
        try:
            local_model = self.get_model(lang)
        except:
            logging.error("Fail to load the model for language " + lang)
        
        if local_model is None:
            raise Exception("Failed to process language " + lang)
        
        text_scores = []
        for text in local_model.read_text_sequence(text, max_length=500):
            text_scores.append(local_model.score_text(text))
        local_file_score = np.mean(text_scores)

        return local_file_score


    def score_patent_xml(self, xml_file, lang=None):
        '''
        Expected XML format for patent is ST.36

        If no language is provided as parameter (override all XML tags), use the @lang attribute in the XML, 
        or a language detector if no @lang attributes is found in the XML
        '''

        return 1.0


    def score_pdf(self, pdf_file, lang=None):
        '''
        PDF file is parsed by external pdfalto tool. Spatial information can be used. 
        If no language is provided, use a language detector
        '''

        return 1.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simple command line OCR scorer. Use the service for more intensive/pipeline tasks.")
    parser.add_argument("--config-file", type=str, required=False, help="configuration file to be used", default='./config.yml')
    parser.add_argument("--debug", action="store_true", required=False, default=False,
                        help="activate the debug mode (override the config file logging parameter)")
    parser.add_argument("--text-file", type=str, required=False, help="text file to be evaluated, expected encoding is UTF-8")
    parser.add_argument("--pdf-file", type=str, required=False, help="PDF file to be evaluated")
    parser.add_argument("--patent-xml-file", type=str, required=False, help="Patent file in XML ST-36 format to be evaluated")

    args = parser.parse_args()

    debug = args.debug
    text_file = args.text_file
    pdf_file = args.pdf_file
    patent_xml_file = args.patent_xml_file

    config_file = args.config_file
    text_file = args.text_file
    pdf_file = args.pdf_file
    patent_xml_file = args.xml_file

    try:
        scorer = OCRScorer(config_file)
        if pdf_file != None:
            scorer.score_pdf(pdf_file)
        elif text_file != None:
            scorer.score_pdf(text_file)
        elif patent_xml_file != None:
            scorer.score_patent_xml(patent_xml_file)
        else:
            print("At least one file to be evaluated must be provided\n")
            parser.print_help()
            exit(1)
    except Exception as e:
        print("Scorer failed: ", str(e))
        exit(1)
