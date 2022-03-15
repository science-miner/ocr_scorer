import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import binascii
import numpy as np
import random
import argparse
import time
from lxml import etree

from lm_scorer import LMScorer
from utils import _load_config, _remove_outliner
from pdfalto.alto_parser import filter_text
from pdfalto.wrapper import PdfAltoWrapper

import logging
import logging.handlers
# default logging settings, will be override by config file
logging.basicConfig(filename='client.log', filemode='w', level=logging.DEBUG)

from sklearn.utils import shuffle
import xgboost as xgb

import cld3

SCORER_FILE = "scorer.json"

# to do: make this list dynamic by exploring the data/models repository
supported_languages = ['en', 'de', 'fr']

class OCRScorer(object):

    config = None
    config_path = None

    # models is a map of language models, language (two letters ISO 639-1) as key
    models = {}
    
    # scorers is a map of regression models, language (two letters ISO 639-1) as key
    scorers = {}

    def __init__(self, config_path="./config.yml"):
        self.config_path = config_path
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

    def load_lm_model(self, lang):
        lm_model = LMScorer(lang, config_path=self.config_path)
        lm_model.load()
        self.models[lang] = lm_model


    def get_lm_model(self, lang):
        local_model = None
        if not lang in self.models:
            self.load_lm_model(lang)
            if not lang in self.models:
                raise Exception("No model available for the language " + lang)
        
        local_model = self.models[lang]
        if local_model == None:
            raise Exception("Failed to identify the language")

        return local_model

    def get_scorer_model(self, lang):
        if lang in self.scorers:
            return self.scorers[lang]
        self.load_scorer(lang)
        if not lang in self.scorers:
            raise Exception("No model available for the language " + lang)
        return self.scorers[lang]

    def score_text(self, text, lang="en"):
        '''
        If no language is provided, use a language detector
        '''
        local_model = None
        try:
            local_model = self.get_lm_model(lang)
        except:
            logging.error("Fail to load the language model for language " + lang)

        if local_model is None:
            raise Exception("Failed to process language model for " + lang)
        
        text_scores = []

        if len(text) < 500:
            # we process the whole text segment
            text_scores.append(local_model.score_text(text))
        else:
            # we sample random segments
            for text_sample in local_model.read_text_sequence(text, max_length=600, samples=10):
                local_lang = cld3.get_language(text_sample)
                #print(local_lang)
                #print(str(local_lang.probability))
                if not local_lang.is_reliable or local_lang.language != lang or local_lang.proportion != 1.0 :
                    continue
                local_score = local_model.score_text(text_sample)
                text_scores.append(local_score)
        local_text_score = np.mean(text_scores)
        deviation = np.std(text_scores, dtype=np.float32)

        scorer_model = None
        try:
            scorer_model = self.get_scorer_model(lang)
        except:
            logging.error("Fail to load the scorer model for language " + lang)

        if scorer_model is None:
            raise Exception("Failed to process scorer model for language " + lang)

        X = np.zeros((len(text_scores), 1), dtype=np.float32) 
        for i in range(len(text_scores)):
            X[i,0]= (text_scores[i])
            #X[i,1]=  deviation
        
        #print(X)

        x_pred = xgb.DMatrix(X)

        final_text_scores = scorer_model.predict(x_pred)
        #print(final_text_scores)

        avg_text_score = np.mean(final_text_scores)
        max_text_score = np.max(final_text_scores)
        min_text_score = np.min(final_text_scores)
        deviation = np.std(final_text_scores, dtype=np.float32)

        boost_max = 1 / max_text_score
        boost_min = 0.1 / min_text_score

        # normalize
        if avg_text_score > 0.5:
            final_text_score = avg_text_score * boost_max
        else:
            final_text_score = avg_text_score * boost_min
        if final_text_score > 1.0:
            final_text_score = 1.0
        if final_text_score < 0.0:
            final_text_score = 0.0

        #print(final_text_score)

        return float(final_text_score)

    def score_txt_file(self, txt_file, lang=None):
        logging.info("processing text file: " + txt_file)
        if txt_file == None or not os.path.isfile(txt_file) or not txt_file.endswith(".txt"):
            print("issue")
            raise ValueError('Invalid input file: ' + txt_file)

        with open(txt_file, "r") as text_file:
            local_text = text_file.read()
            return self.score_text(local_text)
        
        return 0.0

    def score_pdf(self, pdf_file, lang=None):
        '''
        PDF file is parsed by external pdfalto tool. Spatial information can be used. 
        If no language is provided, use a language detector
        '''
        # convert pdf file
        logging.info("processing PDF file: " + pdf_file)
        pdfalto = PdfAltoWrapper('./data/pdfalto/lin64/pdfalto')
        output_path = os.path.join('./data/pdfalto/tmp/', binascii.b2a_hex(os.urandom(7)).decode() + ".xml")
        pdfalto.convert(pdf_file, output_path)
        logging.info("pdfalto conversion: " + output_path)

        local_text = filter_text(output_path)
        local_score = self.score_text(local_text)

        # cleaning ALTO file(s)
        if os.path.isfile(output_path):
            os.remove(output_path)
        output_path = output_path.replace(".xml", "_metadata.xml")
        if os.path.isfile(output_path):
            os.remove(output_path)

        return local_score

    def score_xml(self, xml_file, lang=None):
        '''
        Processing of XML file with text body section, such as ST.36 format or TEI format
        If no language is provided, use a language detector
        '''
        logging.info("processing XML file: " + xml_file)
        if xml_file == None or not os.path.isfile(xml_file) or not xml_file.endswith(".xml"):
            raise ValueError('Invalid input file: ' + txt_file)

        with open(txt_file, "r") as text_file:
            local_text = file.read()
            root = etree.fromstring(xml_string)
            text = etree.tostring(root, encoding='utf-8', method='text')
            text = text.decode()
            return self.score_text(text, lang)

        return 0.0

    def score_repository(self, repository):
        '''
        Score files in a repository. Supported file formats (by file extensions) are 
        any combination of text files (.txt), XML files (.xml) and PDF files (.pdf) 
        Return scores as a dict mapping file names to OCR quality score for the file.
        '''
        results = {}
        files_scores = []

        logging.info("processing repository: " + repository)

        if repository == None or not os.path.isdir(repository):
            raise ValueError('Invalid directory to be read: ' + target_dir)

        nb_files = 0
        for file in os.listdir(repository):
            logging.info("processing file: " + file)
            if file.endswith(".txt"):
                try:
                    results[file] = self.score_txt_file(os.path.join(repository, file))
                except:
                    logging.warning("Fail to score text file " + os.path.join(repository, file))

            elif file.endswith(".xml"):
                try:
                    results[file] = self.score_xml_file(os.path.join(repository, file))
                except:
                    logging.warning("Fail to score XML file " + os.path.join(repository, file))
            
            elif file.endswith(".pdf"):
                try:
                    results[file] = self.score_pdf_file(os.path.join(repository, file))
                except:
                    logging.warning("Fail to score PDF file " + os.path.join(repository, file))

            if file in results:
                files_scores.append(results[file])

            if nb_files > 100:
                break

            nb_files += 1

        print("\taverage score:", str(np.mean(files_scores)))
        print("\tlowest score:", str(np.min(files_scores)))
        print("\thighest score:", str(np.max(files_scores)))
        deviation = np.std(files_scores, dtype=np.float64)
        print("\tstandard deviation:", str(deviation))

        return results

    def train_scorer(self, lang):
        '''
        Train a scorer regression model, which uses the LM probability score as feature, 
        combined with others to produce a normalized score in [0,1] 
        '''

        x_pos, y_pos = self.load_positive_examples(lang)
        x_neg, y_neg = self.load_degraded_examples(lang)

        if len(x_neg) > len(x_pos):
            x_neg = x_neg[:len(x_pos)]
            y_neg = y_neg[:len(x_pos)]

        x_pos, y_pos = _remove_outliner(x_pos, y_pos)
        x_neg, y_neg = _remove_outliner(x_neg, y_neg)

        x = x_pos + x_neg
        y = y_pos + y_neg

        x, y = shuffle(x, y)

        print(x)
        print(y)

        #dtrain = xgb.DMatrix(x, label=y)

        #xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
        xgb_model.fit(x, y)

        self.scorers[lang] = xgb_model

    def save_scorer(self, lang):
        # save scorer
        save_path = os.path.join(self.config["models_dir"], lang, SCORER_FILE)

        model_xgb = self.get_scorer_model(lang)

        if model_xgb is not None:
            # save to JSON
            model_xgb.save_model(save_path)

    def load_scorer(self, lang):
        load_path = os.path.join(self.config["models_dir"], lang, SCORER_FILE)
        model_xgb = xgb.Booster()
        model_xgb.load_model(load_path)
        self.scorers[lang] = model_xgb

    def load_positive_examples(self, lang):
        x = []
        y = []
        text_scores = []

        local_model = None
        try:
            local_model = self.get_lm_model(lang)
        except:
            logging.error("Fail to load the model for language " + lang)

        if local_model is None:
            raise Exception("Failed to process language " + lang)

        start_time = time.time()
        for text in local_model.read_files_sequence(max_length=500, samples=None):
            text_scores.append(local_model.score_text(text))
        total_time = round(time.time() - start_time, 3)
        print("\nscored", str(len(text_scores)), "text segments in {:.3f}s".format(total_time)) 
        scores = np.array(text_scores)
        print("\taverage score:", str(np.mean(scores)))
        print("\tlowest score:", str(np.min(scores)))
        print("\thighest score:", str(np.max(scores)))
        deviation = np.std(scores, dtype=np.float64)

        for i in range(len(scores)):
            features = []
            # LM probability of the sequence
            features.append(scores[i])
            # general standard deviation
            #features.append(deviation)

            x.append(features)
            y.append(1.0)
        return x, y

    def load_degraded_examples(self, lang):
        x = []
        y = []

        local_model = None
        try:
            local_model = self.get_lm_model(lang)
        except:
            logging.error("Fail to load the model for language " + lang)

        if local_model is None:
            raise Exception("Failed to process language " + lang)

        start_time = time.time()
        text_scores = []
        target_dir = os.path.join(self.config['training_dir'], lang, "ocr")
        nb_file = 0
        for file in os.listdir(target_dir):
            if file.endswith(".txt"):
                print(file)
                i = 0
                for text in local_model.read_file_sequence(target_file=os.path.join(target_dir, file), 
                                                    max_length=500, samples=None):
                    text_scores.append(local_model.score_text(text))
                    i += 1
                    if i>200:
                        break
            if nb_file > 10:
                break
            nb_file += 1
        total_time = round(time.time() - start_time, 3)
        print("\nscored", str(len(text_scores)), "text segments in {:.3f}s".format(total_time)) 
        scores = np.array(text_scores)
        print("\taverage score:", str(np.mean(scores)))
        print("\tlowest score:", str(np.min(scores)))
        print("\thighest score:", str(np.max(scores)))
        deviation = np.std(scores, dtype=np.float64)

        for i in range(len(scores)):
            features = []
            # LM probability of the sequence
            features.append(scores[i])
            # general standard deviation
            #features.append(deviation)

            x.append(features)
            y.append(0.0)
        return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Simple command line OCR scorer. Use the service for more intensive/pipeline tasks.")
    parser.add_argument("--config-file", type=str, required=False, help="configuration file to be used", default='./config.yml')
    parser.add_argument("--debug", action="store_true", required=False, default=False,
                        help="activate the debug mode (override the config file logging parameter)")
    parser.add_argument("--text-file", type=str, required=False, help="text file to be analyzed, expected encoding is UTF-8")
    parser.add_argument("--pdf-file", type=str, required=False, help="PDF file to be analyzed")
    parser.add_argument("--xml-file", type=str, required=False, help="XML file to be analyzed, with text body section")
    parser.add_argument("--repository", type=str, required=False, help="a repository of text/XML/PDF files to be evaluated")

    args = parser.parse_args()

    debug = args.debug
    text_file = args.text_file
    pdf_file = args.pdf_file
    xml_file = args.xml_file
    config_file = args.config_file
    repository = args.repository

    try:
        scorer = OCRScorer(config_file)
        if pdf_file != None:
            print(scorer.score_pdf(pdf_file))
        elif text_file != None:
            print(scorer.score_pdf(text_file))
        elif xml_file != None:
            print(scorer.score_patent_xml(patent_xml_file))
        elif repository != None:
            results = scorer.score_repository(repository)
            print(results)
        else:
            print("At least one file to be evaluated must be provided\n")
            parser.print_help()
            exit(1)
    except Exception as e:
        print("Scorer failed: ", str(e))
        exit(1)
