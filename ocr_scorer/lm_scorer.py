import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
tf.autograph.set_verbosity(1)
tf.get_logger().setLevel("ERROR")

import numpy as np
import random
import sys
import argparse
import re
import json
import time
import math

import logging
import logging.handlers

from utils import _load_config, _normalize_text_for_scoring
from unicode_utils import normalize_text

# default logging settings, will be override by config file
logging.basicConfig(filename='client.log', filemode='w', level=logging.DEBUG)

'''
Language model for scoring sequence with vanilla LSTM
'''
class LMScorer(object):

    lang = None
    model = None
    config = None
    config_path = None
    chars = []
    char_indices = None
    use_spatial_information = False

    # model parameters (to be managed with a dedicated config)
    batch_size = 512
    max_length = 128
    epochs = 60
    voc_size = 0
    UNK = 0

    # scorer normalization/calibration parameters, use to normalize
    # and project scores between [0,1] based on examples
    scorer_parameters = {}
    

    def __init__(self, lang, config_path=config_path, use_spatial_information=False):

        self.lang = lang
        self.config = _load_config(config_path)
        self.config_path = config_path
        self.use_spatial_information = use_spatial_information

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

        # we use a static list of unicode points for indo-european languages and thus a fixed char vocab
        #target_dir = self.config['unicode_dir']
        with open(os.path.join(self.config['unicode_dir'], "unicode_table.txt"), "r") as fu:
            for line in fu.readlines():
                if len(line) == 0:
                    continue
                if line.startswith("#") and line.find("\t") == -1:
                    continue
                char = line[0]
                # keep the character in line with the normalization
                # if character is normalized into another character, it is not added to the vocabulary
                char = normalize_text(char)
                if not char in self.chars:
                    self.chars.append(char)

        self.voc_size = len(self.chars) + 1
        self.UNK = len(self.chars)
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))

        print("number of chars/vocab size:", str(len(self.chars)))
        self.build_model()

    def score_text(self, text):
        # score a single piece of text, the text is expected to be of length at least self.max_length+1
        score = 0
        pos = 0
        segments = []
        next_chars = []
        text = _normalize_text_for_scoring(text)
        #print(text)

        # hacky but it work not badly - in case of short text, we concatenate the same text to reach a full
        # input size and avoid unreliable prediction scores of shortest text
        while len(text) < (self.max_length*2)+1:
            text += ' ' + text
        #print("extended length:", str(len(text)))

        preds = []
        while pos < len(text)-(self.max_length+1):
            segments.append(text[pos:pos+self.max_length])
            next_chars.append(text[pos+self.max_length+1])
            if len(segments) == self.batch_size:
                preds += self.predict_batch(segments, next_chars, self.batch_size)
                segments = []
                next_chars = []
            pos += 1

        # process last (incomplete) batch
        if len(segments)>0:
            preds += self.predict_batch(segments, next_chars, self.batch_size)
        if len(preds) == 0:
            return 0.0

        # move to log for additive semiring
        log_pred = np.log(np.array(preds))
        total_pred = np.sum(log_pred)
        # back to probability
        return np.exp(total_pred/len(preds))

    def predict_batch(self, segments, next_chars, local_batch_size):
        local_preds = []
        X = np.zeros(shape=(local_batch_size, self.max_length, self.voc_size), dtype=np.float32)
        actual_batch_size = min(local_batch_size, len(segments))
        for batch_idx in range(0, actual_batch_size):
            for i in range(0, self.max_length):
                if not segments[batch_idx][i] in self.chars:
                    X[batch_idx, i, self.UNK] = 1
                else:
                    X[batch_idx, i, self.char_indices[segments[batch_idx][i]]] = 1
        # see https://stackoverflow.com/questions/63489433/tensorflow-keras-appears-to-change-my-batch-size
        # for weird predict chaging the batch size of the input 
        predictions = self.model.predict(X, batch_size=local_batch_size)

        # get probabilities for actual next chars
        sum_pred = 0.0
        for batch_idx in range(0, actual_batch_size):
            #print(predictions[batch_idx])
            #print("target char:", next_chars[batch_idx], str(self.char_indices[next_chars[batch_idx]]))
            if not next_chars[batch_idx] in self.chars:
                target = self.UNK
            else:
                target = self.char_indices[next_chars[batch_idx]]
            local_preds.append(predictions[batch_idx,target])

        return local_preds

    def score_texts(self, texts):
        # score a set of texts
        result_preds = []
        for text in texts:
            result_preds.append(self.score_text(text))
        return result_preds

    def read_batch(self, training=True, batch_size=None, max_length=128):
        '''
        Read successively batches of data from a set of files. 
        If parameter training is True (default), the training data is read, otherwise the evaluation data.
        '''
        if batch_size == None:
            batch_size = self.batch_size

        # we use all .txt files in the data repository corresponding to the language
        if training:
            target_dir = os.path.join(self.config['training_dir'], self.lang, "training")
        else:
            target_dir = os.path.join(self.config['training_dir'], self.lang, "evaluation")
        for file in os.listdir(target_dir):
            if file.endswith(".txt"):
                with open(os.path.join(target_dir, file), "r") as text_file:
                    pos = 0
                    segments = []
                    next_chars = []
                    text = text_file.read()
                    text = _normalize_text_for_scoring(text)
                    #print(text)
                    while pos < len(text)-(self.max_length+1):
                        segments.append(text[pos:pos+self.max_length])
                        next_chars.append(text[pos+self.max_length+1])
                        pos += 1

                        if len(segments) == batch_size:
                            X = np.zeros((batch_size, self.max_length, self.voc_size))
                            Y = np.zeros((batch_size, self.voc_size))

                            for batch_idx in range(0, batch_size):
                                for i in range(0, self.max_length):
                                    if not segments[batch_idx][i] in self.chars:
                                        X[batch_idx, i, self.UNK] = 1
                                    else:
                                        X[batch_idx, i, self.char_indices[segments[batch_idx][i]]] = 1
                                if not next_chars[batch_idx] in self.chars:
                                    Y[batch_idx, self.UNK] = 1
                                else:
                                    Y[batch_idx, self.char_indices[next_chars[batch_idx]]] = 1
                            yield X, Y

                            # back here for next call
                            segments = []
                            next_chars = []

    def read_files_sequence(self, target_dir=None, max_length=500, samples=10):
        # we use all .txt files in the target data repository, defaulting to the language evaluation ones
        # samples: max number of random sample text segments to be consider per text file, default is 10 
        # if samples is None, all the sequences are returned
        if target_dir == None:
            target_dir = os.path.join(self.config['training_dir'], self.lang, "evaluation")
        elif not os.path.isdir(target_dir):
            raise ValueError('Invalid directory to be read: ' + target_dir)

        for file in os.listdir(target_dir):
            if file.endswith(".txt"):
                with open(os.path.join(target_dir, file), "r") as text_file:
                    pos = 0
                    text = text_file.read()
                    text = _normalize_text_for_scoring(text)

                    if samples is not None:
                        total_segments = len(text) / max_length
                        ratio = int(total_segments / samples)
                        if ratio == 0:
                            # case file length is lower than max_length, we take everything
                            ratio = 1

                    while True:
                        upper_bound = min((pos*max_length)+max_length, len(text))
                        segment = text[pos*max_length:upper_bound]
                        if samples is None or pos % ratio == 0:
                            yield segment
                        pos += 1
                        if upper_bound >= len(text):
                            break

    def read_text_sequence(self, text, max_length=500, samples=10):
        # samples: max number of random sample text segments to be consider in the text, default is 10 
        # if samples is None, all the sequences are returned
        pos = 0
        text = _normalize_text_for_scoring(text)

        if samples is not None:
            total_segments = len(text) / max_length
            ratio = int(total_segments / samples)

        while True:
            upper_bound = min((pos*max_length)+max_length, len(text))
            segment = text[pos*max_length:upper_bound]
            if samples is None or pos % ratio == 0:
                yield segment
            pos += 1
            if upper_bound >= len(text):
                break

    def read_file_sequence(self, target_file=None, batch_size=None, max_length=500, samples=10):
        # samples: max number of random sample text segments to be consider in the text file, default is 10 
        # if samples is None, all the sequences are returned
        if target_file == None:
            raise ValueError('Unspecified file to be read')

        if not os.path.isfile(target_file):
            raise ValueError('Invalid file to be read: ' + target_file)

        if not target_file.endswith(".txt"):
            raise ValueError('File must be a .txt file: ' + target_file)

        with open(target_file, "r") as text_file:
            pos = 0
            text = text_file.read()
            text = _normalize_text_for_scoring(text)

            if samples is not None:
                total_segments = len(text) / max_length
                ratio = int(total_segments / samples)
                if ratio == 0:
                    # case file length is lower than max_length, we take everything
                    ratio = 1

            while True:
                upper_bound = min((pos*max_length)+max_length, len(text))
                segment = text[pos*max_length:upper_bound]
                if samples is None or pos % ratio == 0:
                    yield segment
                pos += 1
                if upper_bound >= len(text):
                    break

    def build_model(self):
        '''
        vanilla LSTM layers

        stateful RNN makes sense in long sequence and anomaly detection, but it is then required to
        fix the batch size in the input layer 
        '''
        self.model = keras.Sequential(
            [
                keras.Input(shape=(self.max_length, self.voc_size), batch_size=self.batch_size),
                layers.LSTM(self.voc_size, recurrent_dropout=0.5, return_sequences=True, stateful=True), 
                layers.Dropout(0.5),
                #layers.LSTM(128, recurrent_dropout=0.2, return_sequences=True, stateful=True), 
                #layers.Dropout(0.2),
                layers.LSTM(128, recurrent_dropout=0.2, stateful=True), 
                layers.Dropout(0.2),
                layers.Dense(self.voc_size, activation="softmax"),
            ]
        )
        #optimizer = keras.optimizers.RMSprop(learning_rate=0.01, clipnorm=1)
        optimizer = keras.optimizers.Adam(learning_rate=0.01, clipnorm=1)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
        self.model.summary()

    def train(self):
        metrics = None
        start_time = time.time()
        total_time = 0
        print("\ntraining language model...")
        best_avg_loss = 10000
        best_model = None
        best_epoch = 0
        for epoch in range(self.epochs):
            start_epoch_time = time.time()
            losses = []
            accs = []
            for i, (X, Y) in enumerate(self.read_batch(max_length=self.max_length)):
                
                loss, acc = self.model.train_on_batch(X, Y, reset_metrics=True)
                if (i+1) % 10 == 0: 
                    sys.stdout.write("\r - batch {}:\tloss = {:.4f}\t acc = {:.5f}\t({:.3f}s)".format(i + 1, loss, acc, (time.time() - start_epoch_time)))
                    sys.stdout.flush()

                losses.append(loss)
                accs.append(acc)
                
            epoch_time = round(time.time() - start_epoch_time, 3)
            total_time = round(time.time() - start_time, 3)
            new_loss = np.average(losses)
            print("\repoch {} - loss = {:.4f}, acc = {:.5f}, ({:.3f}s/{:.3f}s)".format(epoch, new_loss, np.average(accs), epoch_time, total_time))

            if best_model == None or best_avg_loss > new_loss:
                self.save()
                best_avg_loss = new_loss
                best_epoch = epoch

        # load best model
        self.load()
        bpc = best_avg_loss / tf.constant(math.log(2))
        sys.stdout.write("bpc: ")
        tf.print(bpc, output_stream=sys.stdout)
        sys.stdout.write("\n")

        print("best model: epoch", str(best_epoch))

    def evaluate(self):
        
        print("LM prediction evaluation...")
        start_time = time.time()
        print("\nevaluating language model...")
        loss, acc = self.model.evaluate(self.read_batch(max_length=self.max_length, training=False))
        total_time = round(time.time() - start_time, 3)
        print("\nevaluation: accuracy = {:.5f}, ({:.3f}s)".format(acc, total_time))
        bpc = loss / math.log(2)
        print("bpc:", bpc)
        
        print("\nscoring sequence evaluation...")
        start_time = time.time()
        print("     scoring segments...")
        text_scores = []
        for text in self.read_files_sequence(max_length=500, samples=20):
            text_scores.append(self.score_text(text))
        total_time = round(time.time() - start_time, 3)
        print("\nscored", str(len(text_scores)), "text segments in {:.3f}s".format(total_time)) 
        scores = np.array(text_scores)
        print("\taverage score:", str(np.mean(scores)))
        print("\tlowest score:", str(np.min(scores)))
        print("\thighest score:", str(np.max(scores)))
        deviation = np.std(scores, dtype=np.float64)
        print("\tstandard deviation:", str(deviation))

        print("\nscoring file evaluation...")
        start_time = time.time()
        print("     scoring files...")
        files_scores = []
        target_dir = os.path.join(self.config['training_dir'], self.lang, "evaluation")
        for file in os.listdir(target_dir):
            if file.endswith(".txt"):
                print(file)
                text_scores = []
                for text in self.read_file_sequence(target_file=os.path.join(target_dir, file), 
                                                    max_length=500, samples=20):
                    text_scores.append(self.score_text(text))
                local_file_score = np.mean(text_scores)
                files_scores.append(local_file_score)
                print("score for file", file, ":", str(local_file_score))
        total_time = round(time.time() - start_time, 3)
        print("\nscored", str(len(files_scores)), "files in {:.3f}s".format(total_time)) 
        scores = np.array(files_scores)
        print("\taverage score:", str(np.mean(files_scores)))
        print("\tlowest score:", str(np.min(files_scores)))
        print("\thighest score:", str(np.max(files_scores)))
        deviation = np.std(files_scores, dtype=np.float64)
        print("\tstandard deviation:", str(deviation))
        
        print("\nscoring OCRized files...")
        start_time = time.time()
        print("     scoring files...")
        files_scores = []
        target_dir = os.path.join(self.config['training_dir'], self.lang, "ocr")
        for file in os.listdir(target_dir):
            if file.endswith(".txt"):
                print(file)
                text_scores = []
                i = 0
                for text in self.read_file_sequence(target_file=os.path.join(target_dir, file), 
                                                    max_length=500, samples=20):
                    text_scores.append(self.score_text(text))
                    i += 1
                    if i>200:
                        break
                local_file_score = np.mean(text_scores)
                files_scores.append(local_file_score)
                print("score for file", file, ":", str(local_file_score))
        total_time = round(time.time() - start_time, 3)
        print("\nscored", str(len(files_scores)), "files in {:.3f}s".format(total_time)) 
        if len(files_scores)>0:
            print("\taverage score:", str(np.mean(files_scores)))
            print("\tlowest score:", str(np.min(files_scores)))
            print("\thighest score:", str(np.max(files_scores)))
            deviation = np.std(files_scores, dtype=np.float64)
            print("\tstandard deviation:", str(deviation))

    def train_scorers(self, langs=None):
        # lazy loading to avoid circular import 
        from ocr_scorer import OCRScorer, supported_languages
        if langs == None:
            langs = supported_languages

        for lang in langs:
            local_scorer = OCRScorer(self.config_path)
            local_scorer.train_scorer(lang)
            local_scorer.save_scorer(lang)

    def save(self):
        target_dir = os.path.join(self.config["models_dir"], self.lang)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        # save weights
        self.model.save_weights(os.path.join(target_dir, 'weights.h5'))

        # save vocab
        #with open(os.path.join(target_dir, 'vocabulary.json'), 'w') as json_file:
        #    json.dump(self.char_indices, json_file)

    def load(self,):
        target_dir = os.path.join(self.config["models_dir"], self.lang)

        # load weights
        self.model.load_weights(os.path.join(target_dir, 'weights.h5'))

        # load vocab
        #with open(os.path.join(target_dir, 'vocabulary.json')) as json_file:
        #    self.char_indices = json.load(json_file)

        #self.chars = [c for c, i in enumerate(self.char_indices)]

if __name__ == '__main__':

    possible_actions = [ 'train', 'eval', 'test', 'train_scorers', 'train_scorer' ]

    parser = argparse.ArgumentParser(
        description="Simple command line OCR scorer. Use the service for more intensive/pipeline tasks.")
    parser.add_argument("action", help="one of " + str(possible_actions), choices=possible_actions)
    parser.add_argument("--config-file", type=str, required=False, help="configuration file to be used", default='./config.yml')
    parser.add_argument("--lang", type=str, required=False, help="language code (two letters ISO 639-1) of the model", default='en')

    args = parser.parse_args()

    action = args.action
    config_file = args.config_file
    lang = args.lang

    model = LMScorer(lang, config_file, False)
    
    if action == 'train':
        model.train()
        model.save()
    
    if action == 'train_scorers':
        model.train_scorers()

    if action == 'train_scorer':
        model.train_scorers([model.lang])

    if action == 'eval':
        model.load()
        model.evaluate()
    
    if action == 'test':
        model.load()
        
        example1 = "This is an example that might be a little short, but will be sufficent as a text."
        print(example1)
        score = model.score_text(example1)
        print(str(score))
        
        example2 = "This is a examble tha might bee a little shirt, bot wall be subicent as a tax. This is an example that might be a little short, but will be sufficent as a text. "
        print(example2)
        score = model.score_text(example2)
        print(str(score))

        example3 = "strange-quark fragmentation. While similar, the shapes are not nearly identical. While the similarity is an interesting observation, it may be coincidental, and likely can only be disentangled in a full QCD analysis, which is at this point "
        print(example3)
        score = model.score_text(example3)
        print(str(score))

