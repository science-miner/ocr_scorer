# OCR Scorer

This tool aims at evaluating the quality of the OCR for any text sources, without prior knowledge of the usage of an OCR or not, and in case of OCR usage without prior knowledge about the OCR used. 

We focus on technical and scientific texts/documents. The typical scenario is text-mining on thousand/millions of scholar PDF, where many documents might have been OCRized decades ago and present unknown quality. Detecting low quality OCRized documents make possible to re-OCRize them with modern tools and to apply text mining tools without considerable accuracy drop. However, the tool can be adapted and retrained easily to other types of documents and domains. 

The approach is based on a RNN Language Model (LM) learned from a corpus of technical and scientific texts in digital native form (no OCR). LM approach for OCR evaluation has been experimented in particular in (Popat, 2009), showing significantly more reliable accuracy than dictionary-based approach. While (Popat, 2009) was using N-gram character model, in this work, we use stronger character LM based on LSTM. Character LM supports open vocabulary which is adapted to OCR scoring. RNN offers manageable and close to SOTA implementation for Character LM. The normalized probability of OCRized text against the LM provides a quality score for the OCR.   

The OCR Scorer can be used as Python command line or as a web service. A docker image is available. 

LM are language-specific. Build-in languages are currently English, French and German. To add more languages or models for new domains and document types, see [below](#adding-new-languages-and-models).

## Implementation

Keras/TensorFlow character language model implementation using as architecture:

- a 2 layers vanilla LSTM with (dynamic) Monte Carlo Dropout

## Requirements and install

The present tool is implemented in Python and should work correctly with Python 3.7 or higher. It requires Keras/TensorFlow >2.0. 

Get the github repo:

```sh
git clone https://github.com/science-miner/ocr_scorer
cd ocr_scorer
```
It is strongly advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands - you can adjust the version of Python to be used, but be sure to be 3.7 or higher:

```sh
virtualenv --system-site-packages -p python3.8 env
source env/bin/activate
```

Install the dependencies:

```sh
pip3 install -r requirements.txt
```

Finally install the project in editable state

```sh
pip3 install -e .
```

### Start the service

The OCR Scorer Web API service is implemented with [FastAPI](https://fastapi.tiangolo.com) and can be started as follow:  

> python3 ocr_scorer/service.py --config my_config.yml

```
logs are written in client.log
INFO:     Started server process [60427]
INFO:     Waiting for application startup.
  ___   ____ ____                                     _    ____ ___ 
 / _ \ / ___|  _ \   ___  ___ ___  _ __ ___ _ __     / \  |  _ \_ _|
| | | | |   | |_) | / __|/ __/ _ \| '__/ _ \ '__|   / _ \ | |_) | | 
| |_| | |___|  _ <  \__ \ (_| (_) | | |  __/ |     / ___ \|  __/| | 
 \___/ \____|_| \_\ |___/\___\___/|_|  \___|_|    /_/   \_\_|  |___|
                                                                    

INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8050 (Press CTRL+C to quit)
```

The documentation of the service is available at `http(s)://*host*:*port*/docs`, e.g. `http://localhost:8050/docs` (based on Swagger), for ReDoc documentation style, use `http://localhost:8050/redoc`).


### Use the service

Once the service is started as described in the previous section, the web service API documnetation is available at at `http(s)://*host*:*port*/docs`, e.g. `http://localhost:8050/docs`, based on Swagger, and `http://localhost:8050/redoc` for ReDoc documentation style. These documentations offer interactive support to support test queries. 


### Add new languages and train LM models

Current available scorer language models are:
- for English, German and French languages only
- and for scientific article and patents only

For covering other languages and other type of textual content, training new models is necessary. 

To train a language model for any languages, the following resources are needed:

- Text content in `.txt` files encoded in Unicode under `data/texts/xx/training` where `xx` is the two-letter ISO 639-1 code of the language. The amount of text content must be superior to 1M characters at least, otherwise it might cause issues in getting the training converging, 

To evaluate a language model for an existing trained language:

- Text content in `.txt` files encoded in Unicode under `data/texts/xx/evaluation` where `xx` is the two-letter ISO 639-1 code of the language. Any amount of text is possible. 




To train a language model:


```
python3 ocr_scorer/lm_scorer.py --lang en train
```

This will train the language model for English using the text content under `data/texts/en/training`.

The evaluate a language model:


```
python3 ocr_scorer/lm_scorer.py --lang en evaluate
```

This will evaluate the trained model for the indicated language using the text content under `data/texts/en/evaluation`. The evaluation is giving the accuracy of next character predictions in the evaluation data and the BPC (Bits Per Character) tradditionally used in LM.



## References


```
[Popat, 2009] A Panlingual Anomalous Text Detector. Ashok C. Popat. 
DocEng '09: Proceedings of the 9th ACM symposium on Document Engineering, 
ACM, New York (2009), pp. 201-204
```

## License

The OCR Scorer implementation is distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). 

The documentation of the project is distributed under [CC-0](https://creativecommons.org/publicdomain/zero/1.0/) license and the possible annotated data under [CC-BY](https://creativecommons.org/licenses/by/4.0/) license.

If you contribute to the OCR Scorer project, you agree to share your contribution following these licenses. 

Contact: Patrice Lopez (<patrice.lopez@science-miner.com>)
