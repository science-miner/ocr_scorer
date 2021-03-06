# OCR Scorer

This tool aims at evaluating the quality of the OCR for any text sources, without prior knowledge of the usage of an OCR or not, and in case of OCR usage without prior knowledge about the OCR used and its reliability. 

We focus on technical and scientific texts/documents. The typical scenario is text-mining on thousand/millions of scholar PDF, where many documents might have been OCRized decades ago with unknown quality. Detecting low quality OCRized documents makes possible to re-OCRize them with modern tools and to apply text mining tools without considerable accuracy drop. However, the tool can be adapted and retrained easily to other types of documents and domains. 

The approach is based on a RNN Language Model (LM) learned from a corpus of technical and scientific texts in digital native form (no OCR). Character LM approach for OCR evaluation has been experimented in particular in (Popat, 2009), showing significantly more reliable accuracy than dictionary-based approach. While (Popat, 2009) was using N-gram character model, in this work we use stronger character LM based on LSTM. Character LM supports open vocabulary which is adapted to OCR scoring. RNN offers manageable and close to SOTA implementation for Character LM. The normalized probability of OCRized text against the LM provides a basis for quality score for the OCR. The LM probability of OCR text sequences is then used as feature by a Gradient Boosted Trees regression model (XGBoost), optionally combined with other features, to produce a normalized quality score in `[0,1]`.

The OCR Scorer can be used as Python command line or as a web service. A docker image is available. 

LM are language-specific. Build-in languages are currently English, French and German, all trained with technical and scientific documents. To add more languages or models for new domains and document types, see [below](#adding-new-languages-and-models).

## Requirements and install

The present tool is implemented in Python and should work correctly with Python 3.7 or higher. It requires Keras/TensorFlow >2.0. 

Get the github repo:

```console
git clone https://github.com/science-miner/ocr_scorer
cd ocr_scorer
```
It is strongly advised to setup first a virtual environment to avoid falling into one of these gloomy python dependency marshlands - you can adjust the version of Python to be used, but be sure to be 3.7 or higher:

```console
virtualenv --system-site-packages -p python3.8 env
source env/bin/activate
```

Install the dependencies:

```console
pip3 install -r requirements.txt
```

Finally install the project in editable state

```console
pip3 install -e .
```

### Start the service

The OCR Scorer Web API service is implemented with [FastAPI](https://fastapi.tiangolo.com) and can be started as follow on default port `8050`:  

```console
python3 ocr_scorer/service.py --config my_config.yml
```

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

### Command line

```
$ python3 ocr_scorer/ocr_scorer.py --help
usage: ocr_scorer.py [-h] [--config-file CONFIG_FILE] [--debug] [--text-file TEXT_FILE]
                     [--pdf-file PDF_FILE] [--xml-file XML_FILE] [--repository REPOSITORY]

Simple command line OCR scorer. Use the service for more intensive/pipeline tasks.

optional arguments:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE
                        configuration file to be used
  --debug               activate the debug mode (override the config file logging parameter)
  --text-file TEXT_FILE
                        text file to be analyzed, expected encoding is UTF-8
  --pdf-file PDF_FILE   PDF file to be analyzed
  --xml-file XML_FILE   XML file to be analyzed, with text body section
  --repository REPOSITORY
                        a repository of text/XML/PDF files to be evaluated
```

### Using the docker image

#### Build the docker image

To build the docker image for the OCR scorer service, from the root project directory where the Dockerfile is located:

```console
docker build -t ocr_scorer/scorer .
```

#### Start the service with Docker

To run the container on port `8080` of the docker host machine, with the default configuration (i.e. using the `config.yml` service config file present in the project at build time, with default port `8050`):

```console
docker run -it --rm -p 8080:8050 ocr_scorer/scorer
```

It is possible to supply the local `config.yml` file when running the image as follow:

```console
docker run -it --rm -p 8080:8050 --mount type=bind,source=$(pwd)/ocr_scorer/config.yml,target=/opt/ocr_scorer/config.yml ocr_scorer/scorer
```

Or using volume:

```console
docker run -it --rm -p 8080:8050 -v $(pwd)/ocr_scorer/config.yml:/opt/ocr_scorer/config.yml:ro ocr_scorer/scorer
```

The same image can thus easily use specific configuration at deployment time, including when deployed on a Kubernetes cluster using the `hostPath` volume.


### Use the service

Once the service is started as described in the previous sections, the web service API documnetation is available at `http(s)://*host*:*port*/docs`, e.g. for instance `http://localhost:8050/docs`, based on Swagger, and `http://localhost:8050/redoc` for ReDoc documentation style. These documentations offer interactive support to support test queries. 

### Add new languages and train LM models

Current scorer language models available in this repository are:
- for English, German and French languages only
- and for scientific article and patents only

For covering other languages and other type of textual content, training new models is necessary. 

To train a language model for any languages, the following resources are needed:

- Text content in `.txt` files encoded in Unicode under `data/texts/xx/training` where `xx` is the two-letter ISO 639-1 code of the language. The amount of text content must be superior to 1M characters at least, otherwise it might cause issues in getting the training converging, 

To evaluate a language model for an existing trained language:

- Text content in `.txt` files encoded in Unicode under `data/texts/xx/evaluation` where `xx` is the two-letter ISO 639-1 code of the language. Any amount of text is possible. 

To train a language model:

```console
python3 ocr_scorer/lm_scorer.py --lang en train
```

This will train the language model for English using the text content under `data/texts/en/training`.

To evaluate the language model:

```console
python3 ocr_scorer/lm_scorer.py --lang en evaluate
```

This will evaluate the trained model for the indicated language using the text content under `data/texts/en/evaluation`. The evaluation is giving the accuracy of next character predictions in the evaluation data and the BPC (Bits Per Character) tradditionally used in LM.

To train an XGBoost scorer based on the LM for a given language:

```console
python3 ocr_scorer/lm_scorer.py train_scorer --lang fr
```

The XGBoost scorer requires that a language model has been trained for the target language. It will use the text content under `data/texts/en/evaluation` as high quality positive samples and examples of OCR texts with degraded quality under `data/texts/en/ocr`.

## Implementation

Keras/TensorFlow character language model implementation using as architecture a 2 layers vanilla LSTM.

```
number of chars/vocab size: 216
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm (LSTM)                 (512, 128, 217)           377580    
                                                                 
 dropout (Dropout)           (512, 128, 217)           0         
                                                                 
 lstm_1 (LSTM)               (512, 128)                177152    
                                                                 
 dropout_1 (Dropout)         (512, 128)                0         
                                                                 
 dense (Dense)               (512, 217)                27993     
                                                                 
=================================================================
Total params: 582,725
Trainable params: 582,725
Non-trainable params: 0
_________________________________________________________________

```

The final scorer is a Gradient Boosted Trees regression model (XGBoost) taking the language model scores as input features from any numbers of text segments. It outputs a single quality score in [0,1] from these observations.

### Training volumes

Half patent text, half non-patent:

|  language | # files  | # charcaters  |  
|---        |---       |---            |
|  en       |   14     |  1,244,273    |  
|  de       |   13     |  1,076,636    |   
|  fr       |   14     |  1,121,353    |   

Training a LM takes around 1 day (nvidia 1080Ti)


### Example of evaluation of the LM


English LM:

```

evaluating language model...
489/489 [==============================] - 59s 119ms/step - loss: 2.2441 - accuracy: 0.4108

evaluation: accuracy = 0.41078, (59.105s)
bpc: 3.237501513744312

scored 166 text segments in 31.826s
    average score: 0.11354362231513442
    lowest score: 0.025512250250879644
    highest score: 0.47485065974032054
    standard deviation: 0.06443119909075011

scored 7 publisher files in 31.457s
    average score: 0.11343442670871337
    lowest score: 0.07829285503898774
    highest score: 0.1350078012924938
    standard deviation: 0.022655542419996694

scoring OCRized files...
scored 1 files in 6.755s
    average score: 0.04420166675978527
```

French LM:

```
evaluating language model...
256/256 [==============================] - 31s 119ms/step - loss: 2.2387 - accuracy: 0.4137

evaluation: accuracy = 0.41367, (31.093s)
bpc: 3.2297602306636755

scoring sequence evaluation...
     scoring segments...

scored 70 text segments in 13.411s
    average score: 0.13888790959352967
    lowest score: 0.03434812310117611
    highest score: 0.4403852463311049
    standard deviation: 0.06215460833641291

scored 5 publisher files in 13.212s
    average score: 0.149004661743621
    lowest score: 0.10973474930334305
    highest score: 0.2031337353617884
    standard deviation: 0.03833099631899638

scoring OCRized files...
scored 124 files in 631.416s
    average score: 0.05663416663869246
    lowest score: 0.03305310297788116
    highest score: 0.08385362714866659
    standard deviation: 0.009110809890462812
```

## How current SOTA OCR are scored? 

Current OCR engines can be considered as very reliable when we consider the overall OCR quality in the last 30 years. Scientific documents OCRized in the nineties for instance will be scored much lower than currently OCR-ized documents. In the following table, we compare average scoring produced by modern OCR tools (Abbyy and Tesseract 4) with the digital native version of documents. 

* Non-patent documents: the collection is a set of 2,000 full text PDF articles from various sources (PMC, arXiv, Hindawi, bioRxiv)

|  origin           | # files  | avg. OCR quality score  | 
|---                |---       |---                      |
|  digital native   |    2000  |      0.789              |       
|  Abbyy OCR        |    2000  |      0.751              |        
|  Tessearct 4 OCR  |    2000  |      0.718              |         


* Patent documents: the collection is a set of 500 patent XML EP publication from the European Patent Office, downloaded from Google Patents. Although these "original" full texts are derived from OCR of the patent applications as filed at the patent office, they should be highly accurate thanks to systematic manual OCR corrections before publication. 

|  origin           |  # files  | avg. OCR quality score |  
|---                |---       |---                   |
| Original PDF (from Google Patents) | 500 | 0.798    |
|  Abbyy OCR        |   500    |     0.776            |
|  Tessearct 4 OCR  |   500    |     0.752            |


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
