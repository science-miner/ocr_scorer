# OCR Scorer

This tool aims at evaluate the quality of the OCR for any text sources, without prior knowledge of the usage of an OCR or not, and in case of OCR usage without prior knowledge about the OCR used. We focus on technical and scientific texts/documents. The typical scenario is text mining on thousand/millions of scholar PDF, where many documents might have been OCRized decades ago and present extremely low quality. Detecting low quality OCRized documents make possible to re-OCRize them with modern tools and to apply text mining tools without considerable accuracy drop. However the tool can be adapted and retrained easily to other types of documents. 

The approach is based on a RNN Language Model (LM) learned from a corpus of technical and scientific texts in digital native form (no OCR). LM approach for OCR evaluation has been experimented in particular in (Popat, 2009), showing significantly more reliable accuracy than dictionary-based approach. While (Popat, 2009) was using N-gram character model, in this work, we use stronger character LM based on LSTM. Character LM supports open vocabulary which is adapted to OCR scoring. RNN offers manageable and close to SOTA implementation for Character LM. The normalized probability of OCRized text against the LM provides a quality score for the OCR.   

The OCR Scorer can be used as Python command line or as a web service. A docker image is available. 

LM are language-specific. Build-in languages are currently English, French and German. To add more languages, see [below](#adding-new-languages).

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

> python3 ocr_scorer/service.py --config my_config.yaml

```
...
```

The documentation of the service is available at `http(s)://*host*:*port*/docs`, e.g. `http://localhost:8050/docs` (based on Swagger), for ReDoc documentation style, use `http://localhost:8050/redoc`).


### Adding new languages

...

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
