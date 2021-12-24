# OCR Scorer

This tool tries to evaluate the quality of the OCR for any texts, without prior knowledge of the used OCR. The approach is based on a RNN Language Model (LM) learned from a corpus of technical and scientific texts in digital native form (no OCR). This work is inspired by (Popat, 2009), but use stronger LM based on LSTM instead of an N-gram model. The LM is language-specific. The normalized probability of OCRized text against the LM provides a quality score for the OCR, more reliable than a dictionary-based approach.  

The OCR Scorer can be used as Python command line or as a web service. A docker image is available. 

Build-in languages are currently English, French and German. To add more languages...

## Implementation

This is a Keras/TensorFlow character language model implementation using as architecture:

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

## License

The OCR Scorer implementation is distributed under [Apache 2.0 license](http://www.apache.org/licenses/LICENSE-2.0). 

The documentation of the project is distributed under [CC-0](https://creativecommons.org/publicdomain/zero/1.0/) license and the possible annotated data under [CC-BY](https://creativecommons.org/licenses/by/4.0/) license.

If you contribute to the OCR Scorer project, you agree to share your contribution following these licenses. 

Contact: Patrice Lopez (<patrice.lopez@science-miner.com>)
