## Docker image for the packaged OCR scorer service

## See https://github.com/science-miner/ocr_scorer

FROM python:3.7-slim-buster

# setting locale is likely useless but to be sure
ENV LANG C.UTF-8

USER root

RUN python3 -m pip install pip --upgrade

# copy project
COPY ocr_scorer /opt/ocr_scorer
COPY data /opt/data
COPY requirements.txt /opt/

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN pip install -r requirements.txt

WORKDIR /opt/ocr_scorer

CMD PYTHONPATH="..:${PYTHONPATH}" python3 service.py
