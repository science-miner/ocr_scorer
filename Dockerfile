## Docker image for the packaged OCR scorer service

## See https://github.com/science-miner/ocr_scorer

# this should automatically recognize nvidia GPU drivers on host machine when available
FROM tensorflow/tensorflow:2.7.0-gpu
CMD nvidia-smi

# setting locale is likely useless but to be sure
ENV LANG C.UTF-8

USER root

RUN apt-get update && \
    apt-get -y --no-install-recommends install libfontconfig1

RUN python3 -m pip install pip --upgrade

# copy project
COPY ocr_scorer /opt/ocr_scorer
COPY data /opt/data
COPY requirements.txt /opt/
COPY config.yml /opt/
copy setup.py /opt/

RUN rm -rf /var/lib/apt/lists/*

# fix logging
ENV PYTHONWARNINGS="ignore"

WORKDIR /opt

RUN pip install -r requirements.txt
RUN pip install -e .

CMD PYTHONPATH="..:${PYTHONPATH}" python3 ocr_scorer/service.py
