FROM openmsi/openmsistream:latest

USER root

ARG ROOT=/home/openmsi
WORKDIR ${ROOT}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Jake-Diamond-9/ALPSS.git

WORKDIR ${ROOT}/ALPSS

RUN git fetch --all && git checkout processor2

RUN pip install -e .

RUN pip install -r requirements.txt

