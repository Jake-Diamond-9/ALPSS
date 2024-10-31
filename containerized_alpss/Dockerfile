FROM python:3.8-slim
# FROM openmsi/openmsistream:latest

# CMD ["bash"]

USER root

ARG ROOT=/home/openmsi
WORKDIR ${ROOT}

RUN apt-get update && apt-get install -y git libsodium23 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Jake-Diamond-9/ALPSS.git

WORKDIR ${ROOT}/ALPSS

RUN git fetch --all && git checkout processor2

RUN pip install -e .

RUN pip install -r requirements.txt

# CMD ['sleep', 'infinity']
CMD ["/usr/local/bin/python", "ALPSS/alpss_stream_processor.py", "--topic_name", "pdv_data_stieff", "--config", "config.config", "--output_dir", "output_dir/", "--out_files_dir", "out_files_dir/", "--mode", "disk"]
