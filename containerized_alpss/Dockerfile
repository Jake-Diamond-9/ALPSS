FROM python:3.8-slim

USER root

ARG ROOT=/home/openmsi
WORKDIR ${ROOT}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Jake-Diamond-9/ALPSS.git

WORKDIR ${ROOT}/ALPSS

RUN git fetch --all && git checkout processor2

RUN pip install -e .

RUN pip install -r requirements.txt

RUN python --version
RUN /usr/local/bin/python --version

RUN which python
RUN chmod +x /usr/local/bin/python

# CMD ["/bin/sh"]
CMD ["/bin/sh", "-c", "echo 'Container is working' && /usr/local/bin/python --version"]

# ENTRYPOINT ['/usr/local/bin/python' , 'ALPSS/alpss_stream_processor.py',  '--topic_name', 'pdv_data_stieff',  '--config', 'config.config',' --output_dir', 'output_dir/', '--out_files_dir', 'out_files_dir/', '--mode', 'disk']