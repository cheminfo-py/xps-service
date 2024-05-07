# The build-stage image:
FROM continuumio/miniconda3

RUN conda install python=3.9 -y
#RUN conda install rdkit -c rdkit -y
RUN conda install xtb-python -c conda-forge -y


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#RUN pip install -r requirements.txt

COPY README.md .
COPY xpsservice ./xpsservice
#COPY xpscache ./xpscache
COPY SOAP_configs ./SOAP_configs
COPY ML_models ./ML_models

CMD gunicorn -w $WORKERS xpsservice.xpsservice:app -b 0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker -t $TIMEOUT --keep-alive $TIMEOUT
