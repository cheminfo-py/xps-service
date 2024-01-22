# The build-stage image:
FROM continuumio/miniconda3

# Install required system packages
RUN apt-get update && apt-get install -y \
    gfortran 
RUN conda install python=3.7 -y
RUN conda install rdkit -c rdkit -y

COPY requirements.txt .

COPY xps ./xps

RUN pip install --no-cache-dir -r requirements.txt

COPY README.md .

CMD gunicorn -w $WORKERS xps.main:app -b 0.0.0.0:$PORT -k uvicorn.workers.UvicornWorker -t $TIMEOUT --keep-alive $TIMEOUT