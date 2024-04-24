# Use a specific Python 3.9 base image
FROM python:3.9-slim

# Set environment variables
ENV LANG=C.UTF-8

# Install required system packages in one step to minimize layers
RUN apt-get update && apt-get install -y \
    libxext6 \
    libxrender-dev \
    wget \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Set PATH to use conda
ENV PATH="/opt/conda/bin:$PATH"

# Create a new conda environment with Python 3.9
RUN /opt/conda/bin/conda create --name myenv python=3.9 -y

# Activate the environment and install required packages in one command
RUN /opt/conda/bin/conda install -n myenv rdkit -c rdkit -y \
    && /opt/conda/bin/conda clean --all

# Set the working directory
WORKDIR /app

# Copy application files to the working directory
COPY requirements.txt /app/
COPY xps /app/xps/
COPY README.md /app/

# Install Python dependencies in the conda environment
RUN /opt/conda/envs/myenv/bin/pip install --no-cache-dir -r requirements.txt

# Set the entrypoint and command to run the application
ENTRYPOINT ["/opt/conda/envs/myenv/bin/gunicorn"]
CMD ["-w", "4", "xps.main:app", "-b", "0.0.0.0", "-k", "uvicorn.workers.UvicornWorker", "-t", "30", "--keep-alive", "30"]
