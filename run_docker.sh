#!/bin/bash
docker build -t xps .
echo "connect to http://localhost:8091"
docker run -d -p 8091:8091 -e PORT=8091 -e WORKERS=1 -e OMP_NUM_THREADS=1 -e TIMEOUT=200 --name=xps --security-opt=seccomp:unconfined --rm xps > logs.txt

