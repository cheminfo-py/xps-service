#!/bin/bash

echo "connect to http://localhost:8091"
docker run -d -p 8091:8091 -e PORT=8091 -e WORKERS=1 -e OMP_NUM_THREADS=1 -e TIMEOUT=500 --rm --name=xps-serviceinstance --security-opt=seccomp:unconfined  xps-service