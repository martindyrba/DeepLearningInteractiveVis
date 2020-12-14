#!/bin/bash

docker build --file ./Dockerfile --tag intvis .
docker save intvis:latest | gzip > intvis_latest.tar.gz
