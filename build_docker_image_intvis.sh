#!/bin/bash

docker build --file ./Dockerfile --tag interactivevis .
docker save interactivevis:latest | gzip > intvis_latest.tar.gz
