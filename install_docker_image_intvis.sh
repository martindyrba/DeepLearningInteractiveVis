#!/bin/bash

# extract image and import into docker
echo importing intvis_latest.tar.gz
gunzip -c intvis_latest.tar.gz | docker load
