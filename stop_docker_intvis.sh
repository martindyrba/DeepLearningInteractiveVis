#!/bin/bash

# get id of interactivevis container
containerid=$( docker ps | grep interactivevis | cut -d" " -f1 )

if [ -z "$containerid" ]; then
  echo "no running interactivevis docker instance found" > /dev/stderr
else
  echo -n "stopping docker interactivevis instance: "
  docker stop "$containerid"
fi
