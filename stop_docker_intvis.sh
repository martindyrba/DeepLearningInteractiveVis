#!/bin/bash

# get id of intvis container
containerid=$( docker ps | grep intvis | cut -d" " -f1 )

if [ -z "$containerid" ]; then
  echo "no running intvis docker instance found" > /dev/stderr
else
  echo -n "stopping docker intvis instance: "
  docker stop "$containerid"
fi
