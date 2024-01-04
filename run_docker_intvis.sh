#!/bin/bash

# toggle command line options if required:
# -t   to display output in current terminal, without input
# -it  to run interactively with option to enter input (but then also remove the ampersand at the end)
# -d   to run detached, without showing output or log messages
# use --restart unless-stopped  to run as background service and restart if the host machine (re)starts
#docker run -t --name interactivevis -p 5006:5006 martindyrba/interactivevis:latest &
docker run -d --restart unless-stopped --name interactivevis -p 5006:5006 martindyrba/interactivevis:latest
