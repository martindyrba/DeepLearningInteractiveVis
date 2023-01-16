#!/bin/bash

# toggle command line options if required:
# -t   to display output in current terminal, without input
# -it  to run interactively with option to enter input (but then also remove the ampersand at the end)
docker run -t --name interactivevis -p 5006:5006 martindyrba/interactivevis:latest &
