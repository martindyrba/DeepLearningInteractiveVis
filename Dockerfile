# set base image (host OS)
FROM python:3.7.6-slim

# set the working directory in the container
WORKDIR /dockerDir

# copy the dependencies file to the working directory
COPY ./requirements.txt   /dockerDir/

# install dependencies
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY *.hdf5 *.xlsx     /dockerDir/
COPY aal               /dockerDir/aal/
COPY InteractiveVis    /dockerDir/InteractiveVis/
COPY model_checkpoints /dockerDir/model_checkpoints/

# metadata to describe which port the container is listening on at runtime
EXPOSE 5006

# command to run on container start
CMD bokeh serve --port 5006 --allow-websocket-origin='*' --allow-websocket-origin='*:5006' InteractiveVis
