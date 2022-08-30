# Use an official Python runtime as a base image
FROM python:3.7-slim

## The maintainer name and email
LABEL maintainer="CHIME/FRB Collaboration"

ADD . /ch_util

RUN apt-get update && \
    #--------------------------------------------
    #Install any needed packages
    #--------------------------------------------
    apt-get install -y git build-essential libopenmpi-dev && \
    pip install -r /ch_util/requirements.txt && \
    pip install /ch_util && \
    #-----------------------
    # Minimize container size
    #-----------------------
    pip uninstall -y cython && \
    apt-get remove -y git libopenmpi-dev gcc && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /tmp/build

# Run comet when the container launches
CMD ["bash"]
