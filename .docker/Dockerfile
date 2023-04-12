# syntax = docker/dockerfile:1.2
ARG KIND=3.10
FROM mkrausio/ml_research:torch-${KIND}

ARG COMPOSE_PROJ_NAME="code"

ENV PROJECT_NAME=${COMPOSE_PROJ_NAME}
RUN echo ${COMPOSE_PROJ_NAME}/$PROJECT_NAME

RUN apt update && apt install -y git-lfs ffmpeg&& git lfs install

RUN python -m pip install gdown

RUN mkdir -p workspaces/$PROJECT_NAME


WORKDIR /workspaces/$PROJECT_NAME
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
  python -m pip install --upgrade pip && \
  pip install -r requirements.txt

# RUN pip install git+https://github.com/mauricekraus/torchtime.git && pip install sktime 
#sktime[dl]


CMD zsh
