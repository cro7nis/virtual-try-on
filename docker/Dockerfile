FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
LABEL maintainer="cro7 <cro7nis@gmail.com>"

ENV API=akash-try-on
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN mkdir -p /opt/$API

WORKDIR /opt/$API
RUN pip install poetry
COPY pyproject.toml pyproject.toml
#COPY  poetry.lock  poetry.lock
RUN  poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction \
    && rm -rf /root/.cache/pypoetry

COPY ckpt/ ckpt/
COPY configs/ configs/
COPY example/ example/
COPY src/ .

ENTRYPOINT python app.py