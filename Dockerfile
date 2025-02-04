FROM python:3.12-slim

RUN apt update && apt install -y curl make unzip
RUN pip install --upgrade pip && pip install uv

WORKDIR /opt
RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz && \
    tar zxvf google-cloud-cli-linux-x86_64.tar.gz && \
    ./google-cloud-sdk/install.sh && \
    rm google-cloud-cli-linux-x86_64.tar.gz
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"
COPY secrets/google-application-credentials.json /fdua-competition/secrets/google-application-credentials.json
RUN gcloud auth activate-service-account --key-file="/fdua-competition/secrets/google-application-credentials.json"

WORKDIR /fdua-competition