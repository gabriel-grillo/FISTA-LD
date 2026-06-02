# Para placas mais antigas, você pode tentar usar uma imagem com
# versões mais antigas do CUDA e Ubuntu, por exemplo 11.8.0 e 22.04,
# respectivamente.

ARG CUDAVERSION=12.6.1
ARG UBUNTUVERSION=24.04
ARG BASE_IMAGE=nvidia/cuda:${CUDAVERSION}-devel-ubuntu${UBUNTUVERSION}
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3 && \
    apt-get install -y python3-pip && \
    apt-get install -y python3-venv

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install jupyter && \
    pip install torch --index-url https://download.pytorch.org/whl/$(echo ${CUDAVERSION} | cut --output-delimiter='' -d. -f1-2 ) && \
    pip install matplotlib

COPY . /app
