FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# Installing requirements from pip
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir PyWavelets==1.8.0 pydicom==3.0.1 matplotlib==3.10.6 torch==2.8.0

# Installing texlive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        texlive-latex-base \
        texlive-latex-recommended \
        texlive-fonts-recommended \
        texlive-latex-extra \
        cm-super \
        dvipng && \
    rm -rf /var/lib/apt/lists/*

# Installing ray tracing
WORKDIR /ray_tracing
COPY ray_tracing/ .
ARG CUDA_ARCH=75
RUN echo "Using CUDA_ARCH=${CUDA_ARCH}" && \
    make -C /ray_tracing CUDA_ARCH=$CUDA_ARCH
ENV PYTHONPATH=/ray_tracing/tmp

# Creating and moving to the working directory
WORKDIR /code

CMD ["/bin/bash"]