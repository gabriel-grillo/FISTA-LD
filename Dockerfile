FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

WORKDIR /code

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir PyWavelets==1.8.0 pydicom==3.0.1 matplotlib==3.10.6 torch==2.8.0

CMD ["/bin/bash"]