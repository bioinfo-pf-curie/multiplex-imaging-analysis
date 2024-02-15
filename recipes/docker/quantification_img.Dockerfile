FROM python:3.11

RUN apt-get update && \
  rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
  pandas==2.1.1 \
  numpy==1.25.2 \
  scikit-image==0.20.0 \
  h5py==3.9.0 \
  pathlib==1.0.1 \
  imagecodecs==2023.1.23