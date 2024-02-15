FROM python:3.11

RUN apt-get update && \
  apt-get install -y python3-opencv && \
  rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
  pandas==2.1.1 \
  numpy==1.25.2 \
  tifffile==2023.4.12 \
  scikit-image==0.19.3 \
  ome_types==0.4.2 \
  zarr==2.16.1 \
  opencv-python-headless==4.8.1.78 \