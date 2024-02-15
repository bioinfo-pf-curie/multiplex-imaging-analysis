FROM python:3.11

RUN apt-get update && \
  apt-get install -y python3-opencv && \
  rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
  scikit-image==0.19.3 \
  tifffile==2023.4.12 \
  zarr \
  ome_types>=0.4.2 \
  palom==2023.9.2