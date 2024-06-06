FROM python:3.11

RUN apt-get update && \
  apt-get install -y python3-opencv && \
  rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
  h5py==3.9.0 \
  pathlib==1.0.1 \
  imagecodecs==2023.1.23 \
  shapely==2.0.1 \
  dask-core==2023.6.0 \
  scikit-image==0.23.1 \
  psutil==5.9.6 \
  fastremap==1.13.4 \
  pandas==2.1.1 \
  numpy==1.25.2 \
  tifffile==2023.4.12 \
  ome_types==0.4.2 \
  zarr==2.16.1 \
  opencv-python-headless==4.8.1.78 \
  pydantic==2.5.3 \
  xsdata==23.8