FROM biocontainers/cellpose:2.1.1_cv2

RUN python -m pip install --no-cache-dir \
  pandas==2.1.4 \
  ome_types==0.4.2 \
  zarr==2.16.1 \
  pydantic==2.5.3 \
  xsdata==23.8