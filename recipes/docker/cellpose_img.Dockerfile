FROM biocontainers/cellpose:3.0.1_cv1

RUN python -m pip install --no-cache-dir \
  pandas==2.0.3 \
  ome_types==0.4.2 \
  zarr==2.16.1 \
  pydantic==2.5.3 \
  xsdata==23.8