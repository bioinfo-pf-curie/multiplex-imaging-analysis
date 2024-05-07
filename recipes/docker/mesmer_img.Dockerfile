FROM vanvalenlab/deepcell-applications:0.4.0


RUN python -m pip install --no-cache-dir \
    numpy==1.25.2 \
    tifffile==2023.4.12 \
    ome_types==0.4.2 \
    zarr==2.16.1

COPY assets/mesmer_model/MultiplexSegmentation-9.tar.gz .