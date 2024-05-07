FROM vanvalenlab/deepcell-applications:0.4.0


RUN python -m pip install --no-cache-dir \
    ome_types==0.4.2 \
    zarr==2.16.1

COPY assets/mesmer_model/MultiplexSegmentation-9.tar.gz .

ENTRYPOINT [ "/bin/bash" ]