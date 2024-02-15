FROM python:3.7

RUN apt-get update && \
  rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
  pymdown-extensions==7.1 \
  markdown==3.4.1
