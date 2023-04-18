FROM python:3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget \
      python3-tk && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN (apt-get autoremove -y; \
     apt-get autoclean -y)

ADD cloudserver.py cloudserver.py

EXPOSE 9090

CMD [ "gunicorn", "--bind", "0.0.0.0:9090", "cloudserver:app" ]
