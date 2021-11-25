# syntax=docker/dockerfile:1

FROM python:3.6-slim-buster
WORKDIR /game_server

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y wget

# RUN apt install -y libprotobuf-dev protobuf-compiler
# RUN apt-get update && apt-get -y install cmake

RUN git clone https://github.com/bolinas-game/spatial_queue.git \
    && cd spatial_queue && git checkout master && cd .. \
    && git clone https://github.com/cb-cities/sp.git temp \
    && cd temp && git checkout dataframe \
    && cd .. \
    && cp -a temp/. spatial_queue/sp/. \
    && rm -rf temp \
    && mkdir spatial_queue/sp/build \
#     && cd spatial_queue/sp/build \
#     && cmake -DCMAKE_BUILD_TYPE=Release ..\
#     && make clean \
#     && make -j4
    && wget --no-check-certificate --content-disposition https://github.com/UCB-CE170a/Fall2020/raw/master/traffic_data/liblsp.so -P ./spatial_queue/sp/build


RUN cd spatial_queue \
    && pip3 install -r requirements.txt

WORKDIR /game_server/spatial_queue

# EXPOSE 50051/tcp
CMD ["python3", "communicate/server_rpc2.py", "--root_dir", "./"]
