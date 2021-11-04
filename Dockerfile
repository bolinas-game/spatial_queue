# syntax=docker/dockerfile:1

FROM python:3.6-slim-buster
WORKDIR /game_server
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

copy communicate communicate
copy game game
copy model model
copy output output
copy projects projects
copy split_link split_link

CMD ["python3", "communicate/server_rpc.py", "--root_dir", "./"]
