# Traffic Model

The traffic model plays as a backend server calculation in the bolinas fire game, the main algorithm of which is adapted
from [`Spatial Queue Simulator`](https://github.com/cb-cities/spatial_queue) developed by Bingyu Zhao. If you want to understand
this spatial-queue-based model, you can go to algorithm [`repository`](https://github.com/cb-cities/spatial_queue). If you just want
it to run without knowing the detail, you should build the dependency Shortest path [`sp`](https://github.com/cb-cities/sp).


In README.md, We mainly explain three parts, including:

1. how to realize the communication between unity game and traffic model;
2. how to deploy traffic model into google cloud.

## 1. Realize the Communication

We use grpc to make unity communicate with traffic algorithm written by python. The advantage is that we can just define functions and its 
parameters in one .proto file, then after generating corresponding client and server interface, the same functions can 
be used by different languages. If you are not familiar with grpc, go to this [`website`](https://www.grpc.io/docs/what-is-grpc/).


For python server, we have include grpcio and grpcio-tools in requirements.txt. After installing all these packages, generate python
server interface by command:

`python -m grpc_tools.protoc -Icommunicate --python_out=communicate/server --grpc_python_out=communicate/server communicate/Drive.proto`

The c# unity plugin can be downloaded at this [`link`](https://intl.cloud.tencent.com/document/product/1055/39057#test) first, then generate c# client interface by command:

`cd communicate`

`protoc -I . --csharp_out=client --grpc_out=client --plugin=protoc-gen-grpc=grpc_csharp_plugin Drive.proto`

Grpc is easy to use. But I did not find any tutorials or information in python-grpc to distinguish between 
different clients, (Go-grpc has). However, in our scenario, we must distinguish between different clients, because 
each client must communicate with traffic model every timestamp to get the updated position. If server cannot distinguish 
them, they cannot send correct position info to correct client. I read some answers of relevant StackOverflow questions, and 
write one solution by myself, the details are shown in one
[`post`](https://stackoverflow.com/questions/70044862/how-to-make-each-client-get-their-state-if-there-is-class-instance-in-grpc-pytho) 
in StackOverflow. If you have any other better solutions, feel free to tell me.

We also write the dockerfile for this repository, you can build image by using Dockerfile in this repository or pull from docker hub
`docker pull yanglan/game-server:1.0`.

### Run the simulation

`docker run -it -p 50051:50051 game_server`

## 2. deploy traffic model into google cloud.

