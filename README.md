# Spatial Queue Simulator

Simulating traffic flow using spatial-queue-based model.

* Link model: spatial queue
* Node model: obeying the inflow/outflow/storage capacity of links; protected left-turns
* Routing: fastest path; rerouting at fixed time interval

### Dependency
* Shortest path [`sp`](https://github.com/cb-cities/sp)

### Folder structure
* `queue_class.py`: Python class for Node, Link and Agent
* `dta_meso_[case_study_name].py`: simulation customized for each case study
* `projects/`: inputs and outputs for each case study
    * `[case_study_name]/`: data for each case study stored in separate folders
        * `network_inputs/`: road network graph inputs
            * `nodes.csv`
            * `edges.csv`
        * `demand_inputs/`: o-d pairs inputs
            * `od.csv`
        * `simulation_outputs/`: outputs
            * `log/`
            * `t_stats/`
            * `link_stats/`
            * `node_stats/`

### Run the simulation
`python dta_meso_[case_study_name].py`

`docker run -it -w /game_server/spatial_queue -p 50051:50051 game_server`

`python -m grpc_tools.protoc -Icommunicate --python_out=communicate/server --grpc_python_out=communicate/server communicate/Drive.proto`

https://intl.cloud.tencent.com/document/product/1055/39057#test

cd communicate
protoc -I . --csharp_out=client --grpc_out=client --plugin=protoc-gen-grpc=grpc_csharp_plugin Drive.proto
