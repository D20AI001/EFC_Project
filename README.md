# EFC_Project
FogFL: Fog Assisted Federated Learning


Data set link -

We have used publicly available check xray data for this inference -

https://drive.google.com/drive/folders/1nasqBJHVbypCmdn8ZzdDu1LJXA7m131L?usp=share_link


Please use the below commands to start the fog nodes and cloud server

CD into the corresponding directory before executing the below commands -

CD fog_node_1

fog_node_1

docker build -t fog_node_1 .
docker run -it --rm -p 9191:9191 --network=host -v <local dir>/SharedStorage/:/app/SharedStorage fog_node_1

CD fog_node_2

fog_node_2

docker build -t fog_node_2 .
docker run -it --rm -p 9292:9292 --network=host -v <local dir>/SharedStorage/:/app/SharedStorage fog_node_2

CD cloud_server

cloud_server

docker build -t cloud_server .
docker run -it --rm -p 9090:9090 --network=host -v <local dir>/SharedStorage/:/app/SharedStorage cloud_server
