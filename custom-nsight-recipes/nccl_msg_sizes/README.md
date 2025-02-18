# Script to get NCCL Operations and Message Sizes from any Nsight Report

Get a sumary of NCCL operations and Message sizes as below. Just provide the Nsight report name as an argument.

```
awsankur@ip-192-171-1-189:~/nccl-msg-size-recipe$ python3 get_nccl_msg_size.py --name nemotron_15B_bf16_16g_nvidia_config.nsys-rep
Processing 566135 events: [================================================100%]
nemotron_15B_bf16_16g_nvidia_config.sqlite
       NCCL Operation  Message Size Bytes Reduction Operation
0       ncclAllReduce           201326592                 Sum
1       ncclAllReduce               65536                 Max
2       ncclAllReduce               65536                 Sum
3       ncclAllReduce                   4                 Sum
4       ncclAllReduce             3194880                 Sum
5       ncclAllGather            26214400                None
6       ncclAllGather            25165824                None
7   ncclReduceScatter            25165824                 Sum
8   ncclReduceScatter            52428800                 Avg
9       ncclBroadcast                   8                None
10      ncclBroadcast                   4                None
```

The script `get_nccl_msg_size.py` first exports the Nsight report as a `.sqlite` file and then summarizes the NCCL operations from the `NVTX_EVENTS` table
