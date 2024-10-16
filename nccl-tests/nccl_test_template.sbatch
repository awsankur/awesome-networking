#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH -N ${nnodes} # number of nodes to use, 24 p4d(e) = 192 A100 GPUs
#SBATCH --job-name=${nccl_test_collective}_${nccl_message_size} # name of your job
#SBATCH --ntasks-per-node 8 # Number of GPU per node
#SBATCH --gres=gpu:8 # number of GPU we reserve
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

###########################
###### User Variables #####
###########################


## Plenty of EFA level variables
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export NCCL_DEBUG=INFO


### Increase the send queue depth and can turn NCCL communications into non-blocking.
### https://www.usenix.org/system/files/atc23-choi.pdf
export NCCL_BUFFSIZE=8388608
### Improve performance by increasing buffer size for Send/Recv, Gather, Scatter and Alltoall communications
### https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
export NCCL_P2P_NET_CHUNKSIZE=524288

### Improve performance for AllReduce by selecting specific protocol and algorithm for specific
### message size and number of ranks.
### More information https://github.com/aws/aws-ofi-nccl/wiki/Algorithm-and-Protocol-Tuner-for-AWS.
export NCCL_TUNER_PLUGIN=${NCCL_TUNER_PLUGIN}


declare -a ARGS=(
    --container-image ${nccl_test_sqsh_path}
    --container-mount-home
    --container-mounts /fsxl:/fsxl
    --no-container-remap-root
)
#Get Hostname and Instance IDs
mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

srun "${ARGS[@]}" --mpi=pmix --cpu-bind=none ${nccl_slurm_exec_filename} /opt/nccl-tests/build/${nccl_test_collective}_perf -b ${nccl_message_size} -e ${nccl_message_size} -f 1 -g 1 -c 1 -n 1000 -o sum
