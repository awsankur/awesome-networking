# NCCL All Reduce Test

In this section, we will analyze NCCL All Reduce Performance numbers with NVIDIA Nsight. This test is performed with the following parameters on a 2 node `p4de.24xlarge` cluster:

```
ARG GDRCOPY_VERSION=v2.4.1
ARG EFA_INSTALLER_VERSION=1.34.0
ARG AWS_OFI_NCCL_VERSION=v1.10.0-aws
ARG NCCL_VERSION=v2.22.3-1
ARG NCCL_TESTS_VERSION=v2.13.10

export NCCL_BUFFSIZE=8388608
export NCCL_P2P_NET_CHUNKSIZE=524288

all_reduce_perf -b 25M -e 25M -f 1 -g 1 -c 1 -n 100 -o sum
```
