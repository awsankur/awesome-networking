# NCCL Test and Nsight Sweeps

In this section, we present a framework to run sweeps over different configurations of NCCL tests and analyze them with data from Nsight reports and generate plots in a few clicks.

The motivation behind this work is to understand how p50, p95, p99 latencies vary with different combinations of NCCL Collectives, number of nodes and NCCL message sizes. 

## Pre-requisites

We require that you have:

1. A slurm cluster available of `p5.48xlarge` nodes
2. You have NCCL test Docker image built and Enroot `.sqsh` file created. You can follow the steps here: https://github.com/aws-samples/awsome-distributed-training/tree/main/micro-benchmarks/nccl-tests
3. You have Nsight pre-installed at `/fsxl/nsight-efa`. Follow steps here to install latest releases of Nsight: https://github.com/aws-samples/awsome-distributed-training/tree/main/4.validation_and_observability/5.nsight

## Automated data extraction from Nsight profile

We will use `nsys recipe` to automatically extract NCCL kernel durations from Nsight reports. To do so, follow these steps:

```
cd /fsxl/nsight-efa/target-linux-x64/python/packages/nsys_recipe/recipes/cuda_gpu_kern_sum/
Find the `_mapper_func` function in `CudaGpuKernSum` class
Add `kernel_df.to_csv('./kernel_df.csv')` line at the bottom of the function
```
## Explanation of files

1. nccl_test_template.sbatch -- Slurm submission sbatch script template for NCCL Test
2. nccl-slurm-exec-template -- Template script setup to profile the job with nsys
3. generate_nccl_msgs.py -- Python script to generate an array of NCCL message sizes
4. nccl_sweep.sh -- This script would submit multiple NCCL Test jobs and generate Nsight reports for each.
5. generate_recipe_data.sh -- Script to generate recipes and generate a csv with start, end timestamps and durations of NCCL Kernels.
6. plot_nccl.py -- Script to extract NCCL test results from slurm output files and generate linecharts and histograms for the NCCL Kernel Latencies

## User setup

You can export the following variables to define the input space to sweep:

For the collective, you can choose one from the following: `all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube`

You can specify NCCL Message sizes in Bytes, KB, MB or GB just like you would do in a NCCL Test. To run test for only 1 size, keep the begin and end sizes to be the same. 

```
export nccl_test_collective=all_reduce
export nccl_message_size_begin=8
export nccl_message_size_end=3G
export nccl_test_sqsh_path=/fsxl/awsankur/nccl.sqsh
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so
export BASE_PATH=/fsxl/awsankur
nnodes_list=(2 4 8 12 16)
```

## Steps

1. `./nccl_sweep.sh` would generate a list of NCCL Message sizes in Bytes and would create a folder for each message size in `${BASE_PATH}/${nccl_test_collective}/data`. Within each folder, there will be subfolders  for different number of nodes in `${nnodes_list}`. Within the `nodes_xx` folder you will find the sbatch script that was submitted, the slurm output file and also the Nsight profile

2. `generate_recipe_data.sh` would run the `cuda_gpu_kern_sum` recipe and save `kernel_df.csv` in `${BASE_PATH}/${nccl_test_collective}/data/nccl_message_size/nodes_xx/` folders

3. `python3 plot_nccl.py` will extract NCCL test results and save in `${BASE_PATH}/${nccl_test_collective}/nccl_test_df.csv`. It will generate a latency_df.csv in the same path with the columns `'nnodes','p50_ms','p75_ms','p95_ms','p99_ms','pmax_ms','nccl_msg_size'`. It will also create a plots folder with plots for scaling vs nodes and scaling vs nccl message sizes.
