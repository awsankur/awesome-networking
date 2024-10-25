# Custom Recipes for NVIDIA Nsight reports

NVIDIA Nsight provies a number of recipes that can be used to analyze and extract insights from NVIDIA Nsight reports. However, there is often a need to generate custom visualization and analysis of data. In this section, we will go over the process of how to create a custom Nsight recipe.

## Pre-requisites

We require that you have:

A slurm cluster available of p5.48xlarge nodes
You have NCCL test Docker image built and Enroot .sqsh file created. You can follow the steps here: https://github.com/aws-samples/awsome-distributed-training/tree/main/micro-benchmarks/nccl-tests
You have Nsight pre-installed at `/fsxl/nsight-efa`. Follow steps here to install latest releases of Nsight: https://github.com/aws-samples/awsome-distributed-training/tree/main/4.validation_and_observability/5.nsight

## Existing Nsight recipes

To get a list of Nsight recipes you can try:

```
awsankur@ip-10-0-1-210:~$ /fsxl/nsight-efa/target-linux-x64/nsys recipe -h

usage: nsys recipe [<args>] <recipe name> [<recipe args>]

	-h, --help

	    Print the command's help menu.

	-q, --quiet

           Only display errors.

The following built-in recipes are available:

  cuda_api_sum -- CUDA API Summary
  cuda_api_sync -- CUDA Synchronization APIs
  cuda_gpu_kern_hist -- CUDA GPU Kernel Duration Histogram
  cuda_gpu_kern_pace -- CUDA GPU Kernel Pacing
  cuda_gpu_kern_sum -- CUDA GPU Kernel Summary
  cuda_gpu_mem_size_sum -- CUDA GPU MemOps Summary (by Size)
  cuda_gpu_mem_time_sum -- CUDA GPU MemOps Summary (by Time)
  cuda_gpu_time_util_map -- CUDA GPU Time Utilization Heatmap
  cuda_memcpy_async -- CUDA Async Memcpy with Pageable Memory
  cuda_memcpy_sync -- CUDA Synchronous Memcpy
  cuda_memset_sync -- CUDA Synchronous Memset
  diff -- Statistics Diff
  dx12_mem_ops -- DX12 Memory Operations
  gpu_gaps -- GPU Gaps
  gpu_metric_util_map -- GPU Metric Utilization Heatmap
  gpu_time_util -- GPU Time Utilization
  mpi_gpu_time_util_map -- MPI and GPU Time Utilization Heatmap
  mpi_sum -- MPI Summary
  my_custom_recipe -- My Custom Recipe
  nccl_gpu_overlap_trace -- NCCL GPU Overlap Trace
  nccl_gpu_proj_sum -- NCCL GPU Projection Summary
  nccl_gpu_time_util_map -- NCCL GPU Time Utilization Heatmap
  nccl_sum -- NCCL Summary
  network_sum -- Network Traffic Summary
  network_traffic_map -- Network Devices Traffic Heatmap
  nvlink_sum -- NVLink Network Bandwidth Summary
  nvtx_gpu_proj_pace -- NVTX GPU Projection Pacing
  nvtx_gpu_proj_sum -- NVTX GPU Projection Summary
  nvtx_gpu_proj_trace -- NVTX GPU Projection Trace
  nvtx_pace -- NVTX Pacing
  nvtx_sum -- NVTX Range Summary
  osrt_sum -- OS Runtime Summary
  ucx_gpu_time_util_map -- UCX and GPU Time Utilization Heatmap

To get help on a specific recipe, run 'nsys recipe <recipe name> --help'.

Note that running 'nsys recipe <recipe name>' requires extra Python packages:
  - List of required Python packages: '/fsxl/nsight-efa/target-linux-x64/python/packages/nsys_recipe/requirements/common.txt'
  - Helper script to automate installation of dependencies: '/fsxl/nsight-efa/target-linux-x64/python/packages/nsys_recipe/install.py'

```

## Creating a custom recipe

Lets say we have to create a custom recipe `my_custom_recipe`, first we need to create a folder as below:

```
export CUSTOM_RECIPE_NAME='my_custom_recipe'

mkdir -p /fsxl/nsight-efa/target-linux-x64/python/packages/nsys_recipe/recipes/$CUSTOM_RECIPE_NAME
```

Save the following as `metadata.json`

```
{
    "module_name": "my_custom_recipe - Sould be the same name as my_custom_recipe.py",
    "display_name": "Text that shows up when you try nsys recipe -h",
    "description": "Description for My Custom Recipe"
}
```
Now if you try `/fsxl/nsight-efa/target-linux-x64/nsys recipe -h` you should see the custom recipe listed. Also you can see the expected inputs as:

```
awsankur@ip-10-0-1-210:~/custom-recipe$ /fsxl/nsight-efa/target-linux-x64/nsys recipe my_custom_recipe -h
usage: my_custom_recipe [-h] [--output OUTPUT] [--force-overwrite] --input
                        INPUT [INPUT ...] [--csv]
                        [--filter-time [start_time]/[end_time] | --filter-nvtx
                        range[@domain][/index]]
                        [--mode {none,concurrent,dask-futures}]

Custom Recipe

optional arguments:
  -h, --help            show this help message and exit

Context:
  --mode {none,concurrent,dask-futures}
                        Mode to run tasks

Recipe:
  --output OUTPUT       Output directory name.
                        Any %q{ENV_VAR} pattern in the filename will be
                        substituted with the value of the environment
                        variable.
                        Any %h pattern in the filename will be substituted
                        with the hostname of the system.
                        Any %p pattern in the filename will be substituted
                        with the PID.
                        Any %n pattern in the filename will be substituted
                        with the minimal positive integer that is not already
                        occupied.
                        Any %% pattern in the filename will be substituted
                        with %
  --force-overwrite     Overwrite existing directory
  --input INPUT [INPUT ...]
                        One or more paths to nsys-rep files or directories.
                        Directories can optionally be followed by ':n' to
                        limit the number of files
  --csv                 Additionally output data as CSV
  --filter-time [start_time]/[end_time]
                        Time range used for filtering in nanoseconds.
  --filter-nvtx range[@domain][/index]
                        NVTX range information used for filtering.
                        Specify the domain only when the range is not in the
                        default domain, or use '*' to include all domains. Any
                        '@' or '/' in the names should be escaped with a
                        backslash.
                        The index is zero-based and is used to select the nth
                        range. If no index is specified, all ranges will be
                        used
```



























