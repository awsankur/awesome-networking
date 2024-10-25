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

If you need a Jupyter notebook to be produced as part of the recipe, a template needs to be present. Popular templates from existing recipes include - `stats.ipynb, heatmap.ipynb, trace.ipynb, pace.ipynb, topN.ipynb, analysis.ipynb, histogram.ipynb`

When a recipe is called, the following things happen in order:

1. `run_recipe` function in `__main__.py` calls the run function in `my_custom_recipe.py`
2. `run` function `my_custom_recipe.py` sends the inputs such as nsys profile name, filter-time etc to the `_mapper_func` in `MyCustomRecipe` class.
3. `service` is created by `data_service.py` which has the tooling to read the Nsight report after converting it to a parquet format.

You can use the following to pull the right data for your recipe:

1. Cuda GPU Kernel Data: `service.queue_table("CUPTI_ACTIVITY_KIND_KERNEL", ["shortName", "start", "end", "deviceId"])`
2. NVTX Data: `service.queue_custom_table(CompositeTable.NVTX)`
3. NCCL Data: `service.queue_custom_table(CompositeTable.NCCL)`
4. OSRT Data: `service.queue_table("OSRT_API", ["nameId", "start", "end"])`
5. MPI Data: `service.queue_table("OSRT_API", ["nameId", "start", "end"])`

To pull data in pandas data frames: `df_dict = service.read_queued_tables()`. `df_dict` will be a dictionary of all queued tables as data frames.

Next filter and adjust time as: `service.filter_and_adjust_time(kernel_df)`

Replace string ids with values as: `kernel_df = data_utils.replace_id_with_value(
            kernel_df, df_dict["StringIds"], "shortName", "name")`

An example recipe is provided in `my_custom_recipe.py`

### Custom Composite Table

You can create a custom `CompositeTable` like NVTX and NCCL tables as below:

1. Find the `table_config.py` file in `/fsxl/nsight-efa/target-linux-x64/python/packages/nsys_recipe/lib`
2. Add the name of the custom composite table in `CompositeTable` and assign an integer like 

```
class CompositeTable(Enum):
    CUDA_GPU = 0
    CUDA_GPU_GRAPH = 1
    CUDA_COMBINED = 2
    CUDA_COMBINED_KERNEL = 3
    NVTX = 4
    NCCL = 5
    NIC = 6
    IB_SWITCH = 7
    MPI = 8
    UCX = 9
    AWS_OFI_NCCL = 10
```
3. Find `get_table_column_dict` and add `CompositeTable.AWS_OFI_NCCL: get_aws_ofi_dict` in `table_dict_map`. Or if you want to reuse an existing table dictionary such as `get_nvtx_dict`.
4. Find `get_refine_func` and add `CompositeTable.AWS_OFI_NCCL: process_to_aws_ofi_table` in `table_func_map`.
5. Next you would have to add `get_aws_ofi_dict` and `process_to_aws_ofi_table` functions




























