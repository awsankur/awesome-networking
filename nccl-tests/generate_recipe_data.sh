#!/bin/bash

# Select one from all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube
export nccl_test_collective=all_reduce
export nccl_message_size=2G
export nccl_test_sqsh_path=/fsxl/awsankur/nccl.sqsh
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so
export BASE_PATH=/fsxl/awsankur
nnodes_list=(2 4 8 12 16)

# Submit jobs to generate recipes if Nsight reports are generated
for nnodes in "${nnodes_list[@]}"
do
        WORKING_PATH=${BASE_PATH}/${nccl_test_collective}/${nccl_message_size}/nodes_${nnodes}
	pushd ${WORKING_PATH}
        for file in ./*.nsys-rep
	do
		if [ -f "$file" ]; then
    			echo "Report exists: $file"

			/fsxl/nsight-efa/target-linux-x64/nsys recipe cuda_gpu_kern_sum --input $file

  		fi
	done
	popd

done
