#!/bin/bash

# Select one from all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube
export nccl_test_collective=all_reduce
export nccl_message_size_begin=8
export nccl_message_size_end=3G
export nccl_test_sqsh_path=/fsxl/awsankur/nccl.sqsh
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so
export BASE_PATH=/fsxl/awsankur
nnodes_list=(2 4 8 12 16)

nccl_message_size_list_python=$(python3 generate_nccl_msgs.py $nccl_message_size_begin $nccl_message_size_end)

read -r -a nccl_message_size_list <<< "$nccl_message_size_list_python"
# Submit Slurm jobs to generate recipes if Nsight reports are generated
for nccl_message_size in "${nccl_message_size_list[@]}"
do
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
done
