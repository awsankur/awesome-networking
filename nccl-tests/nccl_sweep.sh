#!/bin/bash

# Select one from all_reduce all_gather broadcast reduce_scatter reduce alltoall scatter gather sendrecv hypercube
export nccl_test_collective=all_reduce
export nccl_message_size=2G
export nccl_test_sqsh_path=/fsxl/awsankur/nccl.sqsh
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so
export BASE_PATH=/fsxl/awsankur
nnodes_list=(2 4 8 12 16)


# Submit Slurm jobs
for nnodes in "${nnodes_list[@]}"
do
	WORKING_PATH=${BASE_PATH}/${nccl_test_collective}/${nccl_message_size}/nodes_${nnodes}
	mkdir -p ${WORKING_PATH}
	export WORKING_PATH=${WORKING_PATH}
	export nnodes=${nnodes}
	echo "Submitting ${nccl_test_collective} job for $nnodes nodes"

	export nccl_slurm_exec_filename="${WORKING_PATH}/nccl-slurm-exec-${nccl_test_collective}-${nccl_message_size}-${nnodes}"
	envsubst '$WORKING_PATH $nccl_test_collective $nccl_message_size' < ./nccl-slurm-exec-template > ${nccl_slurm_exec_filename}
	chmod 777 ${nccl_slurm_exec_filename}
	sbatch_file_name="${WORKING_PATH}/${nccl_test_collective}_${nccl_message_size}"
	envsubst  < ./nccl_test_template.sbatch > ${sbatch_file_name}.sbatch

	pushd ${WORKING_PATH}
	#sbatch ${sbatch_file_name}.sbatch
	popd

done

