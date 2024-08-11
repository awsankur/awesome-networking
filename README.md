# What is Elastic Fabric Adapter (EFA)?

When creating infrastructure for distributed ML or HPC systems, networking between different nodes of a cluster becomes important. Typically distributed applications such as distributed training of deep learning models use Message Passing Interface (MPI) or NVIDIA's [Network Collective Communication Library (NCCL)](https://github.com/NVIDIA/nccl) to interface with the Libfabric API. The Libfabric API bypasses the operating system kernel and communicates directly with the EFA device to put packets on the network. This reduces overhead and enables the distributed application to run more efficiently. 

Many CPU and GPU based instances are supported by EFA. Please see [list of supported instance types](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html#efa-instance-types)

# How to create EFA enabled clusters?

## Sagemaker Hyperpod

Created clusters will automatically be EFA enabled and will need no further action

## AWS Parallelcluster

Follow steps here: https://github.com/aws-samples/awsome-distributed-training/tree/main/1.architectures/2.aws-parallelcluster

## Amazon EKS

Once the cluster is created you can install the [AWS EFA Kubernetes Device Plugin](https://github.com/aws/eks-charts/tree/master/stable/aws-efa-k8s-device-plugin) as follows:

```bash
helm repo add eks https://aws.github.io/eks-charts
helm install efa eks/aws-efa-k8s-device-plugin -n kube-system
```
Once this is done, you should see the following pods:
```bash
root@cb9511473ccc:/eks/deployment/efa-device-plugin# kubectl get pods -A
NAMESPACE     NAME                                        READY   STATUS    RESTARTS   AGE
kube-system   aws-efa-k8s-device-plugin-daemonset-78x4q   1/1     Running   0          38m
kube-system   aws-efa-k8s-device-plugin-daemonset-tgfbk   1/1     Running   0          38m
```
You can use the [EKS node viewer](https://github.com/awslabs/eks-node-viewer) tool to view nodes and their status in your cluster. Once it is installed, you can simply type `eks-node-viewer` in the console or `nv` in the `aws-do-eks` container to get the following view:

```bash
3 nodes (650m/199290m) 0.3% cpu ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ $82.272/hour | $60058.195/month
21 pods (0 pending 21 running 21 bound)

ip-192-168-120-214.us-west-2.compute.internal cpu ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   4% (8 pods) c5.2xlarge/$0.3400     On-Demand - Ready
ip-192-168-165-37.us-west-2.compute.internal  cpu ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% (7 pods) p4de.24xlarge/$40.9657 On-Demand - Ready
ip-192-168-164-33.us-west-2.compute.internal  cpu ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   0% (6 pods) p4de.24xlarge/$40.9657 On-Demand - Ready
•
←/→ page • q: quit
```

Here the node viewer shows the IP addresses of my 2 p4de.24xlarge compute nodes. We can take one of the IP addresses to describe the node as:

```bash
kubectl describe node ip-192-168-165-37.us-west-2.compute.internal
```
The above command describes a lot of detail of the node. To make sure EFA is installed correctly make sure you see the following:

```bash
Allocatable:
  cpu:                    95690m
  ephemeral-storage:      868645791124
  hugepages-1Gi:          0
  hugepages-2Mi:          21122Mi
  memory:                 1146004920Ki
  nvidia.com/gpu:         8
  pods:                   250
  vpc.amazonaws.com/efa:  4
```
For p4 nodes you will see ` vpc.amazonaws.com/efa:  4` and for p5.48xlarge nodes you should see ` vpc.amazonaws.com/efa:  32`.

> [!TIP]
> NOTE: If EFA is enabled in the node group, edit the security group that the nodes are attached to and add a rule to allow all outgoing traffic originating from the same security group. This is required for EFA to work.

> [!TIP]
> NOTE: Do this if your pods are labeled.

# How to verify if EFA is installed correctly?

## On a Slurm cluster
First, ssh into a compute node and run `fi_info` as below. On a `p4de.24xlarge` node, you should see the following output. For a `p5.48xlarge` node you should see 32 `rdmapxxsxx-rdm` instances. The numerical string following `rdmap` remians the same across nodes.
```
ubuntu@awsankur-p4de-st-awsankur-p4de-1:~$ fi_info -p efa -t FI_EP_RDM
provider: efa
    fabric: efa
    domain: rdmap16s27-rdm
    version: 119.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap32s27-rdm
    version: 119.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap144s27-rdm
    version: 119.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
provider: efa
    fabric: efa
    domain: rdmap160s27-rdm
    version: 119.0
    type: FI_EP_RDM
    protocol: FI_PROTO_EFA
```
## On a Kubernetes cluster
Follow steps here: https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/efaburn

# Is the EFA security group setup correctly?

To test if the EFA security group is setup correctly, you can run the pingpong test over EFA

## On a slurm cluster
To test this, we will do a pingpong test over EFA. SSH into a compute node and start a server:

```
fi_pingpong -p efa
```
Then in another shell, ssh into anothe compute node and run a client:

```
fi_pingpong -p efa <node-name on which server was created>
```
On a 2 node `p4de.24xlarge` cluster I see:

```
bytes   #sent   #ack     total       time     MB/sec    usec/xfer   Mxfers/sec
64      10      =10      1.2k        0.00s      1.88      34.10       0.03
256     10      =10      5k          0.00s     12.93      19.80       0.05
1k      10      =10      20k         0.00s     52.51      19.50       0.05
4k      10      =10      80k         0.00s    200.78      20.40       0.05
```
## On a Kubernetes cluster
Follow steps here: https://github.com/aws-samples/aws-do-eks/tree/main/Container-Root/eks/deployment/efaburn


# How to test if EFA is working as expected?
NCCL tests are a standard way to test network bandwidth over EFA. Follow steps here to run [NCCL Tests](https://github.com/aws-samples/awsome-distributed-training/tree/main/micro-benchmarks/nccl-tests)

## Expected Bandwidth

|  EC2 Instance |# EFA devices|Total network bandwidth|Max bandwidth per GPU|          RDMA          |
|:-------------:|:-----------:|:---------------------:|:-------------------:|:----------------------:|
| p4de.24xlarge |      4      |        50 GB/s        |                     |    Only Reads          |
|  p5.48xlarge  |     32      |       400 GB/s        |                     |  Only Writes(for now?) |

# How to get real time EFA metrics?

# How to dive deeper with NVIDIA Nsight?


# EFA and RDMA
With RDMA, GPUs across nodes can directly read and write to each other's memory.

# EFA and Hugepages
Huge pages are a memory management technique used in modern computer systems to improve performance by using larger memory blocks than the default page size. Set `FI_EFA_USE_HUGE_PAGE=0` when you see `os.fork() causes OSError: Cannot allocate memory`. Typically happen by multi-process PyTorch data loader. Disabling huge page causes minor performance hit, but it's needed to prevent fork fails due to the operating system running out of huge pages.
## Performance improvement with Hugepages










