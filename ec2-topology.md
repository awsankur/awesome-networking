To call the API, you’ll need at least aws cli version 2.13.36. You can check the version with:

```
$ aws --version
aws-cli/2.17.56 Python/3.12.6 Linux/5.15.0-1055-aws exe/x86_64.ubuntu.20


curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update


# To update
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update
```
```
(apc-ve) ubuntu@ip-172-31-10-6:~/bionemo$ pcluster describe-cluster-instances -n awsankur-p4de-pcluster --node-type ComputeNode --query 'instances[*].instanceId'
[
  "i-05a4ae1432177a3a5",
  "i-0d7a5c941dc912692"
]

ubuntu@ip-10-0-91-215:~$ aws ec2 describe-instance-topology --instance-ids "i-05a4ae1432177a3a5" "i-0d7a5c941dc912692"
{
    "Instances": [
        {
            "InstanceId": "i-05a4ae1432177a3a5",
            "InstanceType": "p4de.24xlarge",
            "NetworkNodes": [
                "nn-b59b170155f6801f8",
                "nn-a42b0750f49b48c69",
                "nn-38464347ecf47e324"
            ],
            "AvailabilityZone": "us-west-2b",
            "ZoneId": "usw2-az2"
        },
        {
            "InstanceId": "i-0d7a5c941dc912692",
            "InstanceType": "p4de.24xlarge",
            "NetworkNodes": [
                "nn-b59b170155f6801f8",
                "nn-a42b0750f49b48c69",
                "nn-38464347ecf47e324"
            ],
            "AvailabilityZone": "us-west-2b",
            "ZoneId": "usw2-az2"
        }
    ]
}
```
A couple of pints to keep in mind:

1. Cluster PGs have a non oversubscribed Spine and Brick architecture
2. If 2 network nodes are in common, nodes are on the same spine
3. If 3 network nodes are in common, nodes are on the same brick
4. 102.4Tbps bandwith per brick
5. Up to 256 P4 servers per brick (4 servers /rack)
6. One-way latency TOR<>TOR via Spine: 10us

