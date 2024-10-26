# Custom recipe to analyze AWS OFI NCCL data

## Install AWS-OFI-NCCL Plugin with NVTX traces

To spit out NVTX traces, install AWS-OFI-NCCL plugin in the Dockerfile as below:

```
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
#Switch from sh to bash to allow parameter expansion
SHELL ["/bin/bash", "-c"]
RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
        --with-nvtx=/usr/local/cuda \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

SHELL ["/bin/sh", "-c"]

```

## Run the recipe

```
/fsxl/nsight-efa/target-linux-x64/nsys recipe aws_ofi_nccl --input profile_all_reduce_ofi_trace_408_node_0_rank_0_on_p5-dy-gpu-1.nsys-rep --csv
```

This recipe first collects NVTX data from the `NVTX_EVENTS` table which has the following schema:

```
sqlite> .schema NVTX_EVENTS
CREATE TABLE NVTX_EVENTS (
    start                       INTEGER   NOT NULL,                    -- Event start timestamp (ns).
    end                         INTEGER,                               -- Event end timestamp (ns).
    eventType                   INTEGER   NOT NULL,                    -- REFERENCES ENUM_NSYS_EVENT_TYPE(id)
    rangeId                     INTEGER,                               -- Correlation ID returned from a nvtxRangeStart call.
    category                    INTEGER,                               -- User-controlled ID that can be used to group events.
    color                       INTEGER,                               -- Encoded ARGB color value.
    text                        TEXT,                                  -- Explicit name/text (non-registered string)
    globalTid                   INTEGER,                               -- Serialized GlobalId.
    endGlobalTid                INTEGER,                               -- Serialized GlobalId.
    textId                      INTEGER,                               -- REFERENCES StringIds(id) -- Registered NVTX domain/string
    domainId                    INTEGER,                               -- User-controlled ID that can be used to group events.
    uint64Value                 INTEGER,                               -- One of possible payload value union members.
    int64Value                  INTEGER,                               -- One of possible payload value union members.
    doubleValue                 REAL,                                  -- One of possible payload value union members.
    uint32Value                 INTEGER,                               -- One of possible payload value union members.
    int32Value                  INTEGER,                               -- One of possible payload value union members.
    floatValue                  REAL,                                  -- One of possible payload value union members.
    jsonTextId                  INTEGER,                               -- One of possible payload value union members.
    jsonText                    TEXT,                                  -- One of possible payload value union members.
    binaryData                  TEXT                                   -- Binary payload. See docs for format.
);
```
The recipe pulls the following columns from the `NVTX_EVENTS` table and merges with the `StringIds` table that has the mapping of text to ids.

```
"text",
"start",
"end",
"textId",
"globalTid",
"endGlobalTid",
"domainId",
"eventType",
```

