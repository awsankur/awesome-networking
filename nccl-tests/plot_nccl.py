import pandas as pd
import matplotlib.pyplot as plt
import glob

nnodes=[2,4,8,12,16]
OUTPUT_DIRECTORY_PATH='/Users/awsankur/Documents/PyTorch/awesome-networking-reports/p5_all_reduce_2GB/2G'
NCCL_MSG_SIZE='2GB'
NCCL_Collective='all_reduce'

def extract_nccl_test_result(OUTPUT_DIRECTORY_PATH,nnodes):

    NCCL_Test_allreduce_micro_sec = []

    for i in nnodes:

        slurm_file  = glob.glob(f'{OUTPUT_DIRECTORY_PATH}/nodes_{i}/*.out')
        with open(slurm_file[0], 'r') as file:
            content = file.read()
            lines = content.split('\n')

            current_lines = []

            for line in lines:
                split_line = line.split()
                if len(split_line) > 0 and split_line[0] == '#':
                    current_lines.append(line)
                if len(split_line) == 13:
                    if '-1' in line:
                        current_lines.append(line)
                        nccl_result = split_line

        NCCL_Test_allreduce_micro_sec.append(float(nccl_result[5]))

    return(NCCL_Test_allreduce_micro_sec)

NCCL_Test_allreduce_micro_sec = extract_nccl_test_result(OUTPUT_DIRECTORY_PATH,nnodes)

NCCL_Test_allreduce_ms = [i * 0.001 for i in NCCL_Test_allreduce_micro_sec]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))

axes_row_list = [0,0,1,1,2]
axes_col_list = [0,1,0,1,0]


c=0
for one_node in nnodes:

    print(f'Processing NCCL Test for {one_node} nodes')

    kernel_df = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/nodes_{one_node}/kernel_df.csv')
    nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]

    df = nccldf.copy()
    df['duration_ms'] = nccldf['duration']/1000000


    # Ignore anything beyond 99th percentile
    df = df[df['duration_ms'] < df['duration_ms'].quantile(0.99)]

    NCCL_Test_allreduce = NCCL_Test_allreduce_ms[c]

    ax1 = axes[axes_row_list[c],axes_col_list[c]]

    df.hist(column = 'duration_ms',ax = ax1, density=True, bins=100, edgecolor='blue')

    ax1.axvline(NCCL_Test_allreduce, color='k', linestyle='dashed', linewidth=1)

    ax1.text(NCCL_Test_allreduce, 0.5, f'NCCL Test Avg', color='r',
             rotation=90, va='center', ha='center',
             transform=ax1.get_xaxis_transform(),
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax1.set_title(f'nnodes = {one_node}')
    c=c+1

plt.suptitle(f'P5 All Reduce:Sum Message Size = {NCCL_MSG_SIZE} Num nodes = {one_node}', fontsize=16, fontweight='bold')
plt.savefig(f'{OUTPUT_DIRECTORY_PATH}/histogram_all_reduce_sum_{NCCL_MSG_SIZE}_P5_nodes_{one_node}.png')
#plt.show()


# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

axes_row_list = [0,0,1,1,2]
axes_col_list = [0,1,0,1,0]

c = 0
for one_node in nnodes:

    print(f'Processing NCCL Test for {one_node} nodes')

    kernel_df = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/nodes_{one_node}/kernel_df.csv')
    nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]
    df = nccldf.copy()
    df['duration_ms'] = nccldf['duration']/1000000

    NCCL_Test_allreduce = NCCL_Test_allreduce_ms[c]

    # Ignore anything beyond 99th percentile
    df = df[df['duration_ms'] < df['duration_ms'].quantile(0.99)]

    ax1 = axes[axes_row_list[c],axes_col_list[c]]

    df.plot(ax=ax1, y='duration_ms', use_index=True,linewidth=2, linestyle='--', marker='o')
    # Plot p99
    y_position = df['duration_ms'].quantile(0.99)
    ax1.axhline(y = y_position, color='k', linestyle='dashed', linewidth=1)
    ax1.set_title(f'nnodes = {one_node}')
    # Add text on the horizontal line
    ax1.text(0.25, y_position, f'p99 latency = {y_position:.2f} ms', color='r', va='center', ha='center',
        transform=ax1.get_yaxis_transform(), bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))


    c=c+1
# Add a super title to the entire figure
plt.suptitle(f'P5 All Reduce:Sum Message Size ={NCCL_MSG_SIZE} Num nodes = {one_node}', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIRECTORY_PATH}/lineplot_all_reduce_sum_{NCCL_MSG_SIZE}_P5_nodes_{one_node}.png')

## Scaling plots

p50_list = []
p75_list = []
p95_list = []
p99_list = []
pmax_list = []
c=0
for one_node in nnodes:

    print(f'Processing NCCL Test for {one_node} nodes')

    kernel_df = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/nodes_{one_node}/kernel_df.csv')
    nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]
    df = nccldf.copy()
    df['duration_ms'] = nccldf['duration']/1000000

    p50_list.append(df['duration_ms'].quantile(0.50))
    p75_list.append(df['duration_ms'].quantile(0.75))
    p95_list.append(df['duration_ms'].quantile(0.95))
    p99_list.append(df['duration_ms'].quantile(0.99))
    pmax_list.append(max(df['duration_ms']))


p_df = pd.DataFrame(list(zip(nnodes,p50_list,p75_list,p95_list,p99_list,pmax_list)),columns = ['nnodes','p50_ms','p75_ms','p95_ms','p99_ms','pmax_ms'])

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

ax1 = axes[0,0]
# List of colors
colors = ['red', 'green', 'blue']

# Plot each column as a separate line
for i, column in enumerate(p_df.columns[1:]):
    if i < 3:
        ax1.plot(p_df['nnodes'], p_df[column], color=colors[i],
                 label=column, linewidth=2, linestyle='--', marker='o')

# Add legend and labels
ax1.legend()
ax1.set_xlabel('Number of nodes', fontsize=10, fontweight='bold')
ax1.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
ax1.set_title(f'P5 All Reduce:Sum Message Size = {NCCL_MSG_SIZE}', fontsize=16, fontweight='bold')

ax2 = axes[0,1]
ax2.plot(p_df['nnodes'], p_df['p99_ms'], 'black',
                 label='p99_ms', linewidth=2, linestyle='--', marker='o')
# Add legend and labels
ax2.legend()
ax2.set_xlabel('Number of nodes', fontsize=10, fontweight='bold')
ax2.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
ax2.set_title('P99 Latency', fontsize=16, fontweight='bold')

ax3 = axes[1,0]
ax3.plot(p_df['nnodes'], p_df['pmax_ms'], 'black',
                 label='pmax_ms', linewidth=2, linestyle='--', marker='o')
# Add legend and labels
ax3.legend()
ax3.set_xlabel('Number of nodes', fontsize=10, fontweight='bold')
ax3.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
ax3.set_title('Max Latency', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIRECTORY_PATH}/scaling_all_reduce_sum_{NCCL_MSG_SIZE}_P5.png')
#plt.show()
