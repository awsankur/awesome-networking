import pandas as pd
import matplotlib.pyplot as plt

nnodes=[2,4,8,12,16]

NCCL_Test_allreduce_micro_sec=[394.5,654.2,802.3,955.1,963.8]
NCCL_Test_allreduce_ms = [i * 0.001 for i in NCCL_Test_allreduce_micro_sec]

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))

axes_row_list = [0,0,1,1,2]
axes_col_list = [0,1,0,1,0]


c=0
for one_node in nnodes:

    print(f'Processing NCCL Test for {one_node} nodes')

    kernel_df = pd.read_csv(f'./kernel_dfs/kernel_df_nodes_{one_node}.csv')
    nccldf = kernel_df[kernel_df['name']=='ncclDevKernel_AllReduce_Sum_f32_TREE_LL']
    df = nccldf.copy()
    df['duration_ms'] = nccldf['duration']/1000000
    
    #df = pd.read_excel('p5_all_red_25mb.xlsx',sheet_name=f'nodes{one_node}')

    #df[['duration_ms','unit']] = df['Duration'].str.split(' ',expand = True)
    #df['duration_ms'] = df['duration_ms'].astype(float)
    #df.loc[df['unit']=='μs','duration_ms']=0.001*df.loc[df['unit']=='μs','duration_ms']

    # Ignore anything beyond 99th percentile
    df = df[df['duration_ms'] < df['duration_ms'].quantile(0.99)]

    NCCL_Test_allreduce = NCCL_Test_allreduce_ms[c]

    print(NCCL_Test_allreduce)

    ax1 = axes[axes_row_list[c],axes_col_list[c]]

    df.hist(column = 'duration_ms',ax = ax1, density=True, bins=100, edgecolor='blue')
    
    ax1.axvline(NCCL_Test_allreduce, color='k', linestyle='dashed', linewidth=1) 

    ax1.text(NCCL_Test_allreduce, 0.5, f'NCCL Test Avg', color='r', 
             rotation=90, va='center', ha='center',
             transform=ax1.get_xaxis_transform(), 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax1.set_title(f'nnodes = {one_node}')
    c=c+1

plt.suptitle(f'P5 All Reduce:Sum Message Size = 25 MB Num nodes = {one_node}', fontsize=16, fontweight='bold')
plt.savefig(f'histogram_all_reduce_sum_25MB_P5_nodes_{one_node}.png')
#plt.show()


# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

axes_row_list = [0,0,1,1,2]
axes_col_list = [0,1,0,1,0]

c = 0
for one_node in nnodes:

    print(f'Processing NCCL Test for {one_node} nodes')

    kernel_df = pd.read_csv(f'./kernel_dfs/kernel_df_nodes_{one_node}.csv')
    nccldf = kernel_df[kernel_df['name']=='ncclDevKernel_AllReduce_Sum_f32_TREE_LL']
    df = nccldf.copy()
    df['duration_ms'] = nccldf['duration']/1000000
    
    #df = pd.read_excel('p5_all_red_25mb.xlsx',sheet_name=f'nodes{one_node}')

    #df[['duration_ms','unit']] = df['Duration'].str.split(' ',expand = True)
    #df['duration_ms'] = df['duration_ms'].astype(float)
    #df.loc[df['unit']=='μs','duration_ms']=0.001*df.loc[df['unit']=='μs','duration_ms']

    NCCL_Test_allreduce = NCCL_Test_allreduce_ms[c]

    # Ignore anything beyond 25ms
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
plt.suptitle(f'P5 All Reduce:Sum Message Size = 25 MB Num nodes = {one_node}', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(f'lineplot_all_reduce_sum_25MB_P5_nodes_{one_node}.png')

## Scaling plots

p50_list = []
p75_list = []
p95_list = []
p99_list = []
pmax_list = []
c=0
for one_node in nnodes:

    print(f'Processing NCCL Test for {one_node} nodes')

    kernel_df = pd.read_csv(f'./kernel_dfs/kernel_df_nodes_{one_node}.csv')
    nccldf = kernel_df[kernel_df['name']=='ncclDevKernel_AllReduce_Sum_f32_TREE_LL']
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
ax1.set_title('P5 All Reduce:Sum Message Size = 25 MB', fontsize=16, fontweight='bold')

ax2 = axes[0,1]
ax2.plot(p_df['nnodes'], p_df['p99_ms'], 'black', 
                 label=column, linewidth=2, linestyle='--', marker='o')
# Add legend and labels
ax2.legend()
ax2.set_xlabel('Number of nodes', fontsize=10, fontweight='bold')
ax2.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
ax2.set_title('P99 Latency', fontsize=16, fontweight='bold')

ax2 = axes[1,0]
ax2.plot(p_df['nnodes'], p_df['pmax_ms'], 'black', 
                 label=column, linewidth=2, linestyle='--', marker='o')
# Add legend and labels
ax2.legend()
ax2.set_xlabel('Number of nodes', fontsize=10, fontweight='bold')
ax2.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
ax2.set_title('Max Latency', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(f'scaling_all_reduce_sum_25MB_P5.png')
#plt.show()
    
