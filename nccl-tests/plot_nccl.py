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
    
    df = pd.read_excel('p5_all_red_25mb.xlsx',sheet_name=f'nodes{one_node}')

    df[['duration_ms','unit']] = df['Duration'].str.split(' ',expand = True)
    df['duration_ms'] = df['duration_ms'].astype(float)
    df.loc[df['unit']=='μs','duration_ms']=0.001*df.loc[df['unit']=='μs','duration_ms']

    # Ignore anything beyond 25ms
    df = df[df['duration_ms'] < df['duration_ms'].quantile(0.99)]

    NCCL_Test_allreduce = NCCL_Test_allreduce_ms[c]

    df.hist(column = 'duration_ms',ax = axes[axes_row_list[c],axes_col_list[c]], 
             density=True, bins=100, edgecolor='blue')
    
    axes[axes_row_list[c],axes_col_list[c]].axvline(NCCL_Test_allreduce, color='k', linestyle='dashed', linewidth=1) 
    axes[axes_row_list[c],axes_col_list[c]].set_title(f'nnodes = {one_node}')
    c=c+1

plt.suptitle(f'P5 All Reduce:Sum Message Size = 25 MB Num nodes = {one_node}', fontsize=16, fontweight='bold')
plt.savefig(f'histogram_all_reduce_sum_25MB_P5_nodes_{one_node}.png')


# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 15))

axes_row_list = [0,0,1,1,2]
axes_col_list = [0,1,0,1,0]

c=0
for one_node in nnodes:

    print(f'Processing NCCL Test for {one_node} nodes')
    
    df = pd.read_excel('p5_all_red_25mb.xlsx',sheet_name=f'nodes{one_node}')

    df[['duration_ms','unit']] = df['Duration'].str.split(' ',expand = True)
    df['duration_ms'] = df['duration_ms'].astype(float)
    df.loc[df['unit']=='μs','duration_ms']=0.001*df.loc[df['unit']=='μs','duration_ms']

    NCCL_Test_allreduce = NCCL_Test_allreduce_ms[c]

    # Ignore anything beyond 25ms
    df = df[df['duration_ms'] < df['duration_ms'].quantile(0.99)]
    
    df.plot(ax=axes[axes_row_list[c],axes_col_list[c]], y='duration_ms', 
            use_index=True,linewidth=2, linestyle='--', marker='o')
    # Plot p99
    y_position = df['duration_ms'].quantile(0.99)
    axes[axes_row_list[c],axes_col_list[c]].axhline(y = y_position, color='k', linestyle='dashed', linewidth=1)
    # Add text on the horizontal line
    ax1 = axes[axes_row_list[c],axes_col_list[c]]
    #ax1.text(0.25, 10, f'p99 latency = {y_position:.2f} ms', color='r', va='center', ha='center',
    #    transform=ax1.get_yaxis_transform(), bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    #axes[axes_row_list[c],axes_col_list[c]].set_title(f'nnodes = {one_node}')

    

    c=c+1
# Add a super title to the entire figure
plt.suptitle(f'P5 All Reduce:Sum Message Size = 25 MB Num nodes = {one_node}', fontsize=16, fontweight='bold')

plt.tight_layout()
#plt.savefig(f'lineplot_all_reduce_sum_25MB_P5_nodes_{one_node}.png')
plt.show()   
