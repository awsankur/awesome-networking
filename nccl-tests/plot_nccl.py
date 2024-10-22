import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

nnodes=[2,4,8,12,16]
OUTPUT_DIRECTORY_PATH='/Users/awsankur/Documents/PyTorch/awesome-networking-reports/p5_scaling_all_reduce/'
NCCL_Collective='all_reduce'


# NCCL MSG SIZES
nccl_msg_size_paths = glob.glob(f'{OUTPUT_DIRECTORY_PATH}/data' + '/*/')
nccl_msg_sizes = []
for a in nccl_msg_size_paths:
    nccl_msg_sizes.append(a.split('/')[-2])

def extract_nccl_test_result(OUTPUT_DIRECTORY_PATH,nnodes,nccl_msg_sizes):

    nccl_test_df = pd.DataFrame(columns=['nccl_msg_size','nnodes','value_ms'])
    for one_nccl_msg_size in nccl_msg_sizes:
        for i in nnodes:
            print(f'Reading slurm output file for {one_nccl_msg_size} bytes and {i} nodes...')
            slurm_file  = glob.glob(f'{OUTPUT_DIRECTORY_PATH}/data/{one_nccl_msg_size}/nodes_{i}/*.out')

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
                            #print(nccl_result)

            value_ms = round(0.001*float(nccl_result[5]),3)
            nccl_test_df = nccl_test_df._append({'nccl_msg_size': one_nccl_msg_size, 'nnodes': i, 'value_ms': value_ms}, ignore_index=True)


    return(nccl_test_df)

print('Extracting Data from NCCL Tests...')
nccl_test_df = extract_nccl_test_result(OUTPUT_DIRECTORY_PATH,nnodes,nccl_msg_sizes)

print(nccl_test_df)

nccl_test_df.to_csv(f'{OUTPUT_DIRECTORY_PATH}/nccl_test_df.csv')

# Generate plots for each NCCL Msg size vs nodes
# Three plots will be generated per Msg Size
# 1. Histogram charts
# 2. Line charts
# 3. Scaling vs nodes

latency_df = pd.DataFrame(columns = ['nnodes','p50_ms','p75_ms','p95_ms','p99_ms','pmax_ms','nccl_msg_size'])

for one_nccl_msg_size in nccl_msg_sizes:

    PLOT_PATH = f'{OUTPUT_DIRECTORY_PATH}/plots/scaling_vs_nodes/{one_nccl_msg_size}'
    os.makedirs(PLOT_PATH, exist_ok=True)


    ## Scaling plots

    p50_list = []
    p75_list = []
    p95_list = []
    p99_list = []
    pmax_list = []
    nccl_msg_list = []
    c=0
    for one_node in nnodes:

        print(f'Processing NCCL Test Data for {one_node} nodes')

        kernel_df = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/data/{one_nccl_msg_size}/nodes_{one_node}/kernel_df.csv')
        nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]
        df = nccldf.copy()
        df['duration_ms'] = nccldf['duration']/1000000

        p50_list.append(df['duration_ms'].quantile(0.50))
        p75_list.append(df['duration_ms'].quantile(0.75))
        p95_list.append(df['duration_ms'].quantile(0.95))
        p99_list.append(df['duration_ms'].quantile(0.99))
        pmax_list.append(max(df['duration_ms']))
        nccl_msg_list.append(one_nccl_msg_size)


    p_df = pd.DataFrame(list(zip(nnodes,p50_list,p75_list,p95_list,p99_list,pmax_list,nccl_msg_list)),
        columns = ['nnodes','p50_ms','p75_ms','p95_ms','p99_ms','pmax_ms','nccl_msg_size'])

    latency_df = pd.concat([latency_df, p_df], ignore_index=True)


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
    ax1.set_title(f'P5 {NCCL_Collective} Message Size = {one_nccl_msg_size}', fontsize=16, fontweight='bold')

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
    plt.savefig(f'{PLOT_PATH}/scaling_vs_nodes_{one_nccl_msg_size}_P5.png')
    #plt.show()


#----------------------------------------------------------------------------------------------------------------

    # Histogram charts

    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))

    axes_row_list = [0,0,1,1,2]
    axes_col_list = [0,1,0,1,0]
    
    c=0
    for one_node in nnodes:

        print(f'Generating histogram charts for {one_nccl_msg_size} msg size and {one_node} nodes')

        kernel_df = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/data/{one_nccl_msg_size}/nodes_{one_node}/kernel_df.csv')

        # Subset df with relevant nccl kernels
        nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]

        df = nccldf.copy()
        df['duration_ms'] = nccldf['duration']/1000000


        # Ignore anything beyond 99th percentile
        df = df[df['duration_ms'] < df['duration_ms'].quantile(0.99)]


        NCCL_Test = nccl_test_df.loc[((nccl_test_df['nccl_msg_size'] == one_nccl_msg_size) & (nccl_test_df['nnodes'] == one_node)),'value_ms' ].reset_index()['value_ms'][0]

        ax1 = axes[axes_row_list[c],axes_col_list[c]]

        df.hist(column = 'duration_ms',ax = ax1, density=True, bins=100, edgecolor='blue')

        ax1.axvline(NCCL_Test, color='k', linestyle='dashed', linewidth=1)

        ax1.text(NCCL_Test, 0.5, f'NCCL Test Avg', color='r',
                 rotation=90, va='center', ha='center',
                 transform=ax1.get_xaxis_transform(),
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        ax1.set_title(f'nnodes = {one_node}')
        c=c+1

    plt.suptitle(f'P5 {NCCL_Collective} Message Size = {one_nccl_msg_size} Num nodes = {one_node}', fontsize=16, fontweight='bold')
    plt.savefig(f'{PLOT_PATH}/histogram_{one_nccl_msg_size}_P5_nodes_{one_node}.png')


#----------------------------------------------------------------------------------------------------------------

    # Line charts

    # Create subplots
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))

    axes_row_list = [0,0,1,1,2]
    axes_col_list = [0,1,0,1,0]
    
    c=0
    for one_node in nnodes:

        print(f'Generating Line Charts for {one_nccl_msg_size} msg size and {one_node} nodes')

        kernel_df = pd.read_csv(f'{OUTPUT_DIRECTORY_PATH}/data/{one_nccl_msg_size}/nodes_{one_node}/kernel_df.csv')

        # Subset df with relevant nccl kernels
        nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]

        df = nccldf.copy()
        df['duration_ms'] = nccldf['duration']/1000000


        # Ignore anything beyond 99th percentile
        df = df[df['duration_ms'] < df['duration_ms'].quantile(0.99)]


        NCCL_Test = nccl_test_df.loc[((nccl_test_df['nccl_msg_size'] == one_nccl_msg_size) & (nccl_test_df['nnodes'] == one_node)),'value_ms' ].reset_index()['value_ms'][0]

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

    plt.suptitle(f'P5 {NCCL_Collective} Message Size = {one_nccl_msg_size} Num nodes = {one_node}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_PATH}/lineplot_{one_nccl_msg_size}_P5_nodes_{one_node}.png')

# Write latency_df to a csv
latency_df.to_csv(f'{OUTPUT_DIRECTORY_PATH}/latency_df.csv')
#----------------------------------------------------------------------------------------------------------------
# Generate plots for each nnode vs msg sizes
# Three plots will be generated per Msg Size
# 1. Scaling vs nccl message sizes

for one_node in nnodes:

    PLOT_PATH = f'{OUTPUT_DIRECTORY_PATH}/plots/scaling_vs_nccl_msg_sizes/{one_node}'
    os.makedirs(PLOT_PATH, exist_ok=True)

    print(f'Generating Latency vs NCCL Msg Size plots for {one_node} nodes')

    latency_one_node_df = latency_df.loc[latency_df['nnodes']==one_node,]

     # Create subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

    ax1 = axes[0,0]
    # List of colors
    colors = ['red', 'green', 'blue']

    # Plot each column as a separate line
    for i, column in enumerate(latency_one_node_df.columns[1:]):
        if i < 3:
            ax1.plot(latency_one_node_df['nccl_msg_size'], latency_one_node_df[column], color=colors[i],
                     label=column, linewidth=2, linestyle='--', marker='o')

    # Add legend and labels
    ax1.legend()
    ax1.set_xlabel('NCCL Msg Size Bytes', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
    ax1.set_title(f'P5 {NCCL_Collective} Number of nodes = {one_node}', fontsize=16, fontweight='bold')

    ax2 = axes[0,1]
    ax2.plot(latency_one_node_df['nccl_msg_size'], latency_one_node_df['p99_ms'], 'black',
                     label='p99_ms', linewidth=2, linestyle='--', marker='o')
    # Add legend and labels
    ax2.legend()
    ax2.set_xlabel('NCCL Msg Size Bytes', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
    ax2.set_title('P99 Latency', fontsize=16, fontweight='bold')

    ax3 = axes[1,0]
    ax3.plot(latency_one_node_df['nccl_msg_size'], latency_one_node_df['pmax_ms'], 'black',
                     label='pmax_ms', linewidth=2, linestyle='--', marker='o')
    # Add legend and labels
    ax3.legend()
    ax3.set_xlabel('NCCL Msg Size Bytes', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Latency in ms', fontsize=10, fontweight='bold')
    ax3.set_title('Max Latency', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{PLOT_PATH}/scaling_vs_nccl_msg_size_nodes_{one_node}_P5.png')

