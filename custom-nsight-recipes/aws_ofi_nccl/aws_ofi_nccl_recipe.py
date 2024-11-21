# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from datetime import datetime
from pathlib import Path

import pandas as pd

from nsys_recipe import log
from nsys_recipe.data_service import DataService
from nsys_recipe.lib import data_utils, helpers, recipe, summary
from nsys_recipe.lib.args import Option
from nsys_recipe.log import logger
from nsys_recipe.lib.table_config import CompositeTable
from nsys_recipe.lib import helpers, pace, recipe, summary

import matplotlib.pyplot as plt
import os

class AwsOfiNcclRecipe(recipe.Recipe):
    @staticmethod
    def _mapper_func(report_path, parsed_args):

        service = DataService(report_path, parsed_args)

        service.queue_table("StringIds")

        # Get Cuda GPU Kernel Data
        service.queue_table("CUPTI_ACTIVITY_KIND_KERNEL", ["shortName", "start", "end", "deviceId"])


        # Get NVTX Data
        service.queue_custom_table(CompositeTable.NVTX)

        # Create data frames
        df_dict = service.read_queued_tables()
        if df_dict is None:
            return None

        kernel_df = df_dict["CUPTI_ACTIVITY_KIND_KERNEL"]
        service.filter_and_adjust_time(kernel_df)

        kernel_df = data_utils.replace_id_with_value(
            kernel_df, df_dict["StringIds"], "shortName", "name"
        )

        if kernel_df.empty:
            logger.info(
                f"{report_path}: Report was successfully processed, but no NCCL data was found."
            )
            return None

        kernel_df["duration"] = kernel_df["end"] - kernel_df["start"]
        kernel_df = kernel_df[kernel_df["duration"] > 0]

        nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]

        nvtx_df = df_dict[CompositeTable.NVTX]
        ##service.filter_and_adjust_time(nvtx_df)

        ##nvtx_df.to_csv('./nvtx_df.csv')

        if nvtx_df.empty:
            logger.info(
                f"{report_path}: Report was successfully processed, but no data was found."
            )
            return None

        filename = Path(report_path).stem

        return filename, nccldf, nvtx_df


    @log.time("Mapper")
    def mapper_func(self, context):
        return context.wait(
            context.map(
                self._mapper_func,
                self._parsed_args.input,
                parsed_args=self._parsed_args,
            )
        )

    def get_domain_ids(self,mapper_res):
        #filtered_res = helpers.filter_none(mapper_res)
        filtered_res = mapper_res
        # Sort by file name.
        filtered_res = sorted(filtered_res, key=lambda x: x[0])
        filenames, nccldf_tuple, nvtx_df_tuple = zip(*filtered_res)

        nccldf = nccldf_tuple[0]
        nvtx_df = nvtx_df_tuple[0]


        domain_df = nvtx_df.loc[nvtx_df['text'].str.contains('aws', case=False)==True,]

        send_domain_ids = domain_df.loc[domain_df['text'].str.contains('s_comm')==True,'domainId'].to_list()

        recv_domain_ids = domain_df.loc[domain_df['text'].str.contains('r_comm')==True,'domainId'].to_list()

        domain_names = domain_df['text'].to_list()

        domain_Id_min = min(domain_df['domainId'])
        domain_Id_max = max(domain_df['domainId'])

        aws_ofi_nccl_df = nvtx_df.loc[(nvtx_df['domainId']>=domain_Id_min)
                                & (nvtx_df['domainId']<=domain_Id_max),]

        data_path = f'{self.get_output_dir()}/data'
        os.makedirs(data_path, exist_ok=True)

        nccldf.to_parquet(f"{data_path}/nccldf.parquet")
        nvtx_df.to_parquet(f"{data_path}/nvtx_df.parquet")

        if self._parsed_args.csv:
            nccldf.to_csv(f"{data_path}/nccldf.csv")
            nvtx_df.to_csv(f"{data_path}/nccldf.csv")

        # Domain ID ==0 for Eagr_recv??
        return (aws_ofi_nccl_df, domain_df, send_domain_ids, recv_domain_ids, domain_names)

    def plot_send_receive_events(self,aws_ofi_nccl_df,domain_ids,domain_names,send_or_recv_tag):

        for i in domain_ids:

            domain_name = domain_names[i-2]

            print(f'Generating {send_or_recv_tag} event plots for {domain_name}....')
            plot_path = f'{self.get_output_dir()}/plots/{domain_name}'

            os.makedirs(plot_path, exist_ok=True)

            # Create subplots
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

            one_df = aws_ofi_nccl_df.loc[(aws_ofi_nccl_df['domainId']==i) &
                                           (aws_ofi_nccl_df['text']==send_or_recv_tag),]

            one_df = one_df.reset_index()

            one_df['duration_micro_s'] = (one_df['end'] - one_df['start'])/1000
            one_df.hist(column = 'duration_micro_s', ax = axes[0],density=True, bins=100, edgecolor='blue')
            one_df.plot(ax = axes[1],y='duration_micro_s', use_index=True,linewidth=2, linestyle='--', marker='o')
            plt.suptitle(f'{send_or_recv_tag} event durations for {domain_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{plot_path}/{{send_or_recv_tag}}_events.png')

    def plot_plugin_send_delay(self,aws_ofi_nccl_df,send_domain_ids,domain_names):
        # Control delay: Time since NCCL gives a Send op til we are ready for RDMA op

        for i in send_domain_ids:

            plot_path = f'{self.get_output_dir()}/plots/{domain_name}'

            os.makedirs(plot_path, exist_ok=True)

            domain_name = domain_names[i-2]

            print(f'Generating plugin send delay plots for {domain_name}....')
            one_send_comm_df = aws_ofi_nccl_df.loc[(aws_ofi_nccl_df['domainId']==i) ,]
            one_send_comm_df = one_send_comm_df.reset_index()

            # Map Send_write_seg events with Send_ctrl_recv events
            counter_send_write_seg = 0
            counter_send_ctrl_recv = 0
            one_send_comm_df['counter'] = 0
            for one_send_comm_df_index, one_send_comm_df_row in one_send_comm_df.iterrows():
                if one_send_comm_df_row['text']=='Send_write_seg':
                    counter_send_write_seg = counter_send_write_seg + 1
                    one_send_comm_df.loc[one_send_comm_df_index,'counter']= counter_send_write_seg
                if one_send_comm_df_row['text']=='Send_ctrl_recv':
                    counter_send_ctrl_recv = counter_send_ctrl_recv + 1
                    one_send_comm_df.loc[one_send_comm_df_index,'counter'] = counter_send_ctrl_recv

            # Compute plugin send delay
            plugin_send_delay = []
            for j in range(min(counter_send_write_seg,counter_send_ctrl_recv)):
                j = j+1

                end_ts = one_send_comm_df.loc[(one_send_comm_df['text']=='Send_write_seg') & (one_send_comm_df['counter']==j),'start'].to_list()[0]
                start_ts = one_send_comm_df.loc[(one_send_comm_df['text']=='Send_ctrl_recv') & (one_send_comm_df['counter']==j),'start'].to_list()[0]

                latency_micro_sec = (end_ts - start_ts)/1000

                plugin_send_delay.append(latency_micro_sec)


            # Plot plugin send delay
            df_plugin_send_delay = pd.DataFrame(plugin_send_delay, columns=['Value'])

            # Create subplots
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))

            df_plugin_send_delay.hist(column = 'duration_micro_s', ax = axes[0],density=True, bins=100, edgecolor='blue')
            df_plugin_send_delay.plot(ax = axes[1],y='duration_micro_s', use_index=True,linewidth=2, linestyle='--', marker='o')
            plt.suptitle(f'Plugin send delay event durations for {domain_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'{plot_path}/plugin_send_delay_events.png')

    @log.time("Reducer")
    def reducer_func(self, mapper_res):
        filtered_res = helpers.filter_none(mapper_res)
        # Sort by file name.
        filtered_res = sorted(filtered_res, key=lambda x: x[0])
        filenames, nccldf, nvtx_df = zip(*filtered_res)

        files_df = pd.DataFrame({"File": filenames}).rename_axis("Rank")
        files_df.to_parquet(self.add_output_file("files.parquet"))

        stats_by_device_dfs = [
            df.assign(Rank=rank) for rank, df in enumerate(stats_by_device_dfs)
        ]
        rank_stats_by_device_df = pd.concat(stats_by_device_dfs)
        rank_stats_by_device_df.to_parquet(
            self.add_output_file("rank_stats_by_device.parquet")
        )

        stats_dfs = [df.assign(Rank=rank) for rank, df in enumerate(stats_dfs)]
        stats_df = pd.concat(stats_dfs)

        rank_stats_df = stats_df
        rank_stats_df.to_parquet(self.add_output_file("rank_stats.parquet"))

        all_stats_df = summary.aggregate_stats_df(stats_df, index_col="Name")
        all_stats_df.to_parquet(self.add_output_file("all_stats.parquet"))

        if self._parsed_args.csv:
            files_df.to_csv(self.add_output_file("files.csv"))
            all_stats_df.to_csv(self.add_output_file("all_stats.csv"))
            rank_stats_df.to_csv(self.add_output_file("rank_stats.csv"))
            rank_stats_by_device_df.to_csv(
                self.add_output_file("rank_stats_by_device.csv")
            )

    def save_notebook(self):
        self.create_notebook("stats.ipynb")
        self.add_notebook_helper_file("nsys_display.py")

    def save_analysis_file(self):
        self._analysis_dict.update(
            {
                "EndTime": str(datetime.now()),
                "Outputs": self._output_files,
            }
        )
        self.create_analysis_file()

    def run(self, context):
        super().run(context)

        mapper_res = self.mapper_func(context)
        domain_tuple = self.get_domain_ids(mapper_res)

        aws_ofi_nccl_df = domain_tuple[0]
        domain_df = domain_tuple[1]
        send_domain_ids = domain_tuple[2]
        recv_domain_ids = domain_tuple[3]
        domain_names = domain_tuple[4]

        self.plot_send_receive_events(aws_ofi_nccl_df,send_domain_ids,domain_names,'Send')
        self.plot_send_receive_events(aws_ofi_nccl_df,send_domain_ids,domain_names,'Recv')
        self.plot_plugin_send_delay(self,aws_ofi_nccl_df,send_domain_ids,domain_names)

        #self.reducer_func(mapper_res)

        #self.save_notebook()
        self.save_analysis_file()

    @classmethod
    def get_argument_parser(cls):
        parser = super().get_argument_parser()

        parser.add_recipe_argument(Option.INPUT, required=True)
        parser.add_recipe_argument(Option.START)
        parser.add_recipe_argument(Option.END)
        parser.add_recipe_argument(Option.CSV)

        filter_group = parser.recipe_group.add_mutually_exclusive_group()
        parser.add_argument_to_group(filter_group, Option.FILTER_TIME)
        parser.add_argument_to_group(filter_group, Option.FILTER_NVTX)

        return parser