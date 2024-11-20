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


        kernel_df = kernel_df_orig.sort_values(by='start')
        nccldf = kernel_df[kernel_df.apply(lambda x: x.astype(str).str.contains('nccl', case=False).any(), axis=1)]

        nvtx_df = df_dict[CompositeTable.NVTX]
        service.filter_and_adjust_time(nvtx_df)

        ##nvtx_df.to_csv('./nvtx_df.csv')

        if nvtx_df.empty:
            logger.info(
                f"{report_path}: Report was successfully processed, but no data was found."
            )
            return None

        filename = Path(report_path).stem

        nccldf.to_parquet(self.add_output_file("nccldf.parquet"))
        nvtx_df.to_parquet(self.add_output_file("nvtx_df.parquet"))

        if self._parsed_args.csv:
            nccldf.to_csv(self.add_output_file("nccldf.csv"))
            nvtx_df.to_csv(self.add_output_file("nvtx_df.csv"))

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
        filtered_res = helpers.filter_none(mapper_res)
        # Sort by file name.
        filtered_res = sorted(filtered_res, key=lambda x: x[0])
        filenames, nccldf, nvtx_df = zip(*filtered_res)

        domain_df = nvtx_df.loc[nvtx_df['text'].str.contains('aws', case=False)==True,]

        send_domain_ids = domain_df.loc[domain_df['text'].str.contains('s_comm')==True,'domainId'].to_list()

        recv_domain_ids = domain_df.loc[domain_df['text'].str.contains('r_comm')==True,'domainId'].to_list()

        domain_names = domain_df['text'].to_list()

        aws_ofi_nccl_df = nvtx_df.loc[(nvtx_df['domainId']>=domain_Id_min)
                                & (nvtx_df['domainId']<=domain_Id_max),]

        
        # Domain ID ==0 for Eagr_recv??
        return (aws_ofi_nccl_df, domain_df, send_domain_ids, recv_domain_ids, domain_names)

    def plot_send_events(aws_ofi_nccl_df, send_domain_ids, domain_names):


        for i in send_domain_ids:

            domain_name = domain_names[i-2]

            os.makedirs(f'{self.get_output_dir()}/{domain_name}', exist_ok=True)

            # Create subplots
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
            
            one_send_df = aws_ofi_nccl_df.loc[(aws_ofi_nccl_df['domainId']==i) & 
                                           (aws_ofi_nccl_df['text']=='Send'),]
            
            one_send_df = one_send_df.reset_index()

            one_send_df['duration_micro_s'] = (one_send_df['end'] - one_send_df['start'])/1000
            one_send_df.hist(column = 'duration_micro_s', ax = axes[0],density=True, bins=100, edgecolor='blue')
            one_send_df.plot(ax = axes[1],y='duration_micro_s', use_index=True,linewidth=2, linestyle='--', marker='o')
            plt.suptitle(f'Send event durations for {domain_name}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig





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