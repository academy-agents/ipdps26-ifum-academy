import asyncio
import logging
from typing import Any
import ifum_agent
import numpy as np
import time
from datetime import timedelta,datetime,timezone
import os
import concurrent.futures
import sys
import argparse

from academy.handle import Handle
from academy.exchange import ExchangeFactory
from academy.exchange import HybridExchangeFactory
from academy.manager import Manager
from academy.logger import init_logging
import parsl
from parsl.concurrent import ParslPoolExecutor
from parsl.dataflow.dflow import DataFlowKernelLoader

from ifum_agent import tasks
from ifum_agent.parsl import get_parsl_config
from ifum_agent.parsl import PARSL_CONFIGS
from ifum_agent.config import IfumConfig
from ifum_agent.config import get_ifum_config
from ifum_agent.agents.alignment import AlignmentAgent
from ifum_agent.agents.alignment import calculate_alignment_app

logger = logging.get_logger("run")

async def run_workflow(
    data_filenames: list[str],
    arc_filenames: list[str],
    flat_filenames: list[str],
    config: IfumConfig,
    alignment_agent_blue: Handle[AlignmentAgent],
    alignment_agent_red: Handle[AlignmentAgent],
    exchange: ExchangeFactory[Any],
    input_dir: str, # "/home/babnigg/globus/IFU-M/in/ut20240210"
    run_dir: str,
):
    start = time.time()
    os.makedirs(run_dir, exist_ok=True)
    colors = ["blue", "red"]
 
    # 1-STITCH: stitch and create file
    logger.info("Submitting stich tasks.")
    biased_files = {"blue": {}, "red": {}}
    for file in [*data_filenames, *arc_filenames, *flat_filenames]:
        ordered_files = tasks.load_files(input_dir, file)
        outputs = (
            os.path.join(run_dir, f"{file}_withbias_b.fits"),
            os.path.join(run_dir, f"{file}_withbias_r.fits")
        )
        save_future = tasks.save_file(ordered_files, file, config.bin_to_2x1, outputs)
        biased_files["blue"][file] = save_future.outputs[0]
        biased_files["red"][file] = save_future.outputs[1]

    # 2-BIAS: solve and save the bias
    logger.info("Submitting stich tasks.")
    denoised_files = {"blue": {}, "red": {}}
    for data_file, arc_file, flat_file in zip(data_filenames, arc_filenames, flat_filenames):
        for color in colors:
            internal_noise = tasks.calculate_intenal_noise(biased_files[color][flat_file])
            for file in (data_file, arc_file, flat_file):
                denoised_file = tasks.combined_bias_app(
                    biased_files[file][0], 
                    internal_noise, 
                    outputs = (os.path.join(run_dir,f"{file}_{color}.fits"),)
                )
                denoised_files[color][file] = denoised_file.outputs[0]

    # 3-CMRAY: academy agent
    logger.info("Submitting mask, rectify and calibrate apps.")
    cmray_files = {"blue": {}, "red": {}}
    for i, ref_file in enumerate(data_filenames):
        ref_files_denoised = denoised_files[ref_file]
        alignment_params_blue = {}
        alignment_params_red = {}
        for j, target_file in enumerate(data_filenames):
            if i == j:
                continue

            target_files_denoised = denoised_files[target_file]
            alignment_params_blue[target_files_denoised[0]] = calculate_alignment_app(
                alignment_agent_blue,
                exchange,
                i,
                ref_files_denoised[0],
                j,
                target_files_denoised[0],
            )

            alignment_params_red[target_files_denoised[1]] = calculate_alignment_app(
                alignment_agent_red,
                exchange,
                i,
                ref_files_denoised[1],
                j,
                target_files_denoised[1],
            )
        cosmic_ray_blue = tasks.generate_mask(
            ref_files_denoised[0], 
            alignment_agent_blue, 
            outputs=(os.path.join(run_dir, "{ref_file}_b_cmray_mask.fits"),)
        )
        cmray_files["blue"][ref_file] = cosmic_ray_blue.outputs[0]

        cosmic_ray_red = tasks.generate_mask(
            ref_files_denoised[1], 
            alignment_agent_red,
            outputs=(os.path.join(run_dir, "{ref_file}_r_cmray_mask.fits"),)
        )
        cmray_files["red"][ref_file] = cosmic_ray_red.outputs[1]

    
    futures = []
    center_deg = 5
    sigma_deg = 3
    sig_mult = 1.5
    logger.info("Submitting cosmic ray tasks.")
    for i, color in enumerate(colors):
        mask_args = {
            "bad_masks": config.bad_masks[i],
            "total_masks": config.total_masks,
            "mask_groups": config.mask_groups
        }
        
        for flatfilename, arcfilename, datafilename in zip(flat_filenames, arc_filenames, data_filenames):
            biased_file = biased_files[color][flatfilename]
            arc_file = denoised_files[color][arcfilename]
            data_file = denoised_files[color][datafilename]

            first_guess = tasks.first_guess_app(
                biased_file=biased_file,
                mask_args = mask_args,
            )
            outputs = (
                os.path.join(run_dir,f"{flatfilename}_{color}_mask_data.npz"),
            )
            mask_data = tasks.flat_mask_wrapped(
                biased_file,
                first_guess,
                mask_args,
                outputs,
            )
            outputs = (
                os.path.join(run_dir,f"{flatfilename}_{color}_trace_fits.npz"),
                os.path.join(run_dir,f"{flatfilename}_{color}_mask.fits"),
            )
            flatfile_mask_future = tasks.create_flatmask_app(
                biased_file,
                mask_data.outputs[0],
                mask_args,
                center_deg=center_deg,
                sigma_deg=sigma_deg,
                sig_mult=sig_mult,
                outputs=(),
            )
            arcfile_trace_future = tasks.optimize_arc_app(
                mask_args,
                biased_file,
                arc_file,
                flatfile_mask_future.outputs[0],
                sig_mult,
                expected_peaks = 25,
                optimize = True,
                outputs=os.path.join(run_dir,f"{arcfilename}_{color}_trace_fits.npz"),
            )

            datafile_trace_future = tasks.optimize_data_app(
                mask_args,
                biased_file,
                data_file,
                flatfile_mask_future.outputs[0],
                arcfile_trace_future.outputs[0],
                cmray_files["blue"][datafilename],
                sig_mult,
                expected_peaks = 25,
                optimize=True,
                outputs=(os.path.join(run_dir,f"{datafilename}_{color}_trace_fits.npz"),)
            )

            arc_mask_future = tasks.create_mask_app(
                mask_args,
                arc_file,
                arcfile_trace_future.outputs[0],
                sig_mult,
                flatfile_mask_future.outputs[0],
                outputs=(os.path.join(run_dir,f"{arcfilename}_{color}_mask.fits"),)
            )

            data_mask_future = tasks.create_mask_app(
                mask_args,
                data_file,
                datafile_trace_future.outputs[0],
                sig_mult,
                flatfile_mask_future.outputs[0],
                outputs=(os.path.join(run_dir,f"{datafilename}_{color}_mask.fits"),)
            )
            
            rectify_args = {
                "datadir": data_file,
                "arcdir": arc_file,
                "flatdir_biased": biased_file,
                "cmraymask": cmray_files[color][datafilename],
                "trace_data": datafile_trace_future.outputs[0],
                "trace_arc": arcfile_trace_future.outputs[0],
                "trace_flat": flatfile_mask_future.outputs[0],
                "wavelength": config.wavelength,
                "bad_masks": config.bad_masks[i],
                "total_masks": config.total_masks,
                "mask_groups": config.mask_groups
            }
            opt_center_data_future = tasks.optimize_center_app(
                rectify_args,
                data_mask_future.outputs[0],
                "data",
                fix_sparse=True,
                outputs=(os.path.join(run_dir,f"{datafilename}_{color}_centers.npz"),)
            )
            opt_center_arc_future = tasks.optimize_center_app(
                rectify_args,
                arc_mask_future.outputs[0],
                "arc",
                fix_sparse=True,
                outputs=(os.path.join(run_dir,f"{arcfilename}_{color}_centers.npz"),)
            )
            rect_data_future = tasks.rectify_app(
                rectify_args,
                opt_center_data_future.outputs[0],
                arc_or_data="data",
                outputs=(os.path.join(run_dir,f"{datafilename}_{color}_rect.npz"),)
            )
            rect_arc_future = tasks.rectify_app(
                rectify_args,
                opt_center_arc_future.outputs[0],
                arc_or_data="arc",
                outputs=(os.path.join(run_dir,f"{arcfilename}_{color}_rect.npz"),)
            )
            calib_future = tasks.calib_app(
                rectify_args,
                rect_data_future.outputs[0],
                rect_arc_future.outputs[0],
                use_sky=True,
                outputs=(os.path.join(run_dir,f"{datafilename}_{color}_calib.npz"),)
            )
            
            spectrum_bins = tasks.launch_specturm_fluxbins(
                data_file,
                cmray_files[color][datafilename],
                datafile_trace_future.outputs[0],
                calib_future.outputs[0],
                config.bad_masks[i],
                config.total_masks,
                sig_mult,
                config.bins
            )
            spectra_future = tasks.collect_spectra(
                data_file, 
                config.total_masks, 
                config.bins,
                config.bad_masks[i],
                *spectrum_bins,
                outputs=(os.path.join(run_dir,f"{datafilename}_{color}_bins.npz"),)
            )
            futures.append(spectra_future)

    concurrent.futures.wait(futures)
    print(f"total runtime: {str(timedelta(seconds=int(time.time()-start)))}", flush=True)

async def run(options: dict[str, Any]) -> None:
    # bad masks for each fiber shoe (on scale 1-x)
    bad_blues = [23]
    bad_reds = []

    # stars to use in WCS (list RA,Dec)
    #  all stars should be present in at least some dithers
    wcs_stars = [[74.8322, -58.6579],
                 [74.8305, -58.6603],
                 [74.8308, -58.6587],
                 [74.8254, -58.6572],
                 [74.8237, -58.6594]]
    
    # preparing inputs as function inputs
    config = get_ifum_config(
        bad_blues,
        bad_reds,
        wcs_stars,
        options["mode"],
        options["wavelength"],
    )

    parsl_config = get_parsl_config(options["parsl"], options["run_dir"], options["workers_per_node"])
    exchange = HybridExchangeFactory(
        options['redis_host'],
        options['redis_port'],
        options['interface']
    )
    executor = ParslPoolExecutor(parsl_config)
    DataFlowKernelLoader._dfk = executor.dfk # Share the executor between academy and parsl

    with Manager.from_exchange_factory(
        factory=exchange,
        executors=executor,
    ) as manager:
        blue_agent = manager.launch(AlignmentAgent)
        red_agent = manager.launch(AlignmentAgent)
        await run_workflow(
            data_filenames=["0721","0722","0723","0727","0728","0729","0736","0737","0738"],
            arc_filenames=["0725","0725","0725","0733","0733","0733","0740","0740","0740"],
            flat_filenames=["0724","0724","0724","0734","0734","0734","0739","0739","0739"],
            config=config,
            alignment_agent_blue=blue_agent,
            alignment_agent_red=red_agent,
            exchange=exchange,
            input_dir=options["input_dir"],
            run_dir=options["run_dir"],
        )

def parse_args() -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--parsl",
        choices=[
            *PARSL_CONFIGS.keys(),
        ], 
        default="htex-local",
    )
    parser.add_argument("--workers_per_node", type=int, default=48)
    parser.add_argument(
        "--mode",
        choices=["LR", "STD", "HR"],
        default="STD",
    )
    parser.add_argument(
        "--wavelength",
        choices=["far red", "blue"],
        default="far red"
    )
    parser.add_argument(
        "redis-host",
        type=str,
        required=True,
    )
    parser.add_argument(
        "redis-port",
        type=int,
        required=True,
    )
    parser.add_argument(
        "interface",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args

async def main() -> int:
    args = parse_args()
    init_logging(logfile = os.path.join(args["run_dir"], "ifum-log.txt"))
    await run(args)

if __name__ == "__main__":
    raise SystemExit(main())