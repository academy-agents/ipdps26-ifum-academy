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
from academy.logging import init_logging
import parsl
from parsl.concurrent import ParslPoolExecutor
from parsl.dataflow.dflow import DataFlowKernelLoader
from parsl import File

from ifum_agent import tasks
from ifum_agent.parsl import get_parsl_config
from ifum_agent.parsl import PARSL_CONFIGS
from ifum_agent.config import IfumConfig
from ifum_agent.config import get_ifum_config
from ifum_agent.agents.alignment import AlignmentAgent
from ifum_agent.agents.alignment import calculate_alignment_app

logger = logging.getLogger("run")

async def run_workflow(
    data_filenames: list[str],
    arc_filenames: list[str],
    flat_filenames: list[str],
    config: IfumConfig,
    alignment_agent_blue: Handle[AlignmentAgent],
    alignment_agent_red: Handle[AlignmentAgent],
    exchange: ExchangeFactory[Any],
    input_dir: str,
    run_dir: str,
):
    start = time.time()
    os.makedirs(run_dir, exist_ok=True)
    colors = ["blue", "red"]
    futures: list[concurrent.futures.Future] = []
 
    # 1-STITCH: stitch and create file
    logger.info("Submitting stich tasks.")
    biased_files = {"blue": {}, "red": {}}
    for file in set([*data_filenames, *arc_filenames, *flat_filenames]):
        ordered_files = tasks.load_files(input_dir, file)
        outputs = [
            File(os.path.join(run_dir, f"{file}_withbias_b.fits")),
            File(os.path.join(run_dir, f"{file}_withbias_r.fits")),
        ]
        save_future = tasks.save_file(
            files=ordered_files,
            filename=file,
            bin_to_2x1=config.bin_to_2x1, 
            outputs=outputs
        )
        futures.append(save_future)
        biased_files["blue"][file] = save_future.outputs[0]
        biased_files["red"][file] = save_future.outputs[1]

    # 2-BIAS: solve and save the bias
    logger.info("Submitting denoising tasks.")
    denoised_files = {"blue": {}, "red": {}}
    for data_file, arc_file, flat_file in zip(data_filenames, arc_filenames, flat_filenames):
        for color in colors:
            internal_noise = tasks.calculate_intenal_noise(biased_files[color][flat_file])
            for file in (data_file, arc_file, flat_file):
                denoised_file = tasks.combined_bias_app(
                    biased_files[color][file], 
                    internal_noise, 
                    outputs = [File(os.path.join(run_dir,f"{file}_{color}.fits")),]
                )
                futures.append(denoised_file)
                denoised_files[color][file] = denoised_file.outputs[0]

    # 3-CMRAY: academy agent
    logger.info("Submitting cosmic ray tasks.")
    cmray_files = {"blue": {}, "red": {}}
    for i, ref_file in enumerate(data_filenames):
        for color, agent in zip(colors, (alignment_agent_blue,)):
            ref_file_denoised = denoised_files[color][ref_file]
            alignment_params = {}
            for j, target_file in enumerate(data_filenames):
                if i == j:
                    continue

                target_file_denoised = denoised_files[color][target_file]
                alignment_params[target_file_denoised] = calculate_alignment_app(
                    asyncio.get_running_loop(),
                    agent,
                    exchange,
                    i,
                    ref_file_denoised,
                    j,
                    target_file_denoised,
                )
                futures.append(alignment_params[target_file_denoised])
                # await asyncio.wait((asyncio.wrap_future(alignment_params[target_file_denoised]),))

            cosmic_ray = tasks.generate_mask(
                ref_file_denoised, 
                alignment_params, 
                outputs=[File(os.path.join(run_dir, f"{ref_file}_b_cmray_mask.fits")),]
            )
            futures.append(cosmic_ray)
            cmray_files[color][ref_file] = cosmic_ray.outputs[0]
    
    center_deg = 5
    sigma_deg = 3
    sig_mult = 1.5
    logger.info("Submitting mask, rectify and calibrate apps.")
    for i, color in enumerate(colors):
        mask_args = {
            "bad_masks": config.bad_masks[i],
            "total_masks": config.total_masks,
            "mask_groups": config.mask_groups
        }
        
        flatfile_mask_futures = {}
        arc_trace_futures = {}
        arc_mask_futures = {}
        arc_center_futures = {}
        arc_rect_futures = {}
        for flatfilename, arcfilename, datafilename in zip(flat_filenames, arc_filenames, data_filenames):
            biased_file = biased_files[color][flatfilename]
            arc_file = denoised_files[color][arcfilename]
            data_file = denoised_files[color][datafilename]

            if flatfilename not in flatfile_mask_futures:
                first_guess = tasks.first_guess_app(
                    biased_file=biased_file,
                    mask_args = mask_args,
                )
                outputs = [
                    File(os.path.join(run_dir,f"{flatfilename}_{color}_mask_data.npz")),
                ]
                mask_data = tasks.flat_mask_wrapped(
                    biased_file,
                    first_guess,
                    mask_args,
                    outputs=outputs,
                )
                outputs = [
                    File(os.path.join(run_dir,f"{flatfilename}_{color}_trace_fits.npz")),
                    File(os.path.join(run_dir,f"{flatfilename}_{color}_mask.fits")),
                ]
                flatfile_mask_future = tasks.create_flatmask_app(
                    biased_file,
                    mask_data.outputs[0],
                    mask_args,
                    center_deg=center_deg,
                    sigma_deg=sigma_deg,
                    sig_mult=sig_mult,
                    outputs=outputs,
                )
                flatfile_mask_futures[flatfilename] = flatfile_mask_future
            else:
                flatfile_mask_future = flatfile_mask_futures[flatfilename]

            if arcfilename not in arc_trace_futures:
                arcfile_trace_future = tasks.optimize_arc_app(
                    mask_args,
                    flatfile_mask_future.outputs[1],
                    arc_file,
                    flatfile_mask_future.outputs[0],
                    sig_mult,
                    expected_peaks = 25,
                    optimize = True,
                    outputs=[File(os.path.join(run_dir,f"{arcfilename}_{color}_trace_fits.npz")),],
                )
                arc_trace_futures[arcfilename] = arcfile_trace_future
            else:
                arcfile_trace_future = arc_trace_futures[arcfilename]

            datafile_trace_future = tasks.optimize_data_app(
                mask_args,
                flatfile_mask_future.outputs[1],
                data_file,
                flatfile_mask_future.outputs[0],
                arcfile_trace_future.outputs[0],
                cmray_files[color][datafilename],
                sig_mult,
                expected_peaks = 25,
                optimize=True,
                outputs=[File(os.path.join(run_dir,f"{datafilename}_{color}_trace_fits.npz")),]
            )
            futures.append(datafile_trace_future)

            if arcfilename not in arc_mask_futures:
                arc_mask_future = tasks.create_mask_app(
                    mask_args,
                    arc_file,
                    arcfile_trace_future.outputs[0],
                    sig_mult,
                    flatfile_mask_future.outputs[0],
                    outputs=[File(os.path.join(run_dir,f"{arcfilename}_{color}_mask.fits")),]
                )
                arc_mask_futures[arcfilename] = arc_mask_future
            else:
                arc_mask_future = arc_mask_futures[arcfilename]
            futures.append(arc_mask_future)

            data_mask_future = tasks.create_mask_app(
                mask_args,
                data_file,
                datafile_trace_future.outputs[0],
                sig_mult,
                flatfile_mask_future.outputs[0],
                outputs=[File(os.path.join(run_dir,f"{datafilename}_{color}_mask.fits")),]
            )
            futures.append(data_mask_future)
            
            rectify_args = {
                "datadir": data_file,
                "arcdir": arc_file,
                "flatdir_biased": biased_file,
                "cmraymask": cmray_files[color][datafilename],
                "trace_data": datafile_trace_future.outputs[0],
                "trace_arc": arcfile_trace_future.outputs[0],
                "trace_flat": flatfile_mask_future.outputs[0],
                "wavelength": config.wavelength,
                "bad_mask": config.bad_masks[i],
                "total_masks": config.total_masks,
                "mask_groups": config.mask_groups
            }
            opt_center_data_future = tasks.optimize_center_app(
                rectify_args,
                arc_mask_future.outputs[0],
                "data",
                fix_sparse=True,
                outputs=[File(os.path.join(run_dir,f"{datafilename}_{color}_centers.npz")),],
            )
            futures.append(opt_center_data_future)

            if arcfilename not in arc_center_futures:
                opt_center_arc_future = tasks.optimize_center_app(
                    rectify_args,
                    arc_mask_future.outputs[0],
                    "arc",
                    fix_sparse=True,
                    outputs=[File(os.path.join(run_dir,f"{arcfilename}_{color}_centers.npz")),],
                )
                arc_center_futures[arcfilename] = opt_center_arc_future
            else:
                opt_center_arc_future = arc_center_futures[arcfilename]

            futures.append(opt_center_arc_future)

            rect_data_future = tasks.rectify_app(
                rectify_args,
                opt_center_data_future.outputs[0],
                arc_or_data="data",
                outputs=[File(os.path.join(run_dir,f"{datafilename}_{color}_rect.npz")),]
            )
            futures.append(rect_data_future)

            if arcfilename not in arc_rect_futures:
                rect_arc_future = tasks.rectify_app(
                    rectify_args,
                    opt_center_arc_future.outputs[0],
                    arc_or_data="arc",
                    outputs=[File(os.path.join(run_dir,f"{arcfilename}_{color}_rect.npz")),]
                )
                arc_rect_futures[arcfilename] = rect_arc_future
            else:
                rect_arc_future = arc_rect_futures[arcfilename]

            futures.append(rect_arc_future)

            calib_future = tasks.calib_app(
                rectify_args,
                rect_data_future.outputs[0],
                rect_arc_future.outputs[0],
                use_sky=True,
                outputs=[File(os.path.join(run_dir,f"{datafilename}_{color}_calib.npz")),]
            )
            futures.append(calib_future)

            spectrum_bins = tasks.launch_spectrum_fluxbins(
                data_file,
                cmray_files[color][datafilename],
                datafile_trace_future.outputs[0],
                calib_future.outputs[0],
                config.bad_masks[i],
                config.total_masks,
                sig_mult,
                config.bins,
            )
            futures.append(spectrum_bins)

            spectra_future = tasks.collect_spectra(
                data_file, 
                config.total_masks, 
                config.bins,
                config.bad_masks[i],
                spectrum_bins,
                outputs=[File(os.path.join(run_dir,f"{datafilename}_{color}_bins.npz")),]
            )
            futures.append(spectra_future)
    
    done, _ = await asyncio.wait([asyncio.wrap_future(f) for f in futures])
    exception = None
    for f in done:
        if f.exception() is not None:
            logger.exception(f.exception())
            exception = f.exception()
    if exception:
        raise exception
    
    print(f"total runtime: {str(timedelta(seconds=int(time.time()-start)))}", flush=True)

async def run(options: dict[str, Any]) -> None:
    # bad masks for each fiber shoe (on scale 1-x)
    bad_blues = [23,]
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
        interface=options['interface'],
    )
    # executor = ParslPoolExecutor(parsl_config)
    # DataFlowKernelLoader._dfk = executor.dfk # Share the executor between academy and parsl

    with parsl.load(parsl_config):
        async with await Manager.from_exchange_factory(
            factory=exchange,
            executors=concurrent.futures.ProcessPoolExecutor(max_workers=2, initializer=init_logging,),
        ) as manager:
            logger.info("Starting agents")
            blue_agent = await manager.launch(AlignmentAgent)
            red_agent = await manager.launch(AlignmentAgent)

            await blue_agent.noop()
            await red_agent.noop()
            logger.info("Agents started.")

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
    parser.add_argument(
        "--workers-per-node",
        type=int,
        default=48
    )
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
        "--redis-host",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--redis-port",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--interface",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args

async def main() -> int:
    args = parse_args()
    options = vars(args)
    init_logging(logfile = os.path.join(options["run_dir"], "ifum-log.txt"))
    await run(options)

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))