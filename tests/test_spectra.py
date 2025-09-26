import os
from typing import Any
import pytest
import numpy as np
from parsl import File

from ifum_agent.tasks.helper import get_spectrum_fluxbins
from ifum_agent.tasks.helper import launch_spectrum_fluxbins
from ifum_agent.tasks.helper import collect_spectra

def test_get_spectrum_fluxbins(data_dir: str):
    npzdata = np.load(os.path.join(data_dir, "0722_blue_trace_fits.npz"))
    sigma_traces = npzdata["init_traces_sigma"]

    get_spectrum_fluxbins(
        0,
        np.arange(7000,10000,1),
        sigma_traces,
        1.5,
        os.path.join(data_dir, "0722_blue.fits"),
        os.path.join(data_dir, "0722_b_cmray_mask.fits"),
        os.path.join(data_dir, "0722_blue_trace_fits.npz"),
        os.path.join(data_dir, "0722_blue_calib.npz"),
    )

def test_launch_spectrum_fluxbin(data_dir: str):
    future = launch_spectrum_fluxbins(
        os.path.join(data_dir, "0722_blue.fits"),
        os.path.join(data_dir, "0722_b_cmray_mask.fits"),
        os.path.join(data_dir, "0722_blue_trace_fits.npz"),
        os.path.join(data_dir, "0722_blue_calib.npz"),
        np.array([2,],),
        6,
        1.5,
        np.arange(7000,10000,1),
    )
    results = future.result()
    assert len(results) == 3

def test_collect_spectrum(data_dir: str, tmp_path: str) -> None:
    spectrum_bins = launch_spectrum_fluxbins(
        os.path.join(data_dir, "0722_blue.fits"),
        os.path.join(data_dir, "0722_b_cmray_mask.fits"),
        os.path.join(data_dir, "0722_blue_trace_fits.npz"),
        os.path.join(data_dir, "0722_blue_calib.npz"),
        np.array([2,],),
        6,
        1.5,
        np.arange(7000,10000,1),
    )

    future = collect_spectra(
        os.path.join(data_dir, "0722_blue.fits"),
        6,
        np.arange(7000,10000,1),
        np.array([2,],),
        spectrum_bins,
        outputs=[File(os.path.join(tmp_path, "0722_blue_bins.npz")),]
    )
    future.result()
