# https://parsl.readthedocs.io/en/stable/userguide/configuration/examples.html#midway-rcc-uchicago

from typing import Literal, NamedTuple
from parsl.addresses import address_by_interface, address_by_hostname
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider, LocalProvider
from parsl.usage_tracking.levels import LEVEL_1
from parsl.monitoring.monitoring import MonitoringHub

import numpy as np
import os

class IfumConfig(NamedTuple):
    bad_masks: np.ndarray[int]
    wcs_stars: np.ndarray[float]
    total_masks: int
    mask_groups: int
    hex_dims: tuple[int, int]
    wavelength: Literal["far red", "blue"]
    bins: np.ndarray[int]
    bin_to_2x1: bool = True
    sig_mult: float = 1.5

def get_ifum_config(
    bad_blues: list[int],
    bad_reds: list[int],
    wcs_stars: list[list[float]],
    mode: Literal["STD", "HR"],
    wavelength: Literal["far red", "blue"],
    bin_to_2x1: bool = True,
    sig_mult: float = 1.5,
) -> IfumConfig:
    bad_masks = [np.array(bad_blues)-1,np.array(bad_reds)-1]
    wcs_stars = np.array(wcs_stars)
    if mode == "STD":
        total_masks = 552
        mask_groups = 12
        hex_dims = (23,24)
    elif mode == "HR":
        total_masks = 864
        mask_groups = 16
        hex_dims = (27,32)
    else:
        print("invalid mode")

    if wavelength == "far red":
        bins = np.arange(7000,10000,1)
    elif wavelength == "blue":
        bins = np.arange(4000,6100,1)
    else:
        print("invalid wavelength")

    return IfumConfig(
        bad_masks=bad_masks,
        wcs_stars=wcs_stars,
        total_masks=total_masks,
        mask_groups=mask_groups,
        hex_dims=hex_dims,
        wavelength=wavelength,
        bins=bins,
        bin_to_2x1=bin_to_2x1,
        sig_mult=sig_mult,
    )