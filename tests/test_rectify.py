import os
from typing import Any
import pytest
import numpy as np
from parsl import File

from ifum_agent.tasks.rectify import Rectify
from ifum_agent.tasks.rectify import optimize_center_app
from ifum_agent.tasks.rectify import rectify_app
from ifum_agent.tasks.rectify import calib_app

@pytest.fixture
def rectify_args(data_dir) -> dict[str, Any]:
    rectify_args = {
        "datadir": os.path.join(data_dir, "0722_blue.fits"),
        "arcdir": os.path.join(data_dir, "0725_blue.fits"),
        "flatdir_biased": os.path.join(data_dir, "0724_withbias_b.fits"),
        "cmraymask": os.path.join(data_dir, "0722_b_cmray_mask.fits"),
        "trace_data": os.path.join(data_dir, "0722_blue_trace_fits.npz"),
        "trace_arc": os.path.join(data_dir, "0725_blue_trace_fits.npz"),
        "trace_flat": os.path.join(data_dir, "0724_blue_trace_fits.npz"),
        "wavelength": "far red",
        "bad_mask": np.array([22,]),
        "total_masks": 552,
        "mask_groups": 12
    }
    return rectify_args

@pytest.fixture
def rectify_obj(rectify_args) -> Rectify:
    return Rectify(**rectify_args)

def test_optimize_centers_data(rectify_obj: Rectify, data_dir: str, tmp_path: str) -> None:
    rectify_obj.optimize_centers(
        arc_maskdir=os.path.join(data_dir, "0725_blue_mask.fits"),
        output=os.path.join(tmp_path,f"0722_blue_centers.npz"),
        arc_or_data="data",
        fix_sparse=True,
    )

def test_optimize_centers_arc(rectify_obj: Rectify, data_dir: str, tmp_path: str) -> None:
    rectify_obj.optimize_centers(
        arc_maskdir=os.path.join(data_dir, "0725_blue_mask.fits"),
        output=os.path.join(tmp_path,f"0725_blue_centers.npz"),
        arc_or_data="arc",
        fix_sparse=True,
    )

def test_optimize_centers_app(rectify_args: dict, data_dir: str, tmp_path: str) -> None:
    future = optimize_center_app(
        rectify_args,
        arc_maskdir=os.path.join(data_dir, "0725_blue_mask.fits"),
        arc_or_data="arc",
        fix_sparse=True,
        outputs=[File(os.path.join(tmp_path,f"0725_blue_centers.npz")),],
    )
    future.result()

def test_rectify_data(rectify_obj: Rectify, data_dir: str, tmp_path: str) -> None:
    rectify_obj.rectify(
        os.path.join(data_dir,"0722_blue_centers.npz"),
        os.path.join(tmp_path, "0722_blue_rect.npz"),
        arc_or_data="data",
    )

def test_rectify_arc(rectify_obj: Rectify, data_dir: str, tmp_path: str) -> None:
    rectify_obj.rectify(
        os.path.join(data_dir,"0725_blue_centers.npz"),
        os.path.join(tmp_path, "0725_blue_rect.npz"),
        arc_or_data="arc",
    )

def test_rectify_app(rectify_args: dict, data_dir: str, tmp_path: str) -> None:
    rectify_app(
        rectify_args,
        os.path.join(data_dir,"0722_blue_centers.npz"),
        "data",
        outputs=[File(os.path.join(tmp_path, "0722_blue_rect.npz"),)],
    )

def test_calib(rectify_obj: Rectify, data_dir: str, tmp_path: str) -> None:
    rectify_obj.calib(
        os.path.join(data_dir, "0722_blue_rect.npz"),
        os.path.join(data_dir, "0725_blue_rect.npz"),
        os.path.join(tmp_path, "0722_blue_calib.npz"),
        use_sky=True,
    )

def test_calib_app(rectify_args: dict, data_dir: str, tmp_path: str) -> None:
    calib_app(
        rectify_args,
        os.path.join(data_dir, "0722_blue_rect.npz"),
        os.path.join(data_dir, "0725_blue_rect.npz"),
        use_sky=True,
        outputs=[File(os.path.join(tmp_path, "0722_blue_calib.npz"),)]
    )