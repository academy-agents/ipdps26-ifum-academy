from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import glob
import os
from astropy.io import fits
from scipy import ndimage
from astropy.convolution import Gaussian2DKernel, convolve
import re
from parsl.app.app import python_app, join_app
from parsl.data_provider.files import File
from .utils import *

from academy.agent import Agent, action, loop
from typing import List, Dict
import asyncio

@python_app(cache=True)
def load_files(directory, prefix) -> None:
    # find the appropriate files
    files = glob(os.path.join(directory,("*"+prefix+"*.fits")))

    # creates convenient matrix of the files
    ordered_identity = np.array([["blue1","blue2","blue3","blue4"],
                                ["red1","red2","red3","red4"]])
    ordered_files = np.empty((2, 4), dtype="object")

    # assigns files correctly within matrix
    for file in files:
        header = fits.open(file)[0].header
        color,opamp = header["SHOE"],header["OPAMP"]
        i = 0 if color=="B" else 1 if color=="R" else None
        j = 0 if opamp==1 else 1 if opamp==2 else 2 if opamp==3 else 3 if opamp==4 else None
        ordered_files[i,j] = file

    if ordered_identity[ordered_files==None].size != 0:
        print(f"{len(files)} files",flush=True)
        print(f"missing files: {ordered_identity[ordered_files==None]}",flush=True)
        return None
    else:
        return ordered_files

@python_app(cache=True)
def save_file(
        files,
        filename,
        bin_to_2x1=True,
        outputs=()
    ) -> None:
    ordered_data = np.empty((2,4), dtype="object")
    for iy, ix in np.ndindex(files.shape):
        file = files[iy,ix]
        header,data = fits.open(file)[0].header,fits.open(file)[0].data
        x1,x2,y1,y2 = [int(s) for s in re.findall(r'\d+', header["TRIMSEC"])]
        x1 -= 1
        y1 -= 1

        # subtracts the mean of bias x slices from the data
        ordered_data[iy,ix] = data[y1:y2,x1:x2] - np.repeat(np.array([np.mean(data[y1:y2,x2:],axis=1)]).T,data[y1:y2,x1:x2].shape[1],axis=1)
        if bin_to_2x1 and header["BINNING"]=='1x1':
            ordered_data[iy,ix] = ordered_data[iy,ix][:,0::2]+ordered_data[iy,ix][:,1::2]

    # stack images
    total_b = np.vstack((np.hstack((ordered_data[0][3],np.flip(ordered_data[0][2], axis=1))),
                            np.hstack((np.flip(ordered_data[0][0], axis=0),np.flip(ordered_data[0][1], axis=(0,1))))))
    total_r = np.vstack((np.hstack((ordered_data[1][3],np.flip(ordered_data[1][2], axis=1))),
                            np.hstack((np.flip(ordered_data[1][0], axis=0),np.flip(ordered_data[1][1], axis=(0,1))))))

    # save images as fits files
    fits.writeto(outputs[0], data=total_b, overwrite=True)
    fits.writeto(outputs[1], data=total_r, overwrite=True)

@python_app
def calculate_intenal_noise(flat_file_with_bias: str) -> np.ndarray:
    with fits.open(flat_file_with_bias) as flat_dataf:
        flat_data = flat_dataf[0].data

    # THINK is there a better way to compute internal noise?
    median_image = ndimage.median_filter(flat_data,size=(1,9))
    internal_noise = flat_data/median_image
    internal_noise = (internal_noise)/(np.percentile(internal_noise,99)-np.min(internal_noise))
    return internal_noise

@python_app
def combined_bias_app(bias_file: str, internal_noise: np.ndarray, outputs=()):
    with fits.open(bias_file) as datahdu:
        data = datahdu[0].data

    denoised = data/internal_noise
    fits.writeto(filename=os.path.join(os.path.abspath("out"),file+self.color+".fits"), data=denoised, overwrite=True)

    return None