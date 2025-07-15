import numpy as np
import time
from datetime import timedelta
import os
import sys

import parsl
import ifum
import config


if __name__ == "__main__":
    start = time.time()
    ######### INPUTS ########

    # directory containing unprocessed files
    directory = "/home/babnigg/globus/IFU-M/in/ut20240210/"

    # all files included in a single stack, repeat where necessary
    # only include string in file that includes all files from single exposure
    data_filenames = ["0721","0722","0723","0727","0728","0729","0736","0737","0738"]
    arc_filenames = ["0725","0725","0725","0733","0733","0733","0740","0740","0740"]
    flat_filenames = ["0724","0724","0724","0734","0734","0734","0739","0739","0739"]

    # mode LR,STD,HR
    mode = "STD"

    # far red vs blue
    wavelength = "far red"

    # bad masks (on scale 1-x)
    bad_blues = [23]
    bad_reds = []

    # stars to use in WCS (list RA,Dec)
    # all stars should be present in at least some dithers
    wcs_stars = [[74.8322, -58.6579],
                [74.8305, -58.6603],
                [74.8308, -58.6587],
                [74.8254, -58.6572],
                [74.8237, -58.6594]]

    # sometimes not already binned; this bin allows for proper gaussians to be fit
    bin_to_2x1 = True

    # value that is used to calculate maximum dispersion (from previous steps!)
    sig_mult = 1.5

    # preparing inputs as function inputs
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



    ######### CONFIG #########

    config = config.midway_config()
    parsl.load(config)
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | parsl config loaded", flush=True)


    ######### WORKFLOW #########
    # out directory where all files are stored
    os.makedirs(os.path.abspath("out"), exist_ok=True)

    # # 1-STITCH: stitch and create file
    # saved_files = []
    # for file in (data_filenames+list(set(arc_filenames))+list(set(flat_filenames))):
    #     stitch_args = {
    #         "directory": directory,
    #         "filename": file,
    #         "files": None,
    #         "color": None,
    #         "datafilename": None,
    #         "arcfilename": None,
    #         "flatfilename": None
    #     }
    #     saved_files.append(ifum.load_and_save_app(stitch_args,bin_to_2x1))
    # for future in saved_files:
    #     try:
    #         future.result()
    #     except Exception as e:
    #         print(f"1-STITCH error: {e}", flush=True)
    #         sys.exit(1)
    # print(f"{str(timedelta(seconds=int(time.time()-start)))} | stitched files saved", flush=True)

    # # 2-BIAS: solve for bias
    # # (NOT fully parallel yet, file error)
    # bias_files = []
    # for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
    #     stitch_args = {
    #         "directory": directory,
    #         "filename": None,
    #         "files": None,
    #         "color": "b",
    #         "datafilename": datafilename,
    #         "arcfilename": arcfilename,
    #         "flatfilename": flatfilename
    #     }
    #     bias_files.append(ifum.bias_sub_app(stitch_args))
    #     bias_files[-1].result()
    #     stitch_args["color"] = "r"
    #     bias_files.append(ifum.bias_sub_app(stitch_args))
    #     bias_files[-1].result()
    # # for future in bias_files:
    # #     try:
    # #         future.result()
    # #     except Exception as e:
    # #         print(f"2-BIAS error: {e}", flush=True)
    # #         sys.exit(1)
    # print(f"{str(timedelta(seconds=int(time.time()-start)))} | internal bias solved", flush=True)

    # # 3-CMRAY: create cosmic ray masks
    # cmray_masks = []
    # for datafilename in data_filenames:
    #     stitch_args = {
    #         "directory": directory,
    #         "filename": None,
    #         "files": None,
    #         "color": "b",
    #         "datafilename": datafilename,
    #         "arcfilename": None,
    #         "flatfilename": None
    #     }
    #     cmray_masks.append(ifum.cmray_mask_app(stitch_args,data_filenames))
    #     stitch_args["color"] = "r"
    #     cmray_masks.append(ifum.cmray_mask_app(stitch_args,data_filenames))
    # for future in cmray_masks:
    #     try:
    #         future.result()
    #     except Exception as e:
    #         print(f"3-CMRAY error: {e}", flush=True)
    #         sys.exit(1)
    # print(f"{str(timedelta(seconds=int(time.time()-start)))} | cosmic ray masks created", flush=True)



    # 4-FLATMASK: using flat field, first guess then complex guess
    # need to still parallelize better on this part!
    flat_masks = []
    for flatfilename in np.unique(flat_filenames):
        mask_args = {
            "color": "b",
            "flatfilename": flatfilename,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        flat_masks.append(ifum.flat_mask_app(mask_args))
        mask_args["color"] = "r"
        flat_masks.append(ifum.flat_mask_app(mask_args))
        print(f"4submitted: {flatfilename}", flush=True)
    for future in flat_masks:
        # try:
        future.result()
        # except Exception as e:
        #     print(f"4-FLATMASK error: {e}", flush=True)
        #     sys.exit(1)
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | flat field based traces optimized", flush=True)

    # 5-FLATTRACE: use flat field solution, along with specified parameters, to create masks
    center_deg = 5
    sigma_deg = 3
    sig_mult = 1.5
    flat_traces = []
    for flatfilename in np.unique(flat_filenames):
        mask_args = {
            "color": "b",
            "flatfilename": flatfilename,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        flat_traces.append(ifum.create_flatmask_app(mask_args,center_deg,sigma_deg,flat_traces))
        mask_args["color"] = "r"
        flat_traces.append(ifum.create_flatmask_app(mask_args,center_deg,sigma_deg,flat_traces))
        print(f"5submitted: {flatfilename}", flush=True)
    for future in flat_traces:
        try:
            future.result()
        except Exception as e:
            print(f"5-FLATTRACE error: {e}", flush=True)
            sys.exit(1)
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | flat field based traces saved", flush=True)


    # for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
    #     info = (datafilename,arcfilename,flatfilename,wavelength,bad_masks,total_masks,mask_groups)
    #     spectra = ifum.get_spectra(sig_mult,bins,color="b",info=info)
    #     spectra = ifum.get_spectra(sig_mult,bins,color="r",info=info)

    #     print(f"{str(timedelta(seconds=int(time.time()-start)))} | {datafilename} processed", flush=True)
    
    print(f"total runtime: {str(timedelta(seconds=int(time.time()-start)))}", flush=True)
