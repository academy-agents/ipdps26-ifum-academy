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

    # 1-STITCH: stitch and create file
    saved_files = []
    for file in (data_filenames+list(set(arc_filenames))+list(set(flat_filenames))):
        stitch_args = {
            "directory": directory,
            "filename": file,
            "files": None,
            "color": None,
            "datafilename": None,
            "arcfilename": None,
            "flatfilename": None
        }
        saved_files.append(ifum.load_and_save_app(stitch_args,bin_to_2x1))
    for future in saved_files:
        # try:
        future.result()
        # except Exception as e:
        #     print(f"1-STITCH error: {e}", flush=True)
        #     sys.exit(1)
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | stitched files saved", flush=True)

    # 2-BIAS: solve for bias
    # (NOT fully parallel yet, weird file error)
    bias_files = []
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        stitch_args = {
            "directory": directory,
            "filename": None,
            "files": None,
            "color": "b",
            "datafilename": datafilename,
            "arcfilename": arcfilename,
            "flatfilename": flatfilename
        }
        bias_files.append(ifum.bias_sub_app(stitch_args))
        bias_files[-1].result()
        stitch_args["color"] = "r"
        bias_files.append(ifum.bias_sub_app(stitch_args))
        bias_files[-1].result()
    for future in bias_files:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | internal bias solved", flush=True)

    # 3-CMRAY: create cosmic ray masks
    cmray_masks = []
    for datafilename in data_filenames:
        stitch_args = {
            "directory": directory,
            "filename": None,
            "files": None,
            "color": "b",
            "datafilename": datafilename,
            "arcfilename": None,
            "flatfilename": None
        }
        cmray_masks.append(ifum.cmray_mask_app(stitch_args,data_filenames))
        stitch_args["color"] = "r"
        cmray_masks.append(ifum.cmray_mask_app(stitch_args,data_filenames))
    for future in cmray_masks:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | cosmic ray masks created", flush=True)



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
        # try:
        future.result()
        # except Exception as e:
            # print(f"5-FLATTRACE error: {e}", flush=True)
            # sys.exit(1)
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | flat field based traces saved", flush=True)

    # 6-ARCOPT: optimize flat traces on arc data
    expected_peaks = 25
    optimize = True
    arc_opts = []
    for arcfilename, flatfilename in zip(np.unique(arc_filenames), np.unique(flat_filenames)):
        mask_args = {
            "color": "b",
            "flatfilename": flatfilename,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        arc_opts.append(ifum.optimize_arc_app(mask_args,arcfilename,sig_mult,expected_peaks,optimize))
        mask_args["color"] = "r"        
        arc_opts.append(ifum.optimize_arc_app(mask_args,arcfilename,sig_mult,expected_peaks,optimize))
    for future in arc_opts:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | traces optimized for arc data", flush=True)    

    # 7-DATAOPT: optimize flat traces on on-sky science data, get rotations of spectral features
    expected_peaks = 25
    optimize = True
    data_opts = []
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        mask_args = {
            "color": "b",
            "flatfilename": flatfilename,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        data_opts.append(ifum.optimize_data_app(mask_args,arcfilename,datafilename,sig_mult,expected_peaks,optimize))
        mask_args["color"] = "b"
        data_opts.append(ifum.optimize_data_app(mask_args,arcfilename,datafilename,sig_mult,expected_peaks,optimize))
    for future in data_opts:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | traces optimized for science data", flush=True)    

    # 8-ARCTRACE: create the mask files with the arc trace optimizations
    arc_traces = []
    for arcfilename, flatfilename in zip(np.unique(arc_filenames), np.unique(flat_filenames)):
        mask_args = {
            "color": "b",
            "flatfilename": flatfilename,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        arc_traces.append(ifum.create_mask_app(mask_args,arcfilename,arcfilename,sig_mult))
        mask_args["color"] = "r"
        arc_traces.append(ifum.create_mask_app(mask_args,arcfilename,arcfilename,sig_mult))
    for future in arc_traces:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | arc optimized trace masks saved", flush=True)    

    # 9-DATATRACE: create the mask files with the data trace optmizations
    copy = False
    data_traces = []
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        mask_args = {
            "color": "b",
            "flatfilename": flatfilename,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        data_traces.append(ifum.create_mask_app(mask_args,datafilename,arcfilename,sig_mult,copy))
        mask_args["color"] = "r"
        data_traces.append(ifum.create_mask_app(mask_args,datafilename,arcfilename,sig_mult,copy))
    for future in data_traces:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | data optimized trace masks saved", flush=True)    



    # 10-CENTEROPTDATA: optimize the precalculated centers for data files, filling in sparse areas
    rect_opt_data = []
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        rectify_args = {
            "color": "b",
            "datafilename": datafilename,
            "arcfilename": arcfilename,
            "flatfilename": flatfilename,
            "wavelength": wavelength,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        rect_opt_data.append(ifum.optimize_center_app(rectify_args,"data",fix_sparse=True))
        rectify_args["color"] = "r"        
        rect_opt_data.append(ifum.optimize_center_app(rectify_args,"data",fix_sparse=True))
    for future in rect_opt_data:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | fixed sparse data centers for rectification", flush=True)    

    # 11-CENTEROPTARC: optimize the precalculated centers for arc files, filling in sparse areas
    rect_opt_arc = []
    for arcfilename, flatfilename in zip(np.unique(arc_filenames), np.unique(flat_filenames)):
        rectify_args = {
            "color": "b",
            "datafilename": "NA",
            "arcfilename": arcfilename,
            "flatfilename": flatfilename,
            "wavelength": wavelength,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        rect_opt_arc.append(ifum.optimize_center_app(rectify_args,"arc",fix_sparse=True))
        rectify_args["color"] = "r"        
        rect_opt_arc.append(ifum.optimize_center_app(rectify_args,"arc",fix_sparse=True))
    for future in rect_opt_arc:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | fixed sparse arc centers for rectification", flush=True)    

    # 12-RECTDATA: rectify the spectra using the data file
    rect_data = []
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        rectify_args = {
            "color": "b",
            "datafilename": datafilename,
            "arcfilename": arcfilename,
            "flatfilename": flatfilename,
            "wavelength": wavelength,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        rect_data.append(ifum.rectify_app(rectify_args,arc_or_data="data"))
        rectify_args["color"] = "r"
        rect_data.append(ifum.rectify_app(rectify_args,arc_or_data="data"))
    for future in rect_data:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | rectified using the arc data", flush=True)    

    # 13-RECTARC: rectify the spetcra using the arc file
    rect_arc = []
    for arcfilename, flatfilename in zip(np.unique(arc_filenames), np.unique(flat_filenames)):
        rectify_args = {
            "color": "b",
            "datafilename": "NA",
            "arcfilename": arcfilename,
            "flatfilename": flatfilename,
            "wavelength": wavelength,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        rect_arc.append(ifum.rectify_app(rectify_args,arc_or_data="arc"))
        rectify_args["color"] = "r"
        rect_arc.append(ifum.rectify_app(rectify_args,arc_or_data="arc"))
    for future in rect_arc:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | rectified using the on-sky science data", flush=True)    



    # 14-CALIB: using the rectifications and the calibration information, calibrate the spectra
    calibs = []
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        rectify_args = {
            "color": "b",
            "datafilename": datafilename,
            "arcfilename": arcfilename,
            "flatfilename": flatfilename,
            "wavelength": wavelength,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        calibs.append(ifum.calib_app(use_sky=True))
        rectify_args["color"] = "r"
        calibs.append(ifum.calib_app(use_sky=True))
    for future in calibs:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | calibrations done using rectified spectra", flush=True)    



    # 15-SPECBIN: using built context, produce the final calibrated spectra using flux binning per pixel
    # (need to redo with class integration)
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        info = (datafilename,arcfilename,flatfilename,wavelength,bad_masks,total_masks,mask_groups)
        spectra = ifum.get_spectra(sig_mult,bins,color="b",info=info)
        spectra = ifum.get_spectra(sig_mult,bins,color="r",info=info)

        print(f"{str(timedelta(seconds=int(time.time()-start)))} | {datafilename} processed", flush=True)
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | flux-binned spectra produced", flush=True)    



    # other metrics? 
    print(f"total runtime: {str(timedelta(seconds=int(time.time()-start)))}", flush=True)
