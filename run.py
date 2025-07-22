import numpy as np
import time
from datetime import timedelta
import os
import concurrent.futures
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
    #  only include string in file that includes all files from single exposure
    #  the files should be as close to chronological as possible
    data_filenames = ["0721","0722","0723","0727","0728","0729","0736","0737","0738"]
    arc_filenames = ["0725","0725","0725","0733","0733","0733","0740","0740","0740"]
    flat_filenames = ["0724","0724","0724","0734","0734","0734","0739","0739","0739"]

    mode = "STD" # LR, STD, HR
    wavelength = "far red" # far red, blue

    # bad masks for each fiber shoe (on scale 1-x)
    bad_blues = [23]
    bad_reds = []

    bin_to_2x1 = True # guarentees optimal binning
    sig_mult = 1.5 # maximum dispersion std from gaussian-based fit

    # stars to use in WCS (list RA,Dec)
    #  all stars should be present in at least some dithers
    wcs_stars = [[74.8322, -58.6579],
                 [74.8305, -58.6603],
                 [74.8308, -58.6587],
                 [74.8254, -58.6572],
                 [74.8237, -58.6594]]

    # preparing inputs as function inputs
    bad_masks,wcs_stars,total_masks,mask_groups,hex_dims,bins = (
        config.prepare_inputs(
            bad_blues,
            bad_reds,
            wcs_stars,
            mode,
            wavelength
        )
    )



    ######### CONFIG #########

    config = config.midway_config()
    parsl.load(config)
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | parsl config loaded", flush=True)



    ######### WORKFLOW #########

    # create output directory if it doesn't exist already
    os.makedirs(os.path.abspath("out"), exist_ok=True)


    # 1-STITCH: stitch and create file
    stitch_files = data_filenames+list(set(arc_filenames))+list(set(flat_filenames))
    stitch_apps = []
    for file in stitch_files:
        stitch_args = {
            "directory": directory,
            "filename": file,
            "files": None,
            "color": None,
            "datafilename": None,
            "arcfilename": None,
            "flatfilename": None
        }
        stitch_apps.append(ifum.load_and_save_app(stitch_args,bin_to_2x1))
    concurrent.futures.wait(stitch_apps)
    # for future in stitch_apps:
    #     future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | stitched files saved", flush=True)


    # 2-BIAS: solve and save the bias
    bias_files = np.unique(flat_filenames)
    bias_apps = []
    for flatfilename in bias_files:
        stitch_deps = [next(
            f for f, fname in zip(stitch_apps, stitch_files)
            if fname == flatfilename
        )]

        stitch_args = {
            "directory": directory,
            "filename": None,
            "files": None,
            "color": "b",
            "datafilename": None,
            "arcfilename": None,
            "flatfilename": flatfilename
        }

        indexes = [i for i, value in enumerate(flat_filenames) if value == flatfilename]
        relevant_files = np.concatenate(
            (np.unique(np.array(data_filenames)[indexes]),
             np.unique(np.array(arc_filenames)[indexes]),
             np.unique(np.array(flat_filenames)[indexes]))
        )
        print(relevant_files,flush=True)

        for file in relevant_files:
            stitch_dep = next(
                f for f, fname in zip(stitch_apps, stitch_files)
                if fname == file
            )
            stitch_deps.append(stitch_dep)


        stitch_deps = list(dict.fromkeys(stitch_deps))
        print(stitch_deps, flush=True)

        bias_apps.append(ifum.combined_bias_app(
            dep_futures = stitch_deps,
            stitch_args = stitch_args,
            files = relevant_files
        ))

        stitch_args["color"] = "r"
        bias_apps.append(ifum.combined_bias_app(
            dep_futures = stitch_deps,
            stitch_args = stitch_args,
            files = relevant_files
        ))


    # # 2.1-BIAS: solve for bias
    # bias_files = np.unique(flat_filenames)
    # bias_apps = []
    # for flatfilename in bias_files:
    #     stitch_dep = next(
    #         f for f, fname in zip(stitch_apps, stitch_files)
    #         if fname == flatfilename
    #     )

    #     stitch_args = {
    #         "directory": directory,
    #         "filename": None,
    #         "files": None,
    #         "color": "b",
    #         "datafilename": None,
    #         "arcfilename": None,
    #         "flatfilename": flatfilename
    #     }
    #     internal_noise = ifum.bias_sub_app(
    #         dep_futures = [stitch_dep],
    #         stitch_args = stitch_args.copy()
    #     )
    #     bias_apps.append(internal_noise)

    #     stitch_args["color"] = "r"
    #     internal_noise = ifum.bias_sub_app(
    #         dep_futures = [stitch_dep],
    #         stitch_args = stitch_args.copy()
    #     )
    #     bias_apps.append(internal_noise)
    # print(f"{str(timedelta(seconds=int(time.time()-start)))} | internal bias started", flush=True)

    # # 2.2-BIAS: write the bias values
    # bias_files = np.unique(flat_filenames)
    # bias_apps_s = []
    # for f_idx,flatfilename in enumerate(bias_files):
    #     stitch_dep_flat = next(
    #         f for f, fname in zip(stitch_apps, stitch_files)
    #         if fname == flatfilename
    #     )

    #     stitch_args = {
    #         "directory": directory,
    #         "filename": None,
    #         "files": None,
    #         "color": "b",
    #         "datafilename": None,
    #         "arcfilename": None,
    #         "flatfilename": flatfilename
    #     }

    #     indexes = [i for i, value in enumerate(flat_filenames) if value == flatfilename]
    #     relevant_files = np.concatenate(
    #         (np.unique(np.array(data_filenames)[indexes]),
    #          np.unique(np.array(arc_filenames)[indexes]),
    #          np.unique(np.array(flat_filenames)[indexes]))
    #     )

    #     # internal_noise = ifum.bias_sub_app(
    #     #     dep_futures = [],
    #     #     stitch_args = stitch_args.copy()
    #     # )
    #     internal_noise = bias_apps[f_idx*2].result()
    #     for file in relevant_files:
    #         stitch_args["filename"] = file
    #         # dep_futures = [f for f, fname in zip(stitch_apps, stitch_files) if fname == file]
    #         stitch_dep = next(
    #             f for f, fname in zip(stitch_apps, stitch_files)
    #             if fname == file
    #         )
    #         bias_apps_s.append(ifum.noise_app(
    #             dep_futures = [stitch_dep],
    #             stitch_args = stitch_args.copy(),
    #             internal_noise = internal_noise
    #         ))

    #     stitch_args["color"] = "r"
        
    #     # dep_futures = [f for f, fname in zip(stitch_apps, stitch_files) if fname == flatfilename]
    #     # internal_noise = ifum.bias_sub_app(
    #     #     dep_futures = [],
    #     #     stitch_args = stitch_args.copy()
    #     # )
    #     internal_noise = bias_apps[f_idx*2+1].result()
    #     for file in relevant_files:
    #         stitch_args["filename"] = file
    #         # dep_futures = [f for f, fname in zip(stitch_apps, stitch_files) if fname == file]
    #         stitch_dep = next(
    #             f for f, fname in zip(stitch_apps, stitch_files)
    #             if fname == file
    #         )
    #         bias_apps_s.append(ifum.noise_app(
    #             dep_futures = [stitch_dep],
    #             stitch_args = stitch_args.copy(),
    #             internal_noise = internal_noise
    #         ))

    print(f"{str(timedelta(seconds=int(time.time()-start)))} | internal bias started", flush=True)
    concurrent.futures.wait(bias_apps,return_when="ALL_COMPLETED")
    for future in bias_apps:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | internal bias solved", flush=True)


    # 3-CMRAY: create cosmic ray masks
    cmray_apps = []
    for datafilename in data_filenames:

        # bias_deps = []
        # for flatfilename in np.unique(flat_filenames):
        #     indexes = [i for i, (df, ff) in enumerate(zip(data_filenames, flat_filenames))
        #                if df == datafilename and ff == flatfilename]
        #     if indexes:
        #         flat_idx = list(np.unique(flat_filenames)).index(flatfilename)
        #         bias_deps.append(bias_apps[flat_idx*2])
        #         bias_deps.append(bias_apps[flat_idx*2+1])

        stitch_args = {
            "directory": directory,
            "filename": None,
            "files": None,
            "color": "b",
            "datafilename": datafilename,
            "arcfilename": None,
            "flatfilename": None
        }
        cmray_apps.append(ifum.cmray_mask_app(
            dep_futures = [],
            stitch_args = stitch_args,
            data_filenames = data_filenames
        ))

        stitch_args["color"] = "r"
        cmray_apps.append(ifum.cmray_mask_app(
            dep_futures = [],
            stitch_args = stitch_args,
            data_filenames = data_filenames
        ))
    # i do not need cmray masks done now! can start next step until rectify
    for future in cmray_apps:
        future.result()
    print(f"{str(timedelta(seconds=int(time.time()-start)))} | cosmic ray masks processing", flush=True)


    sys.exit(1)

    # concurrent futures wait python docs / as_completed

    # 4-FLATMASK: using flat field, first guess then complex guess
    # need to still parallelize better on this part!
    flat_apps = []
    for flatfilename in np.unique(flat_filenames):

        bias_deps = [f for f in bias_apps 
                     if any(fn == flatfilename for fn in flat_filenames)]

        mask_args = {
            "color": "b",
            "flatfilename": flatfilename,
            "bad_masks": bad_masks,
            "total_masks": total_masks,
            "mask_groups": mask_groups
        }
        flat_apps.append(ifum.flat_mask_app(
            dep_futures = bias_deps,
            mask_args = mask_args
        ))
        mask_args["color"] = "r"
        flat_apps.append(ifum.flat_mask_app(
            dep_futures = bias_deps,
            mask_args = mask_args
        ))
        print(f"4submitted: {flatfilename}", flush=True)
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
