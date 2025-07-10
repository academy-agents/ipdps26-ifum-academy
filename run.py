import numpy as np
import parsl
import ifum
import config


if __name__ == "__main__":
    ######### INPUTS ########

    # directory containing unprocessed files
    # directory = "C:\\Users\\daniel\\OneDrive - The University of Chicago\\Documents\\cool lamps\\summer_24\\IFUM data\\ut20240210\\"
    directory = "/home/babnigg/globus/ifum_run/out"

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



    ######### WORKFLOW #########
    for datafilename, arcfilename, flatfilename in zip(data_filenames, arc_filenames, flat_filenames):
        info = (datafilename,arcfilename,flatfilename,wavelength,bad_masks,total_masks,mask_groups)
        spectra = ifum.get_spectra(sig_mult,bins,color="b",info=info)
        spectra = ifum.get_spectra(sig_mult,bins,color="r",info=info)
