from concurrent.futures import Future
import os

import numpy as np
from parsl import python_app
from parsl import join_app
from astropy.io import fits

def get_x_from_point_simple(px, trace_poly, rotation_poly) -> np.ndarray:
    if isinstance(rotation_poly, float) and np.isnan(rotation_poly):
        return px[:,1]
    else:
        ys,xs = px[:,0],px[:,1]

        rot_m = np.tan(np.pi/2+np.poly1d(rotation_poly)(xs))
        rot_b = ys-rot_m*xs

        y0 = np.polyval(trace_poly, xs)
        trace_m = np.polyval(np.polyder(trace_poly), xs)
        trace_b = y0-trace_m*xs

        return (trace_b-rot_b)/(rot_m-trace_m)

# potentially use batch computation here as well? 276 is large number
@python_app
def get_spectrum_fluxbins(
    mask,
    bins,
    sigma_traces,
    sig_mult,
    datadir,
    cmraymask,
    trace_data,
    calib_data,
    rand_dots=50
) -> np.ndarray:
    '''
    gets flux-binned spectrum for a given spectra trace
    inputs:
        mask: spectra trace #
        bins: final wavelength bins
        traces: spectral traces as polynomial functions
        rotations: spectral rotations from orthogonal dispersion axis as polynomial functions
        stds: spectral dispersion from trace as polynomial functions
        sig_mult: maximum gaussian sigma from mean where dispersion axis reaches
        x_s: x axis to rectify
        wl_calib: polynomial that transforms rectified x axis to calibrated wavelengths
        data: actual data; image of spectra
        cmray_mask: mask same size as data, indicating cosmic rays
        rand_dots: for monte-carlo sampling
    outputs:
        flux-binned spectra intensities, with wavelength axis of bins
    '''
    import numpy as np
    from astropy.io import fits
    from scipy.ndimage import binary_dilation

    with fits.open(datadir) as dataf, \
         fits.open(cmraymask) as cmrayf, \
         np.load(trace_data) as npztrace, \
         np.load(calib_data) as npzcalib:
        
        data = dataf[0].data
        cmray_mask = cmrayf[0].data
        
        traces = npztrace["traces"]
        rotations = npztrace["rotation_traces"]
        stds = sigma_traces
        x_s = npzcalib["rect_x"]
        wl_calib = npzcalib["wl_calib"]

        # sometimes rotation can't be calculated, if it doesn't do not include in computation
        use_rot = not (isinstance(rotations, float) and np.isnan(rotations))

        # first, compute bin edges from bins
        #  [1,2,3,4,5] -> [0.5,1.5,2.5,3.5,4.5,5.5]
        midpoints = 0.5 * (bins[1:] + bins[:-1])
        bin_edges = np.concatenate(([bins[0]-0.5*(bins[1]-bins[0])],
                                    midpoints,
                                    [bins[-1]+0.5*(bins[1]-bins[0])]))
        
        # compute a bigger mask to include all possible pixels in binning
        #  effectively; take spectral traces, dispersion information, and create large mask
        #  where pixel can possibily include member flux of the given spectrum
        new_mask = np.zeros(cmray_mask.shape)
        for i in range(cmray_mask.shape[1]):
            mask_center_i = np.poly1d(traces[mask])(i)
            mask_sig_i = sig_mult * np.poly1d(stds[mask])(i)
            new_mask[round(mask_center_i-mask_sig_i):round(mask_center_i+mask_sig_i+1),i] = 1
        new_mask = binary_dilation(new_mask,structure=np.ones((3,3))).astype(int)
        pixels = np.argwhere(new_mask==1)
        data[cmray_mask==1] = np.nan

        # compute true x -> wavelengths for all pixels
        if use_rot:
            x_intercepts = get_x_from_point_simple(pixels,
                                                traces[mask],
                                                rotations[mask])
        else:
            x_intercepts = get_x_from_point_simple(pixels,
                                                traces[mask],
                                                np.nan)
        # use rectify to shift x_intercepts for proper calibration
        x_intercepts = np.interp(x_intercepts,np.arange(data.shape[1]),x_s[mask])
        wls = np.poly1d(wl_calib)(x_intercepts)

        # get the average standard devation from spectra trace along mask (0th degree approximation)
        std_avg = np.median(np.poly1d(stds[mask])(np.arange(2048)))
        # precompute centers, upper_bounds, and lower_bounds for all pixels
        centers = np.poly1d(traces[mask])(pixels[:,1])
        upper_bounds = centers+sig_mult*std_avg
        lower_bounds = centers-sig_mult*std_avg
        # compute weights for all pixels based on distance from spectra trace
        weights = np.where(pixels[:,0]>centers,
                        upper_bounds+0.5-pixels[:, 0],
                        pixels[:,0]-(lower_bounds-0.5))
        weights = np.clip(weights, 0, 1)

        # interpolation to find portion of pixel that belongs to wavelength bin
        spectrum = np.zeros(bins.shape)
        vals = data[pixels[:,0],pixels[:,1]]*weights
        # prepare random point samples within a single pixel
        x_points_all = []
        y_points_all = []
        for px in pixels:
            minx, maxx = px[1]-0.5, px[1]+0.5
            miny, maxy = px[0]-0.5, px[0]+0.5
            # generate random points
            x_points = np.random.uniform(minx,maxx,rand_dots)
            y_points = np.random.uniform(miny,maxy,rand_dots)
            x_points_all.append(x_points)
            y_points_all.append(y_points)
        x_points_all = np.concatenate(x_points_all)
        y_points_all = np.concatenate(y_points_all)
        # get x-intercepts for batch
        points_all = np.column_stack((y_points_all,x_points_all))
        if use_rot:
            x_intercepts_all = get_x_from_point_simple(points_all,traces[mask],rotations[mask])
        else:
            x_intercepts_all = get_x_from_point_simple(points_all,traces[mask],np.nan)
        x_intercepts_all = np.interp(x_intercepts_all,np.arange(data.shape[1]),x_s[mask])
        
        # batch wavelength and histogram calculation
        px_wls_all = np.poly1d(wl_calib)(x_intercepts_all)
        start_idx = 0
        for px, val in zip(pixels,vals):
            end_idx = start_idx+rand_dots
            px_wls = px_wls_all[start_idx:end_idx]
            
            counts, _ = np.histogram(px_wls,bins=bin_edges)
            counts = counts/len(px_wls)

            # cast so only counts != 0 turn into nan
            if np.isnan(val):
                counts[counts!=0] = np.nan
                spectrum += counts
            else:
                spectrum += counts*val

            start_idx = end_idx
        # bins without wavelength become nan
        spectrum[spectrum==0] = np.nan

        return spectrum, pixels, wls
    
@python_app
def get_specturm_fluxbins_bad():
    return (np.nan, None, None)

@join_app
def launch_spectrum_fluxbins(
    datadir: str,
    cmraymask:str,
    trace_data: str,
    calib_data: str,
    bad_mask: np.ndarray,
    total_masks: int,
    sig_mult: float,
    bins: np.ndarray,
    use_global=False,
) -> list[float | Future[float]]:    
    npzdata = np.load(trace_data)
    sigma_traces = npzdata["init_traces_sigma"]
    if use_global:
        mean_sigma_trace = np.nanmean(sigma_traces[:,-1:],axis=0)
        sigma_traces = np.repeat(mean_sigma_trace,sigma_traces.shape[0]).reshape(sigma_traces.shape[0],1)

    results = []
    for mask in range(total_masks//2):
        if mask not in bad_mask:
            results.append(
                get_spectrum_fluxbins(
                    mask,
                    bins,
                    sigma_traces,
                    sig_mult,
                    datadir,
                    cmraymask,
                    trace_data,
                    calib_data,
                ))
        else:
            results.append(get_specturm_fluxbins_bad())
    
    return results

@python_app
def collect_spectra(datadir, total_masks, bins, bad_mask, specturm_bins, outputs=()):
    spectra = np.empty((total_masks//2,bins.shape[0]))
    data = fits.open(datadir)[0].data
    wl_mask = np.zeros(data.shape)
    for mask in range(total_masks//2):
        if mask not in bad_mask:
            spectra[mask],px,wl = specturm_bins[mask]
            wl_mask[px[:,0],px[:,1]] = wl
        else:
            spectra[mask] = np.nan
    
    save_dict = {}
    save_dict["wl_bins"] = bins
    save_dict["flux_bins"] = spectra
    np.savez(outputs[0], **save_dict)