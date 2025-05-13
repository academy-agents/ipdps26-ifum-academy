import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
from skimage.registration import phase_cross_correlation

def gauss_background(x,*var):
    H,H1,a,x0,sigma = var
    # _ = args
    return H+H1*x+a*np.exp(-(x-x0)**2/(2*sigma**2))

def gauss_2d(x,y,x0,y0,A,a,b,c,H):
    return A*np.exp(-(a*(x-x0)**2+2*b*(x-x0)*(y-y0)+c*(y-y0)**2))+H

def minimize_gauss_2d(var,args):
    x_,y_,A,a,b,c,H = var
    x0,y0,x_off,y_off,data,plot = args

    x = np.arange(x0-x_off,x0+x_off+0.9,1.)
    y = np.arange(y0-y_off,y0+y_off+0.9,1.)
    xv,yv = np.meshgrid(x,y)
    
    gauss_points = gauss_2d(xv,yv,x0+x_,y0+y_,A,a,b,c,H)
    
    image_points = (data-np.nanmin(data))/(np.nanmax(data)-np.nanmin(data))

    res = np.nanmean((image_points-gauss_points)**2)

    if plot:
        vmin,vmax = np.nanmin(image_points),np.nanmax(image_points)
        plt.figure(dpi=100).set_facecolor("lightgray")
        plt.subplot(1,3,1)
        plt.title("gaussian")
        plt.imshow(gauss_points,origin="lower",cmap="magma",vmin=vmin,vmax=vmax)
        plt.colorbar(orientation="horizontal", pad=0.0)
        plt.axis("equal")
        plt.axis("off")
        plt.subplot(1,3,2)
        plt.title("data")
        plt.imshow(image_points,origin="lower",cmap="magma",vmin=vmin,vmax=vmax)
        plt.colorbar(orientation="horizontal", pad=0.0)
        plt.axis("equal")
        plt.axis("off")
        plt.subplot(1,3,3)
        plt.title("residual\nSSE: "+f"{res:.3f}")
        plt.imshow(image_points-gauss_points,origin="lower",cmap="magma",vmin=-0.25,vmax=0.25)
        plt.colorbar(orientation="horizontal", pad=0.0)
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    return res

def get_x_from_point(px, trace_poly, rotation_poly) -> float:
    y,x = px

    rot_m = np.tan(np.pi/2+np.poly1d(rotation_poly)(x))
    rot_b = y-rot_m*x
    coeffs_res = trace_poly - np.array(list(np.zeros(len(trace_poly)-2))+[rot_m]+[rot_b])
    x_intercept = np.real(np.roots(coeffs_res))
    x_intercept = x_intercept[(np.abs(x_intercept - x)).argmin()]

    return x_intercept

def get_x_from_point_simple(px, trace_poly, rotation_poly) -> float:
    if np.isnan(rotation_poly):
        return px[:,1]
    else:
        ys,xs = px[:,0],px[:,1]

        rot_m = np.tan(np.pi/2+np.poly1d(rotation_poly)(xs))
        rot_b = ys-rot_m*xs

        y0 = np.polyval(trace_poly, xs)
        trace_m = np.polyval(np.polyder(trace_poly), xs)
        trace_b = y0-trace_m*xs

        return (trace_b-rot_b)/(rot_m-trace_m)

def get_spectrum_fluxbins(mask, bins, traces, rotations, stds, sig_mult, x_s, wl_calib, data, cmray_mask, rand_dots=50) -> np.ndarray:
    if np.isnan(rotations):
        use_rot = False
    else:
        use_rot = True
    # first, compute bin edges from bins
    midpoints = 0.5 * (bins[1:] + bins[:-1])
    bin_edges = np.concatenate(([bins[0]-0.5*(bins[1]-bins[0])],
                                midpoints,
                                [bins[-1]+0.5*(bins[1]-bins[0])]))
    
    # compute a bigger mask to include all possible pixels in binning
    new_mask = np.zeros(cmray_mask.shape)
    for i in range(cmray_mask.shape[1]):
        mask_center_i = np.poly1d(traces[mask])(i)
        mask_sig_i = sig_mult * np.poly1d(stds[mask])(i)
        new_mask[round(mask_center_i-mask_sig_i):round(mask_center_i+mask_sig_i+1),i] = 1
    new_mask = scipy.ndimage.binary_dilation(new_mask,structure=np.ones((3,3))).astype(int)
    pixels = np.argwhere(new_mask==1)
    data[cmray_mask==1] = np.nan

    # compute x_intercepts and wavelengths for all pixels
    if use_rot:
        x_intercepts = get_x_from_point_simple(pixels, traces[mask], rotations[mask])
    else:
        x_intercepts = get_x_from_point_simple(pixels, traces[mask], np.nan)
    # use rectify to shift x_intercepts for proper calibration
    x_intercepts = np.interp(x_intercepts,np.arange(data.shape[1]),x_s[mask])
    wls = np.poly1d(wl_calib)(x_intercepts)

    # get the average standard devation from spectra trace along mask
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

    # weights_ = np.empty(pixels.shape[0])
    # for i,px in enumerate(pixels):
    #     # wls[i] = np.poly1d(wl_calib)(get_x_from_point(px, traces[mask], rotations[mask]))
    #     center = np.poly1d(traces[mask])(px[1])
    #     upper_bound,lower_bound = center+sig_mult*std_avg,center-sig_mult*std_avg
    #     # upper_bound,lower_bound = center+sig_mult*np.poly1d(stds[mask])(px[1]),center-sig_mult*np.poly1d(stds[mask])(px[1])
    #     if px[0]>center:
    #         weights_[i] = upper_bound+0.5-px[0]
    #     else:
    #         weights_[i] = px[0]-(lower_bound-0.5)
    # weights_[weights_<=0] = 0
    # weights_[weights_>=1] = 1



    # ATTEMPT to do interpolation
    spectrum = np.zeros(bins.shape)
    vals = data[pixels[:,0],pixels[:,1]]*weights
    # prepare random point samples
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

    # plt.figure()
    # plt.plot(bins,spectrum)
    # plt.show()

    return spectrum, pixels, wls

    # # ATTEMPT to do interpolation
    # spectrum = np.zeros(bins.shape)
    # vals = data[pixels[:,0],pixels[:,1]]*weights
    # for wl, val, px in zip(wls, vals, pixels):
    #     # drop rand_dots dots in pixel
    #     minx,maxx,miny,maxy = px[1]-0.5,px[1]+0.5,px[0]-0.5,px[0]+0.5
    #     x_points = np.random.uniform(minx,maxx,rand_dots)
    #     y_points = np.random.uniform(miny,maxy,rand_dots)
    #     points = np.column_stack((x_points, y_points))
    #     # gets wavelengths for those dots
    #     x_intercepts = get_x_from_point_simple(points, traces[mask], rotations[mask])
    #     x_intercepts = np.interp(x_intercepts,np.arange(data.shape[1]),x_s[mask])
    #     px_wls = np.poly1d(wl_calib)(x_intercepts)
    #     # print(np.mean(px_wls),np.std(px_wls),np.min(px_wls),np.max(px_wls))
    #     # gets approximate proportions for each wavelength bin
    #     counts, _ = np.histogram(px_wls, bins=bin_edges)
    #     counts = counts/len(px_wls)        

    #     spectrum += counts*val

    # return spectrum, pixels, wls

    # OLD digitized version (check bin_edges vs bins?)
    # vals = data[pixels[:,0],pixels[:,1]]*weights
    # spectrum = np.zeros(bins.shape)
    # digitized = np.digitize(wls,bin_edges)
    # print(digitized)

    # for i in range(len(vals)):
    #     spectrum[digitized[i]] += vals[i]
    # spectrum[spectrum==0] = np.nan

    # return spectrum,pixels,wls




def get_spectrum_simple(data,data_m,m,data_c=None):
    if data_c is None:
        data_c = np.zeros(data.shape)

    cut_i = np.copy(data)
    cut_i[(data_m!=m)|(data_c==1)] = np.nan
    spectra = np.nanmean(cut_i,axis=0)

    return spectra

def get_spectrum_simple_withnan(data,data_m,m,data_c=None):
    if data_c is None:
        data_c = np.zeros(data.shape)

    cut_i = np.copy(data)
    cut_i[(data_m!=m)] = np.nan
    spectra = np.nanmean(cut_i,axis=0)

    cmray_mask = np.any((data_m==m)&(data_c==1),axis=0)
    spectra[cmray_mask] = np.nan

    return spectra

def get_lag(a,ref_a):
    corr = np.correlate(np.nan_to_num(a),np.nan_to_num(ref_a),mode="full")
    return np.argmax(corr) - (ref_a.size - 1)

def get_lag_2d(im1,im2):
    # mask = np.random.choice([True, False], size=im1.size, p=[0.1, 0.9]).reshape(im1.shape)
    # im1[mask] = np.nan
    # im2[mask] = np.nan

    # plt.imshow(im1,origin="lower",cmap="viridis")
    # plt.show()
    # plt.imshow(im2,origin="lower",cmap="magma")
    # plt.show()
    # plt.imshow(scipy.ndimage.gaussian_filter(im2,sigma=5),origin="lower",cmap="viridis")
    # plt.show()

    # im1 = scipy.ndimage.gaussian_filter(im1,sigma=1)
    # im2 = scipy.ndimage.gaussian_filter(im2,sigma=1)

    # mean filter
    # im1 = scipy.ndimage.generic_filter(im1, np.mean, size=3)
    # im2 = scipy.ndimage.generic_filter(im2, np.mean, size=3)

    shift, _, _ = phase_cross_correlation(im1, im2, upsample_factor=100)

    return shift[0], shift[1]

def get_peak_center(peak_area,dneg,dpos,x,a,lag):
    return (peak_area[0]-dneg+1)+np.argmax(a[((x-lag)>(peak_area[0]-dneg))&((x-lag)<(peak_area[-1]+dpos+1))])+lag

def gauss(x,H,a,x0,sigma):
    return H+a*np.exp(-(x-x0)**2/(2*sigma**2))

def double_linear_func(x0,args):
        a,b0,b1 = x0
        x = args
        y0 = a*x[::2]+b0
        y1 = a*x[1::2]+b1

        y = np.empty(len(x))
        y[np.arange(len(x))[::2]] = y0
        y[np.arange(len(x))[1::2]] = y1
        
        return y

def miniminize_double_linear_func(x0,args):
    x,y = args

    return np.mean(np.sum((double_linear_func(x0,x)-y)**2))

def minimize_poly_dist(var,args):
    # 2nd deg poly in var
    xs,ys,flat_poly = args

    new_poly = np.polymul(flat_poly,var)

    return np.nanmean((ys-np.poly1d(new_poly)(xs))**2)

def normalize(arr):
    arr = np.array(arr)
    return (arr-np.nanmin(arr))/(np.nanmax(arr)-np.nanmin(arr))

def standardize(arr):
    arr = np.array(arr)
    return (arr-np.nanmean(arr))/(np.nanstd(arr))

def air_to_vacuum(wavelengths):
        s = 1e4/(wavelengths)
        n = 1 + 0.00008336624212083+0.02408926869968/(130.1065924522-s**2)+0.0001599740894897/(38.92568793293-s**2)
        return wavelengths*n



def xy_to_radec(var,args):
    cd00,cd01,cd10,cd11 = var
    xys,refxys,radecs = args

    trans_x,trans_y = np.mean(refxys,axis=0)
    refra,refdec = np.mean(radecs,axis=0)
    x,y = xys.T
    ra,dec = radecs.T

    x_ = x-trans_x
    y_ = y-trans_y

    x__ = cd00*x_+cd01*y_
    y__ = cd10*x_+cd11*y_

    calcras = []
    calcdecs = []
    for i in np.arange(x__.shape[0]):
        if (np.angle(complex(-y__[i],x__[i]),True)<0):
            phi = np.angle(complex(-y__[i],x__[i]),True)+360.
        else:
            phi = np.angle(complex(-y__[i],x__[i]),True)
                
        theta = np.degrees(np.arctan(180./(np.pi*(np.sqrt(x__[i]*x__[i]+y__[i]*y__[i])))))
    
        lonpole = 180.
        ang1forra = np.sin(np.radians(theta))*np.cos(np.radians(refdec))-np.cos(np.radians(theta))*np.sin(np.radians(refdec))*np.cos(np.radians(phi-lonpole))
        ang2forra = -1.*np.cos(np.radians(theta))*np.sin(np.radians(phi-lonpole))
        if ((np.angle(complex(ang1forra,ang2forra),True)+refra)<0):
            calcra = refra+np.angle(complex(ang1forra,ang2forra),True)+360.
        else:
            calcra = refra+np.angle(complex(ang1forra,ang2forra),True)
        
        calcdec = np.degrees(np.arcsin(np.sin(np.radians(theta))*np.sin(np.radians(refdec))+np.cos(np.radians(theta))*np.cos(np.radians(refdec))*np.cos(np.radians(phi-lonpole))))

        calcras.append(calcra)
        calcdecs.append(calcdec)
        
    calcras = np.array(calcras)
    calcdecs = np.array(calcdecs)

    return calcras,calcdecs

def xy_to_radec_minimize(var,args):
    _,_,radecs = args
    ra,dec = radecs.T
    
    calcras,calcdecs = xy_to_radec(var,args)

    return np.mean((np.concatenate((calcras,calcdecs))-np.concatenate((ra,dec)))**2)



def sigma_clip(x,y,deg,weight,sigma=1,iter=10,include=0.25):
    fit = np.polyfit(x,y,deg,w=weight)
    for j in range(iter):
        mult = 1
        polymask = y<(np.poly1d(fit)(x)+mult*sigma*np.nanstd(np.poly1d(fit)(x)))
        polymask &= y>(np.poly1d(fit)(x)-mult*sigma*np.nanstd(np.poly1d(fit)(x)))
        if np.sum(polymask)/len(x) < include:
            polymask = np.array([True]*round(0.75*len(polymask))+[False]*round(0.25*len(polymask)))
            np.random.shuffle(polymask)
        
        fit = np.polyfit(x[polymask],y[polymask],deg,w=weight[polymask])

    return polymask,fit



def ransac(x,y,deg,max_iter=100,threshold=1.5):
    best_inliers = []
    best_coeffs = None
    
    for _ in range(max_iter):
        sample_indices = np.random.choice(len(x),deg+1,replace=False)
        x_sample = x[sample_indices]
        y_sample = y[sample_indices]
        
        coeffs = np.polyfit(x_sample, y_sample, deg)
        y_pred = np.polyval(coeffs, x)
        
        distances = np.abs(y - y_pred)
        
        inliers = distances < threshold
        
        inlier_points = np.where(inliers)[0]
        if len(inlier_points) > len(best_inliers):
            best_inliers = inlier_points
            best_coeffs = coeffs
    
    return best_coeffs, best_inliers



def fixed_poly_shift(y_shift, x, y_data, coeffs):
    y_pred = np.polyval(x, coeffs) + y_shift
    loss = np.sum((y_data - y_pred)**2)
    return loss