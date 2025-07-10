import ifum_code.ifum.hexagonify as hexagonify
import ifum_code.ifum.utils as utils
import shapely
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from astropy.io import fits
import scipy
matplotlib.use('TkAgg')
from shapely.geometry import shape
from shapely.validation import explain_validity

class Stack():
    '''
    Class to ...

    Attributes:
        

    Methods:
        
    '''
    def __init__(self, datafilenames: list, bad_masks, total_masks: int, mask_groups: int, wcs_stars: np.ndarray, hex_dims):
        self.datafilenames = datafilenames
        self.npzfiles = []
        for datafilename in self.datafilenames:
            self.npzfiles.append(os.path.join(os.path.relpath("out"),datafilename+"_spectra.npz"))
        self.bad_masks = bad_masks
        self.total_masks = total_masks
        self.mask_groups = mask_groups
        self.wcs_stars = wcs_stars

        self.run_params = {"x_hexes": hex_dims[0],
                        "y_hexes": hex_dims[1],
                        "f": False,
                        "off_pos": True,
                        "pixels": 99,
                        "minimize_interp": False,
                        "plot_complex": True,
                        "save": False}
        
        self.isol_sky_lines = [7794.112,7808.467,7821.503,#7949.204,7913.708*0.9+7912.252*0.1,
            7964.650,7993.332,8014.059,8025.668*0.5+8025.952*0.5,#8052.020,
            8061.979*0.5+8062.377*0.5,#8382.392*0.8+8379.903*0.1+8380.737*0.1,8310.719,8344.602*0.9+8343.033*0.1,
            8399.170,8415.231,8430.174,8465.208*0.6+8465.509*0.4,8493.389,
            8504.628*0.6+8505.054*0.4,8791.186*0.9+8790.993*0.1,8885.850,#8778.333*0.7+8776.294*0.3,8827.096*0.9+8825.448*0.1,
            8903.114,8919.533*0.5+8919.736*0.5,8943.395,8957.922*0.5+8958.246*0.5,8988.366,
            9001.115*0.5+9001.577*0.5,9037.952*0.5+9038.162*0.5,9049.232*0.6+9049.845*0.4,9374.318*0.1+9375.961*0.9,9419.729,
            9439.650,9458.528,9476.748*0.4+9476.912*0.6,9502.815,9519.182*0.4+9519.527*0.6,
            9552.534,9567.090*0.6+9567.588*0.4,9607.726,9620.630*0.7+9621.300*0.3]
        self.isol_sky_lines = utils.air_to_vacuum(np.array(self.isol_sky_lines))

    def hex_to_grid(self):
        centers_,percentages_,plot_args_ = hexagonify.hex_grid(**self.run_params)
        plt.show(block=False)
        plt.pause(10)
        plt.close()

        # step 1: make grid so each pixel adds up to 1 or nan
        fig, ax = plt.subplots()
        # fig.set_figwidth(round(plot_args_["width"])//1.8)
        # fig.set_figheight(math.ceil(plot_args_["height"])//1.8)
        fig.set_figwidth(round(plot_args_["width"])//5)
        fig.set_figheight(math.ceil(plot_args_["height"])//5)
        # fig.set_facecolor("lightgray")

        old_percentages = plot_args_["percentages"]
        # turns old_percentages to 0s which eventually becomes nans (hexagon blue #23)
        for bad_mask in self.bad_masks[0]:
            old_percentages[:,bad_mask] = 0
        for bad_mask in self.bad_masks[1]:
            old_percentages[:,bad_mask+self.total_masks//2] = 0
        # old_percentages[:,22] = 0

        new_percentages = old_percentages/(old_percentages.sum(axis=1,keepdims=True))
        cond2 = np.isnan(np.sum(new_percentages,axis=1))
        for i,pix_im in enumerate(plot_args_["pixels"]):
            if cond2[i]:
                ax.plot(*pix_im.exterior.xy,color="maroon",alpha=0.7,linewidth=5.0)

        # normalize all percentages so they add up to 1 or nan
        new_percentages_ = np.transpose(new_percentages.reshape((math.ceil(plot_args_["height"]),round(plot_args_["width"]),plot_args_["x_hexes"]*plot_args_["y_hexes"])),axes=(1,0,2))

        plot_args_["ax"] = ax
        plot_args_["percentages"] = new_percentages
        hexagonify.plot_grid(**plot_args_)
        print("plotting...")
        plt.tight_layout()
        ax.set_aspect('equal')
        plt.savefig('hex_grid.png',dpi=200,transparent=True,bbox_inches='tight',pad_inches=0.1)
        plt.show(block=False)
        plt.pause(10)
        plt.close()

        for npzfile in self.npzfiles:
            npzdata = np.load(npzfile, allow_pickle=True)
            save_dict = dict(npzdata)
            save_dict["centers"] = centers_
            save_dict["percentages"] = percentages_
            save_dict["new_percentages"] = new_percentages_
            save_dict["hexes"] = plot_args_["hexes"]
            np.savez(npzfile, **save_dict)

    def spectra_to_datacube(self):
        for idx,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile, allow_pickle=True)
            percentages = npzdata["new_percentages"]
            wl = npzdata["wl"]
            wl_start,wl_delta = np.min(wl),wl[1]-wl[0]
            intensity = npzdata["intensity"]

            data_cube = np.empty((intensity.shape[1],percentages.shape[1],percentages.shape[0]))
            data_cube_ss = np.empty((intensity.shape[1],percentages.shape[1],percentages.shape[0]))
            intensity_skysub = intensity-np.nanmedian(intensity,axis=0)

            for i in tqdm(range(len(percentages))):
                for j in range(len(percentages[0])):        
                    to_sum = (np.array([percentages[i][j]])).T*intensity
                    to_sum[np.where(percentages[i][j]==0)] = 0
                    data_cube[:,j,i] = np.sum(to_sum,axis=0)#np.sum((np.array([new_percentages_[i][j]]).T*norm_intensities),axis=0)

                    to_sum = (np.array([percentages[i][j]])).T*intensity_skysub
                    to_sum[np.where(percentages[i][j]==0)] = 0
                    data_cube_ss[:,j,i] = np.sum(to_sum,axis=0)

            hdr = fits.Header()
            hdr["CRVAL3"] = wl_start-wl_delta
            hdr["CRPIX3"] = 0
            hdr["CDELT3"] = wl_delta
            fits.writeto(os.path.join(os.path.relpath("out"),self.datafilenames[idx]+"_datacube_.fits"), data_cube, hdr, overwrite=True)
            # fits.writeto(os.path.join(os.path.relpath("out"),self.datafilenames[idx]+"_datacube_ss_.fits"), data_cube_ss, hdr, overwrite=True)

            save_dict = dict(npzdata)
            save_dict["datacube"] = data_cube
            save_dict["datacube_ss"] = data_cube_ss
            save_dict["header"] = hdr
            np.savez(npzfile, **save_dict)

    def wcs_datacubes(self):
        # WORKFLOW
        # 2D cross-correlaiton isn't perfect. but it works sometimes
        # SO, we can use cross-correlation to find general shift
        # THEN, we can fit each datacube seperately for WCS stars

        # get shifts
        print("calculting 2D cross-correlation shift guesses...")
        data_cubes = []
        for npzfile in self.npzfiles:
            npzdata = np.load(npzfile, allow_pickle=True)
            data_cubes.append(npzdata["datacube_ss"])

        ref_im = np.nanmedian(data_cubes[0],axis=0)
        ref_im = utils.normalize(np.nan_to_num(ref_im,np.nanmedian(ref_im)))
        shifts = [[0,0]]
        for data_cube in data_cubes[1:]:
            im = np.nanmedian(data_cube,axis=0)
            im = utils.normalize(np.nan_to_num(im,np.nanmedian(im)))
            shift = utils.get_lag_2d(ref_im,im)
            shifts.append(shift)

            # plt.subplot(121)
            # plt.imshow(ref_im,origin="lower",cmap="Greys_r")
            # plt.subplot(122)
            # plt.imshow(scipy.ndimage.shift(im, shift=shift, mode='constant', cval=np.median(im)),origin="lower",cmap="magma")
            # plt.show()
        shifts = np.array(shifts)
        # print(shifts)

        

        # get WCS stars
        stars_num = len(self.wcs_stars)

        fig = plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title("WCS stars order")
        for i,star in enumerate(self.wcs_stars):
            plt.text(star[0],star[1],f"{i+1}",weight="bold",ma="center")
            plt.scatter(star[0],star[1])
        # (scale aspect so declination distance = ra distance)
        plt.gca().set_aspect(1./np.cos(np.deg2rad(np.mean(self.wcs_stars[:,1]))))
        plt.gca().invert_xaxis()
        plt.xlabel("RA")
        plt.ylabel("Dec")

        plt.subplot(122)
        plt.title(f"click to select {stars_num} stars (IN ORDER)",weight="bold")
        for i,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile, allow_pickle=True)
            hexes = npzdata["hexes"]
            shifted_hexes = []
            for hex in hexes:
                shifted_hexes.append(shapely.affinity.translate(hex,xoff=shifts[i,1],yoff=shifts[i,0]))

            intensity = npzdata["intensity"]
            hex_medians = np.nanmedian(intensity,axis=1)
            colors = plt.get_cmap('gray')(utils.normalize(hex_medians))
            for j,s_hex in enumerate(shifted_hexes):
                plt.fill(*s_hex.exterior.xy,color=colors[j],alpha=1/len(self.npzfiles)*3,edgecolor="none")
        plt.gca().set_aspect("equal")
        plt.axis("off")

        print(f"select {stars_num} WCS stars in order; see popup window")

        star_coords = plt.ginput(stars_num, timeout=300)
        plt.show(block=False)
        plt.close(fig)


        print("optimizing WCS star coordinates...")
        # optimize star coordinates with nearby hexagons
        all_coords = []
        for i,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile, allow_pickle=True)
            hexes = npzdata["hexes"]
            intensity = npzdata["intensity"]
            hex_medians = np.nanmedian(intensity,axis=1)

            shifted_coords = np.array(star_coords)
            shifted_coords[:,0] = shifted_coords[:,0]-shifts[i,1]
            shifted_coords[:,1] = shifted_coords[:,1]-shifts[i,0]

            # plt.imshow(np.nanmedian(data_cubes[i],axis=0),origin="lower",cmap="Greys_r")

            shifted_coords_corr = []
            for coord in shifted_coords:
                nearby_hexes = []
                nearby_centroids = []
                for i,hex in enumerate(hexes):
                    distance = np.sum((coord-np.array(hex.centroid.xy).flatten())**2)**0.5
                    if distance<hexes[0].length*1/3:
                        nearby_hexes.append(i)
                        nearby_centroids.append(np.array(hex.centroid.xy).flatten())
                if len(nearby_hexes)==0:
                    # print("no nearby hexes!")
                    shifted_coords_corr.append([np.nan,np.nan])
                else:
                    nearby_hexes = np.array(nearby_hexes)
                    nearby_weights = utils.normalize(hex_medians[nearby_hexes])
                    nearby_centroids = np.array(nearby_centroids)

                    # for nearby_hex in nearby_hexes:
                        # plt.fill(*hexes[nearby_hex].exterior.xy,color="blue",alpha=0.2,edgecolor="none")

                    # NOTE: will accept NaN values, but this can mess with result a bit
                    # nearby_weights = np.nan_to_num(nearby_weights,nan=0)
                    corr_coord = np.average(nearby_centroids,axis=0,weights=nearby_weights)
                    shifted_coords_corr.append(corr_coord)
                    # print(coord,corr_coord)
            shifted_coords_corr = np.array(shifted_coords_corr)
            all_coords.append(shifted_coords_corr)
            plt.scatter(shifted_coords[:,0],shifted_coords[:,1],color="red")
            plt.scatter(shifted_coords_corr[:,0],shifted_coords_corr[:,1],color="blue")
            plt.show()

        # perform WCS transform
        print("performing WCS transforms...")
        all_wcs_hexes = []
        min_x,max_x = np.inf,-np.inf
        min_y,max_y = np.inf,-np.inf
        for i,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile, allow_pickle=True)
            hexes = npzdata["hexes"]

            xys = all_coords[i]
            mask = np.isnan(xys).any(axis=1)
            xys = xys[~mask]
            radecs = self.wcs_stars[~mask]
            # print(xys,radecs)

            var = [0.001,0.001,0.001,0.001]
            args = [xys,xys,radecs]
                        
            result = scipy.optimize.minimize(utils.xy_to_radec_minimize, var, args=(args))

            calc_radecs = np.array([utils.xy_to_radec(result.x,[xys,xys,radecs])]).T
            # print(result.x)
            # plt.scatter(radecs[:,0],radecs[:,1],alpha=0.8,color="blue",marker="x",s=30)
            # plt.scatter(calc_radecs[:,0],calc_radecs[:,1],color="orange")
            # plt.show()

            save_dict = dict(npzdata)
            wcs_hexes = []
            for hex in hexes:
                wcs_hex = np.array(utils.xy_to_radec(result.x,[np.array(list(hex.exterior.coords)),xys,radecs])).T              
                wcs_hexes.append(shapely.Polygon(wcs_hex))
                if not wcs_hexes[-1].is_valid:
                    print(explain_validity(wcs_hexes[-1]))
                    # plt.fill(*wcs_hexes[-1].exterior.xy,color="blue",alpha=0.5,edgecolor="black")
                    # plt.show()
                all_wcs_hexes.append(shapely.Polygon(wcs_hex))

                min_x,max_x = min(min_x,shapely.Polygon(wcs_hex).bounds[0]),max(max_x,shapely.Polygon(wcs_hex).bounds[2])
                min_y,max_y = min(min_y,shapely.Polygon(wcs_hex).bounds[1]),max(max_y,shapely.Polygon(wcs_hex).bounds[3])
            
            save_dict["wcs_hexes"] = wcs_hexes
            save_dict["cd_matrix"] = result.x
            np.savez(npzfile, **save_dict)
            print(f"\t{self.datafilenames[i]} used {len(xys)} reference stars")

        for i,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile, allow_pickle=True)

            dx = (max_x-min_x)/self.run_params["pixels"]
            dy = np.cos(np.radians(np.mean([max_y,min_y])))*dx
            xv,yv = np.meshgrid(np.arange(min_x+dx*.5,max_x,dx),np.arange(min_y+dy*.5,max_y,dy))
            wcs_pixels = hexagonify.pxs_from_px(xv,yv,dx,dy)
            
            save_dict = dict(npzdata)
            save_dict["wcs_pixels"] = wcs_pixels
            save_dict["wcs_pixels_"] = np.array(wcs_pixels).reshape((xv.shape[0],xv.shape[1]))
            np.savez(npzfile, **save_dict)
        print("WCS transformations complete")

    def full_intensity_callibration(self):
        # proper intensity calculation
        wl = np.load(self.npzfiles[0],allow_pickle=True)["wl"]
        all_intensities = np.empty((0,wl.size))
        avg_intensities = np.empty((0,wl.size))
        for idx,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile, allow_pickle=True)
            for intensity in npzdata["intensity"]:
                all_intensities = np.vstack((all_intensities,intensity))
            avg_intensities = np.vstack((avg_intensities,np.nanmedian(npzdata["intensity"],axis=0)))
        # avg_sky = np.nanmedian(all_intensities,axis=0)
        # print(all_intensities.shape)

        # first, get all sky intensity fits
        sky_ints = np.empty((avg_intensities.shape[0],len(self.isol_sky_lines)))
        sky_errs = np.empty((avg_intensities.shape[0],len(self.isol_sky_lines)))

        for idx,intensity in enumerate(avg_intensities):
            sky_int = []
            sky_err = []
            offset = 8

            for i,line in enumerate(self.isol_sky_lines):
                mask_area = ((wl)>(self.isol_sky_lines[i]-offset))&((wl)<(self.isol_sky_lines[i]+offset))
                mask_area = mask_area&(~np.isnan(intensity))
                
                try:
                    p0 = [0,
                    0,
                    np.nanmax(intensity[mask_area]),
                    np.argmax(intensity[mask_area])+np.nanmin(wl[mask_area]),
                    5/3]
                    popt,pcov = scipy.optimize.curve_fit(utils.gauss_background,
                                                            wl[mask_area],
                                                            intensity[mask_area],p0=p0)
                    
                    perr = np.sqrt(np.diag(pcov))
                    gauss_x = np.linspace(wl[mask_area][0],wl[mask_area][-1],100)
                    
                    popt[0] = 0
                    popt[1] = 0
                    sky_ints[idx,i] = np.trapezoid(utils.gauss_background(gauss_x,*popt),gauss_x)
                    sky_errs[idx,i] = perr[3]
                
                except:
                    sky_ints[idx,i] = 0
                    sky_errs[idx,i] = np.inf

        sky_ints = np.array(sky_ints)
        # ratios = sky_ints/sky_ints[0]
        sky_errs = np.array(sky_errs)

        # colors = []
        # for i in range(9):
        #     colors += ["#"+str(os.urandom(3).hex())]*552
        # for m in np.arange(all_intensities.shape[0]):
        #     plt.plot(wl,all_intensities[m],color=colors[m],alpha=0.01)
        # plt.show()

        avg_sky = np.average(sky_ints,axis=0,weights=1/sky_errs)

        deg = 1
        full_intensity = np.empty((0,wl.size))
        for idx, (sky_int,sky_err) in enumerate(zip(sky_ints,sky_errs)):
            ratio = avg_sky/sky_int

            if not np.all(ratio == 1):
                mask0 = ((ratio<(np.nanmedian(ratio)+3*np.nanstd(ratio[np.isfinite(ratio)])))
                        &(ratio>(np.nanmedian(ratio)-3*np.nanstd(ratio[np.isfinite(ratio)]))))
            else:
                mask0 = ratio==ratio
            ratio = ratio[mask0]
            w_ = np.nan_to_num(1/np.array(sky_err)[mask0],nan=0)

            fit_mask,int_fit = utils.sigma_clip(self.isol_sky_lines[mask0],ratio,deg,w_,sigma=3.0)
            # plt.scatter(self.isol_sky_lines[mask0],ratio,c=ifum_utils.normalize(w_))
            # plt.scatter(self.isol_sky_lines[mask0][fit_mask],ratio[fit_mask],c="red",marker="x",alpha=0.5)
            # plt.show()

            fit_mask_,int_fit = utils.sigma_clip(self.isol_sky_lines[mask0][fit_mask],ratio[fit_mask],deg+1,w_[fit_mask],sigma=1.0)
            # plt.scatter(self.isol_sky_lines[mask0][fit_mask],ratio[fit_mask],c=ifum_utils.normalize(w_[fit_mask]))
            # plt.scatter(self.isol_sky_lines[mask0][fit_mask][fit_mask_],ratio[fit_mask][fit_mask_],c="red",marker="x",alpha=0.5)
            # plt.plot(self.isol_sky_lines[mask0][fit_mask],np.poly1d(int_fit)(self.isol_sky_lines[mask0][fit_mask]),c="red")
            # plt.show()
            
            for intensity in all_intensities[idx*self.total_masks:(idx+1)*self.total_masks]:
                full_intensity = np.vstack((full_intensity,intensity*np.poly1d(int_fit)(wl)))

        # for m in np.arange(full_intensity.shape[0]):
        #     plt.plot(wl,full_intensity[m],color=colors[m],alpha=0.01)
        # plt.show()

        for idx,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile, allow_pickle=True)
            
            save_dict = dict(npzdata)
            intensity_seg = full_intensity[idx*self.total_masks:(idx+1)*self.total_masks]
            save_dict["norm_intensity"] = intensity_seg
            np.savez(npzfile, **save_dict)

    

    def stack_datacubes(self):
        all_wcs_hexes = []
        all_intensity = []
        for idx,npzfile in enumerate(self.npzfiles):
            npzdata = np.load(npzfile,allow_pickle=True)
            for wcs_hex in npzdata["wcs_hexes"]:
                all_wcs_hexes.append(wcs_hex)
            for intensity in npzdata["norm_intensity"]:
                all_intensity.append(intensity)
            wcs_pixels = npzdata["wcs_pixels"]
            wcs_pixels_ = npzdata["wcs_pixels_"]
        all_wcs_hexes = np.array(all_wcs_hexes)
        all_intensity = np.array(all_intensity)

        # print(wcs_pixels.shape)
        # print(wcs_pixels)
        # print(all_wcs_hexes.shape)
        # print(all_wcs_hexes)

        # print("invalid hexes?")
        # for wcs_pixel in all_wcs_hexes:
        #     if not wcs_pixel.is_valid:
        #         print(explain_validity(wcs_pixel))
        #         plt.fill(*wcs_pixel.exterior.xy,color="blue",alpha=0.5,edgecolor="black")
        #         plt.show()

        percentages = hexagonify.get_overlap_percentages(wcs_pixels,all_wcs_hexes)
        for bad_mask in self.bad_masks[0]:
            percentages[:,bad_mask::self.total_masks] = 0
        for bad_mask in self.bad_masks[1]:
            percentages[:,(bad_mask+self.total_masks//2)::self.total_masks] = 0    
        
        # some hexagons do not have valid values
        # valid_hex_mask = ~np.any(np.isnan(all_intensity),axis=1)
        # percentages[:,~valid_hex_mask] = 0
        row_sums = percentages.sum(axis=1,keepdims=True)
        row_sums[row_sums==0] = 1
        new_percentages = percentages/row_sums
        new_percentages_ = np.transpose(new_percentages.reshape((wcs_pixels_.shape[0],wcs_pixels_.shape[1],all_intensity.shape[0])),axes=(1,0,2))

        data_cube = np.full((all_intensity.shape[1], new_percentages_.shape[1], new_percentages_.shape[0]), np.nan)
        data_cube_ss = np.full_like(data_cube, np.nan)
        all_intensity_skysub = all_intensity-np.nanmedian(all_intensity,axis=0)

        # def compute_weighted_average(intensity, weights):
        #     masked_weights = np.where(np.isnan(intensity),0,weights[:,None])
        #     weighted_intensity = masked_weights*np.nan_to_num(intensity)
        #     total_weights = np.sum(masked_weights,axis=0)
        #     total_weights[total_weights==0] = np.nan
        #     return np.sum(weighted_intensity,axis=0) / total_weights

        def compute_weighted_average(intensity, weights):
            valid_mask = ~np.isnan(intensity)
            masked_weights = weights[:, None] * valid_mask
            weighted_intensity = weights[:, None] * np.nan_to_num(intensity)
            total_weights = np.sum(masked_weights, axis=0)
            weighted_sum = np.sum(weighted_intensity, axis=0)
            result = weighted_sum / total_weights
            result[total_weights == 0] = np.nan
            return result

        for i in tqdm(range(len(new_percentages_))):
            for j in range(len(new_percentages_[0])):   
                weights = new_percentages_[i][j]
                data_cube[:,j,i] = compute_weighted_average(all_intensity,weights)
                data_cube_ss[:,j,i] = compute_weighted_average(all_intensity_skysub,weights)



        # get total CD matrix!
        # center pixel, get pixels around it, optimize CD on those pixels
        xpix = np.array(wcs_pixels_).shape[1]
        ypix = np.array(wcs_pixels_).shape[0]
        centerx,centery = xpix//2,ypix//2

        # get a square around the center pixel
        xys = np.array([[centerx+0.5,centery+0.5],
                        [centerx-10.5,centery-10.5],
                        [centerx+10.5,centery-10.5],
                        [centerx-10.5,centery+10.5],
                        [centerx+10.5,centery+10.5]])
        # radecs from those squares
        radecs = []
        for pix_im in np.array(wcs_pixels_)[(xys-0.5).astype(int)[:,1],(xys-0.5).astype(int)[:,0]]:
            radecs.append(np.array(pix_im.centroid.xy).flatten())
        radecs = np.array(radecs)

        # solve for CD matrix
        var = np.load(self.npzfiles[0],allow_pickle=True)["cd_matrix"]
        args = [xys,xys,radecs]
        result = scipy.optimize.minimize(utils.xy_to_radec_minimize, var, args=(args))

        # radec_pxs = []
        # for i in range(len(xys)):
        #     calc_radec_px = np.array(ifum_utils.xy_to_radec(result.x,[xys[i:i+1],xys,radecs])).T
        #     radec_pxs.append(calc_radec_px)
        # radec_pxs = np.array(radec_pxs)[:,0]

        refx,refy = np.mean(xys,axis=0)
        refra,refdec = np.mean(radecs,axis=0)

        # get wl info
        wl = np.load(self.npzfiles[0],allow_pickle=True)["wl"]
        wl_start,wl_delta = np.min(wl),wl[1]-wl[0]

        # create header
        hdr = fits.Header()
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"
        hdr["EQUINOX"] = 2000.0
        hdr["LONPOLE"] = 180.0
        hdr["LATPOLE"] = 0.0
        hdr["CRVAL1"] = refra
        hdr["CRVAL2"] = refdec
        hdr["CRPIX1"] = refx
        hdr["CRPIX2"] = refy
        hdr["CUNIT1"]  = 'deg     '                                  
        hdr["CUNIT2"]  = 'deg     '
        hdr["CD1_1"] = result.x[0]
        hdr["CD1_2"] = result.x[1]
        hdr["CD2_1"] = result.x[2]
        hdr["CD2_2"] = result.x[3]
        hdr["CRVAL3"] = wl_start-wl_delta
        hdr["CRPIX3"] = 0
        hdr["CUNIT3"] = "Angstrom"
        hdr["CDELT3"] = wl_delta

        fits.writeto(os.path.join(os.path.relpath("out"),"stacked_datacube.fits"), data_cube, hdr, overwrite=True)
        fits.writeto(os.path.join(os.path.relpath("out"),"stacked_datacube_ss.fits"), data_cube_ss, hdr, overwrite=True)
        print("final data cubes saved!")