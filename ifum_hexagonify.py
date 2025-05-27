import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from shapely import Polygon
import sys
import os

def hex_corners(center=[0,0],s=1,f=True):
    '''
    creates hexagon corner coordinates with given information
    
    INPUTS
    center   - vector [x,y] of hexagon center
    s (size) - radius of "outer" circle, or from center to vertex
    f (flat) - True means the hexagon is flat on top, False means pointy on top
    
    OUTPUT
    list of corners, in 2D numpy array: [[x0,y0],...,[x5,y5]]
    '''
    
    offset = 0 if f else -30
    angles_rad = np.pi/180 * (np.array([0,1,2,3,4,5])*60 + offset)
    
    return np.array([center[0]+s*np.cos(angles_rad),center[1]+s*np.sin(angles_rad)]).T

def plot_grid(ax,x_hexes,y_hexes,s,width,height,pixels,hexes,percentages,plot_complex=True,show_numbers=True,cmap="twilight"):
    '''
    plots a hexagon grid overlaid by colored pixels

    INPUT
    ax          - figure axes to plot on
    s           - size of hexagons, used for scaling
    width       - width of hexagon lattice
    height      - height of hexagon lattice
    pixels      - pixel shapely polygon objects
    hexes       - hexagon shapely polygon objects
    percentages - output percentages from get_overlap_percentages
    cmap        - matplotlib colormap
    x_hexes, y_hexes, s, plot_complex

    OUTPUT
    returns plot
    '''        
    
    colors = sns.color_palette(cmap,x_hexes*y_hexes)
    colors = sns.color_palette(["#854001","#D45D00","#FF8C01","#F3C84D","#BEE3B6","#7998CC","#625EB2"],x_hexes*y_hexes)

    # colors = ["#542E26","#854001","#D45D00","#FF8C01","#F3C84D","#BEE3B6","#7998CC","#625EB2"]*400

    # plots pixel grid
    if plot_complex:
        # for each pixel, color in using weighted means from percentage overlap
        for i, pix_im in enumerate(tqdm(pixels,desc="graphing pixels",colour="#DA70D6")):
            color = np.sum(np.repeat(percentages[i],3).reshape((len(percentages[i]),3))*colors,axis=0)
            ax.fill(*pix_im.exterior.xy,color=color.round(decimals=3),alpha=0.5)
            # if show_numbers:
            #     ax.text(pix_im.centroid.x,pix_im.centroid.y,f'{i%round(width)},{round((i-i%round(width))/(round(width)))}',horizontalalignment='center',verticalalignment='center',size='x-small',color="white")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        # plots the simple grid
        ax.fill(*Polygon([[0,0],[round(width),0],[round(width),math.ceil(height)],[0,math.ceil(height)]]).exterior.xy,color="purple",alpha=0.1)
        ax.set_xticks(np.arange(0,round(width)+1e-1,1))
        ax.set_yticks(np.arange(0,math.ceil(height)+1e-1,1))
        ax.grid(which='both')
        ax.tick_params(length=0,labelbottom=False,labelleft=False)
    
    # overplot the hexagon grid, along with number and locations
    for i, hex_im in enumerate(hexes):
        color = colors[i] if plot_complex else "darkorange"
        ax.plot(*hex_im.exterior.xy,color=color,linewidth=5.0)
        if show_numbers:
            # ax.text(hex_im.centroid.x,hex_im.centroid.y,f'{i}: {(i-i%y_hexes)//y_hexes},{i%y_hexes}',horizontalalignment='center',verticalalignment='center',weight="bold",size=s*5,color=color)
            ax.text(hex_im.centroid.x,hex_im.centroid.y,f'{i}',horizontalalignment='center',verticalalignment='center',weight="bold",size=s*5,color=color)

    return ax

def scale_hexagon_lattice(x_hexes,y_hexes,f,pixels,minimize_interp):
    '''
    scales a hexagon lattice to fit a certain pixel grid

    INPUTS
    x_hexes, y_hexes, f, pixels, minimize_interp

    OUTPUT
    returns s (size) of hexagon
    '''
    
    spacings = [3/2,3**0.5] if f else [3**0.5,3/2]
    width =  x_hexes*spacings[0]+0.5 if f else x_hexes*spacings[0]+0.5*3**0.5
    height = y_hexes*spacings[1]+0.5*3**0.5 if f else y_hexes*spacings[1]+0.5
    if minimize_interp:
        pixels = pixels-(pixels%((y_hexes+0.5)*2)) if f else pixels-(pixels%((x_hexes+0.5)*2))
        if pixels>0:
            print(f"now using {round(pixels)} {'vertical' if f else 'horizontal'} pixels")
            print()
        else:
            sys.exit(f"pixels must be greater than 2 times number of {'vertical' if f else 'horizontal'} hexagons. if this is intentional, set minimize_interp=False.")
        s = pixels/height if f else pixels/width
    else:
        s = pixels/width

    return s
    
def get_hex_lattice_centers(x_hexes,y_hexes,s,f,off_pos):
    '''
    INPUTS
    x_hexes,y_hexes,s,f,off_pos

    OUTPUT
    total width and height of the lattice
    centers of hexagons, formatted with centers in 2d array
        [[[x00,y00],...,[xn0,yn0]],
         ...
         [[x0m,y0m],...,[xnm,ynm]]]
    '''
    
    if off_pos:
        start = [s,s*0.5*3**0.5] if f else [s*0.5*3**0.5,s]
    else:
        start = [s,s*3**0.5] if f else [s*3**0.5,s]
    offset = 1 if off_pos else -1
    spacings = [s*3/2,s*3**0.5] if f else [s*3**0.5,s*3/2]
    width =  x_hexes*spacings[0]+s*0.5 if f else x_hexes*spacings[0]+s*0.5*3**0.5
    height = y_hexes*spacings[1]+s*0.5*3**0.5 if f else y_hexes*spacings[1]+s*0.5
    
    print(f"hexagon grid: {width:.3f},{height:.3f}")
    print(f"pixel grid: {round(width)},{math.ceil(height)}")
    print(f"{round(width)*math.ceil(height)} pixels")

    centers = []
    for y in np.arange(0,y_hexes):
        for x in np.arange(0,x_hexes)[::-1]:
            center = [x*spacings[0]+start[0],offset*(x%2)*0.5*spacings[1]+y*spacings[1]+start[1]] if f else [offset*(y%2)*0.5*spacings[0]+x*spacings[0]+start[0],y*spacings[1]+start[1]]
            centers.append(center)


    real_map = np.arange(x_hexes*y_hexes)

    for i in range(x_hexes,len(real_map),2*x_hexes):
        real_map[i:i+x_hexes] = real_map[i:i+x_hexes][::-1]
    centers = np.array(centers)[real_map]
    
    # TEMPORARY re-order
    # import pandas as pd
    # real_map = pd.read_csv(r"C:\Users\daniel\Downloads\aperture_map.csv")
    # reverse every set of x_hexes 
    # print(real_map.ccd_aper[:26])
    # centers = np.array(centers)[real_map.ccd_aper-1]
    #############################################################3
    
    return width,height,centers.reshape((x_hexes,y_hexes,2))

def hexes_from_hex(centers,s=1,f=True):
    '''
    from centers and with constant size/orientation, create a list of shapely hexagons

    INPUTS
    centers  - centers of hexagons, formatted as output from get_hex_lattice_centers
    s, f

    OUTPUTS
    list of shapely polygons
    '''
    
    poly_list = []
    for xh in range(len(centers)):
        for yh in range(len(centers[0])):
            poly_list.append(Polygon(hex_corners(centers[xh,yh],s,f)))
            
    return poly_list

def pxs_from_px(xv,yv,s1=1,s2=1):
    '''
    from x and y coordinates, create pixels of specified size centered at coordinates

    INPUTS
    xv - 2D numpy array of x coordinates
    yv - 2D numpy array of y coordinates
        xv, yv are outputs to np.meshgrid
    s  - size of pixel

    OUTPUTS
    list of shapely polygons
    '''
    
    poly_list = []
    for i in range(len(xv)):
        for j in range(len(xv[0])):
            poly_list.append(Polygon([[xv[i,j]-s1*0.5,yv[i,j]-s2*0.5],
                                      [xv[i,j]+s1*0.5,yv[i,j]-s2*0.5],
                                      [xv[i,j]+s1*0.5,yv[i,j]+s2*0.5],
                                      [xv[i,j]-s1*0.5,yv[i,j]+s2*0.5]]))
            
    return poly_list

def get_overlap_percentages(shape1,shape2):
    '''
    gets percentage overlaps, iterating over all shapes in shape1 and shape2
    each row contains shape1_i's percentage overlaps with all shape2_js

    INPUTS
    shape1 - list of shapely polygons
    shape2 - list of shapely polygons 
    
    OUTPUT
    percentage overlap, intersection divided by total area of shape1 (shape1 is x% covered by shape2)
        [[shape1_0_overlap_shape2_0,...,shape1_0_overlap_shape2_n],
         ...
         [shape1_m_overlap_shape2_0,...,shape1_m_overlap_shape2_n]]
    '''
    
    percentages = np.zeros((len(shape1),len(shape2)))
    for i, s_ in enumerate(tqdm(shape1,desc="overlap percentages",colour="#FFA500")):
        for j, h_ in enumerate(shape2):
            if s_.intersects(h_):
                percentages[i][j] = s_.intersection(h_).area/s_.area

    return percentages

def hex_grid(x_hexes,y_hexes,f,off_pos,pixels,minimize_interp,plot_complex=True,save=True):
    '''
    creates an overlaying pixel grid over a lattice of hexagons

    INPUTS
    x_hexes         - number of hexagons in x (horizontal) direction
    y_hexes         - number of hexagons in y (vertical) direction
    f (flat)        - True means the hexagon is flat on top, False means pointy on top
    off_pos         - True means next row/column is shifted to the right/up
    pixels          - number of pixels to fit on horizontal axis
    minimize_interp - True fits an integer number of pixels in square-orthogonal edges of hexagons
        if f is also True, then pixels input will specify vertical axis
    plot_complex    - True will plot pixels as weighted mean colors based on overlapping hexagon(s)
        False will simply plot pixel grid over hexagon lattice, without coordinates
    save            - True will save graph in same directory as code

    OUTPUTS
    centers     - output from get_hex_lattice_centers
    percentages - output from get_overlap_percentages, but formatted like centers
        can access full percentage list for a pixel using percentages[x,y]
    plot_args   - same as inputs to plot_grid
    '''
    # first, calculate size of hexagon to fit into given pixel scale
    s = scale_hexagon_lattice(x_hexes,y_hexes,f,pixels,minimize_interp)
    
    # calculate useful parameters for hexagon grid
    width,height,centers = get_hex_lattice_centers(x_hexes,y_hexes,s,f,off_pos)
    
    # create hexagon shapes
    hexes = hexes_from_hex(centers,s,f)
    # create pixel shapes
    xv,yv = np.meshgrid(np.arange(0.5,round(width),1),np.arange(0.5,math.ceil(height),1))
    pixels = pxs_from_px(xv,yv)
    
    # for each pixel, get percentage in each hexagon
    percentages = get_overlap_percentages(pixels,hexes)
    
    

    # plot grid, with extra conditions if complex
    fig, ax = plt.subplots()
    fig.set_figwidth(round(width)//1.8)
    fig.set_figheight(math.ceil(height)//1.8)
    fig.set_facecolor("lightgray")

    if plot_complex:
        tol = 1e-6
        cond0 = np.sum(percentages,axis=1)>tol
        cond1 = np.sum(percentages,axis=1)>(1-tol)
        print(f"{np.sum(cond0)} ({np.sum(cond0)/len(cond0)*100:.3f}%) of pixels overlap with hexagon grid")
        print(f"{np.sum(cond1)} ({np.sum(cond1)/len(cond1)*100:.3f}%) of pixels overlap fully with hexagon grid")

        for i,pix_im in enumerate(pixels):
            if not cond0[i]:
                ax.fill(*pix_im.exterior.xy,color="grey")
            elif not cond1[i]:
                ax.plot(*pix_im.exterior.xy,color="tomato",alpha=0.7,linewidth=3.0)
    
    plot_args = {"ax": ax,
                 "x_hexes": x_hexes,
                 "y_hexes": y_hexes,
                 "s": s,
                 "width": width,
                 "height": height,
                 "pixels": pixels,
                 "hexes": hexes,
                 "percentages": percentages,
                 "plot_complex": plot_complex,
                 "show_numbers": True,
                 "cmap": "icefire"}
    plot_grid(**plot_args)
    
    plt.tight_layout()
    ax.set_aspect('equal')

    if save:
        print("saving...")
        plt.savefig('hex_grid.png',dpi=400,transparent=True,bbox_inches='tight',pad_inches=0.1)
        print(f"  saved: {os.path.join(os.getcwd(),'')}hex_grid.png")
    print("plotting...")
    

    
    percentages = np.transpose(percentages.reshape((math.ceil(height), round(width), x_hexes*y_hexes)),axes=(1,0,2))

    return centers,percentages,plot_args