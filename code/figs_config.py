'''
aesthetics
'''
from config import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgba
import seaborn as sns

print("Loaded aesthetics module")


mesh_aes = {"show_scalar_bar": False,\
            "smooth_shading": True,\
            "specular": 0, "diffuse":0, "ambient":1}

point_aes = {"render_points_as_spheres": True, "diffuse":0, "ambient":1,"specular":0}

aorder = approaches

# colours for interpolation approaches
acolours = [
    to_hex(plt.cm.spring(0.3)),  # idw: pink
    to_hex(plt.cm.spring(0.65)),  # knn: orange
    to_hex(plt.cm.winter(0.5)),  # rbf: turquoise
    to_hex(plt.cm.summer(0.4)),  # swr: lime
    to_hex(plt.cm.winter(0.3)), # krige: blue
    to_hex(plt.cm.cool(0.65)) # regkrige: deep purple
]

apal = dict(zip(aorder, acolours))

# categories for neuromaps surface maps
corder = ['gene expression', 'MEG', 'structural', 'functional', 'metabolism', 'expansion']
ccolours = [
    to_hex(plt.cm.terrain(0.2)),  # gene: light turquoise
    to_hex(plt.cm.spring(0.65)),  # MEG: orange
    to_hex(plt.cm.cubehelix(0.7)),  # structural: violet
    to_hex(plt.cm.terrain(0.3)),  # functional: green
    to_hex(plt.cm.nipy_spectral(0.3)), # metabolism: blue
    to_hex(plt.cm.spring(0.37)) # expansion: pink
]

cpal = dict(zip(corder, ccolours))

def three_colours_gradient(lo, mid, hi, plot=True):

    n = 256; n_half = n // 2
    
    gradient1 = np.linspace(lo, mid, n_half)
    upper = np.tile(mid, (n - n_half, 1))
    colours = np.vstack([gradient1, upper])
    cmap1 = LinearSegmentedColormap.from_list("lo2mid", colours)
    
    gradient2 = np.linspace(mid, hi, n_half)
    upper = np.tile(hi, (n - n_half, 1))
    colours = np.vstack([gradient2, upper])
    cmap2 = LinearSegmentedColormap.from_list("mid2hi", colours)
    
    lo2mid = cmap1(np.linspace(0, .5, n_half))
    mid2hi = cmap2(np.linspace(0, .5, n - n_half))
    
    final_colours = np.vstack([lo2mid, mid2hi])
    lo2hi = LinearSegmentedColormap.from_list("lo2hi", final_colours)

    if plot:
        plt.figure(figsize = (3,.5))
        plt.imshow(np.linspace(0,1,n)[None,:], aspect = 'auto', cmap = lo2hi)
        plt.show()
        #plt.tight_layout()
    return lo2hi, (cmap1, cmap2)

def show_colour_scheme(cmap, map_name):
    plt.figure(figsize = (3, .5))
    plt.imshow(np.linspace(0,1,256)[None,:], aspect = 'auto', cmap = cmap)
    plt.title(map_name)
    plt.show()

# MAIN COLOUR MAPS
bpr,_ = three_colours_gradient(lo=plt.cm.Blues(0.7), mid=plt.cm.PuRd(0.4), hi=plt.cm.Oranges(0.6), plot=False)
gnrd,_ = three_colours_gradient(lo=plt.cm.Blues(0.7), mid=plt.cm.YlOrBr(0.1), hi=plt.cm.Oranges(0.6), plot=False)
