#--------------------------------------------------------------
# This is a demo file intended to show the use of the SNIC algorithm
# Please compile the C files of snic.h and snic.c using:
# "python snic.c" on the command prompt prior to using this file.
#
# To see the demo use: "python SNICdemo.py" on the command prompt
#------------------------------------------------------------------
import sys
import os
from tqdm import tqdm

# Adjust this to wherever your .so file lives
build_dir = os.path.join(os.path.dirname(__file__), "../build")

if build_dir not in sys.path:
        sys.path.insert(0, build_dir)

import subprocess
from PIL import Image
import numpy as np
from timeit import default_timer as timer
from snic_ext import SNIC_main


def seg_to_boundary(seg: np.ndarray) -> np.ndarray:
    """
    Convert a 2D segmentation label map into a binary boundary map.

    Parameters
    ----------
    seg : (H, W) ndarray of int
        Segmentation labels.

    Returns
    -------
    boundary : (H, W) ndarray of uint8
        Binary boundary map (1 = boundary, 0 = non-boundary).
    """
    boundary = np.zeros(seg.shape, dtype=np.uint8)

    # vertical neighbors
    boundary[1:, :] |= seg[1:, :] != seg[:-1, :]
    boundary[:-1, :] |= seg[:-1, :] != seg[1:, :]

    # horizontal neighbors
    boundary[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    boundary[:, :-1] |= seg[:, :-1] != seg[:, 1:]

    return boundary

def segment(imgname,numsuperpixels,compactness,doRGBtoLAB):
#--------------------------------------------------------------
# read image and change image shape from (h,w,c) to (c,h,w)
		img = Image.open(imgname)
		img = np.asarray(img)

		dims = img.shape
		print(dims)
		h,w,c = dims[0],dims[1],1
		if len(dims) > 1:
			c = dims[2]

		img = img.transpose(2,0,1)

		#--------------------------------------------------------------
		# Reshape image to a single dimensional vector
		#--------------------------------------------------------------
		img = img.reshape(-1).astype(np.double)
		labels = np.zeros((h,w), dtype = np.int32)
		numlabels = np.zeros(1,dtype = np.int32)
		#--------------------------------------------------------------
		# Prepare the pointers to pass to the C function
		#--------------------------------------------------------------

		start = timer()
		SNIC_main(img,w,h,c,numsuperpixels,compactness,doRGBtoLAB,labels,numlabels)
		end = timer()

		#print("time taken in seconds:",end - start)

		#--------------------------------------------------------------
		# Collect labels
		#--------------------------------------------------------------
		return labels.reshape(h,w),numlabels[0]


	# lib.SNICmain.argtypes = [np.ctypeslib.ndpointer(dtype=POINTER(c_double),ndim=2)]+[c_int]*4 +[c_double,c_bool,ctypes.data_as(POINTER(c_int)),ctypes.data_as(POINTER(c_int))]

def drawBoundaries(imgname,labels,numlabels):

    img = Image.open(imgname)
    img = img.convert("RGB")
# img = imread(imgname)
    img = np.array(img)

    ht,wd = labels.shape

    for y in range(1,ht-1):
        for x in range(1,wd-1):
            if labels[y,x-1] != labels[y,x+1] or labels[y-1,x] != labels[y+1,x]:
    #set alpha full and color black
                img[y,x,:3] = 0
                #img[y,x,3:] = 255


    return img
import matplotlib.pyplot as plt

def show_image_with_labels(img, labels):
    """
    img    : (H, W, 3) or (H, W) numpy array
    labels : (H, W) numpy array of int labels
    """
    fig, ax = plt.subplots()
    im = ax.imshow(img)
    ax.set_title("Hover to see pixel info")

    # Annotation box
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="w"),
        arrowprops=dict(arrowstyle="->"),
    )
    annot.set_visible(False)

    h, w = labels.shape

    def update_annot(event):
        x, y = int(event.xdata), int(event.ydata)
        label = labels[y, x]

        if img.ndim == 3:
            rgb = img[y, x]
            text = f"x={x}, y={y}\nlabel={label}\nRGB={tuple(rgb)}"
        else:
            text = f"x={x}, y={y}\nlabel={label}\nvalue={img[y, x]}"

        annot.xy = (x, y)
        annot.set_text(text)
        annot.set_visible(True)

    def on_mouse_move(event):
        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            if 0 <= x < w and 0 <= y < h:
                update_annot(event)
                fig.canvas.draw_idle()
        else:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
    plt.show()

def snicdemo(base_path, img_name):
		#--------------------------------------------------------------
		# Set parameters and call the C function
		#--------------------------------------------------------------
		numsuperpixels = 12000
		compactness = 20
		doRGBtoLAB = True # only works if it is a three channel image
		# imgname = "/Users/achanta/Pictures/classics/lena.png"
		imgname = os.path.join(base_path, img_name)

		labels,numlabels = segment(imgname,numsuperpixels,compactness,doRGBtoLAB)

		segimg = drawBoundaries(imgname,labels,numlabels)

		namesplt = img_name.split(".")
		name = namesplt[0] + "_snic." + namesplt[1]
		outpath = os.path.join(".", name)

		Image.fromarray(segimg).save(outpath)
snicdemo(".", "houses.jpg")
