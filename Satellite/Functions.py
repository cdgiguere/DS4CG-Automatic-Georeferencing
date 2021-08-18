import pyproj
import cv2
from skimage import exposure
import numpy as np
import yaml


def read_yaml(file_path):
    with open(file_path, "r") as fn:
        return yaml.safe_load(fn)


# takes a macconnell image and pixel coords and retuns the length in inches from the bottom left corner of the image
# that the pixel is

def pixel2inches(mac,x,y):
    x_new=x/mac.shape[0]
    y_new=y/mac.shape[1]
    xf=x_new*9.0
    yf=(9.0-(y_new*9.0))
    return xf,yf



# function to translate pixel coordinates from a TIFF image to coords in the images epsg projection
def pixel2coord(img, col, row):
    """Returns global coordinates to pixel center using base-0 raster index"""
    # unravel GDAL affine transform parameters
    c, a, b, f, d, e = img.GetGeoTransform()
    xp = a * col + b * row + a * 0.5 + b * 0.5 + c
    yp = d * col + e * row + d * 0.5 + e * 0.5 + f
    return xp, yp



def mod2mac(x1, y1,inpval,outval):
    mac_coord = pyproj.Proj(projparams='epsg:'+str(outval))
    InputGrid = pyproj.Proj(projparams='epsg:'+str(inpval))
    return pyproj.transform(InputGrid, mac_coord, x1, y1)

# Converting the epsg:26986 to gps which is epsg:4326
def mod2latlon(x1,y1,val):
    wgs84 = pyproj.Proj(projparams='epsg:4326')
    InputGrid = pyproj.Proj(projparams='epsg:'+str(val))
    return pyproj.transform(InputGrid, wgs84, x1, y1)



# takes a psuedo-grayscale MacConnell image and a colored Modern image and returns the edge detected images after
# histogram matching the modern image to the MacConnel image and shifing both to grayscale
def preprocess_images(mac, mod):
    # histogram matching to adjust the modern image to match the pixel intensity of the MacConnell image
    multi = True if mod.shape[-1] > 1 else False

    matched_mod = exposure.match_histograms(mod, mac, multichannel=multi)

    # convert to grayscale
    gray_mac = cv2.cvtColor(mac, cv2.COLOR_BGR2GRAY)
    gray_mod = cv2.cvtColor(matched_mod, cv2.COLOR_BGR2GRAY)

    return gray_mac, gray_mod


# takes gray_scale images and returns the list of keypoints from the Harris Corner detection algorithm
def compute_harris_corner_keypoints(gray_mac, gray_mod):
    harris_mac = np.zeros(np.shape(gray_mac), dtype='uint8')
    harris_mod = np.zeros(np.shape(gray_mod), dtype='uint8')
    # perform harry corner detection
    corners_mac = cv2.cornerHarris(gray_mac, 2, 3, 0.04)
    corners_mod = cv2.cornerHarris(gray_mod, 2, 3, 0.04)
    harris_mac[corners_mac > 0.01 * corners_mac.max()] = 255
    harris_mod[corners_mod > 0.01 * corners_mod.max()] = 255

    mac_keypoints_idx = np.where(corners_mac > 0.01 * corners_mac.max())
    mac_keypoints = [cv2.KeyPoint(float(r), float(c), 10) for r, c in zip(mac_keypoints_idx[1], mac_keypoints_idx[0])]

    mod_keypoints_idx = np.where(corners_mod > 0.01 * corners_mod.max())
    mod_keypoints = [cv2.KeyPoint(float(r), float(c), 10) for r, c in zip(mod_keypoints_idx[1], mod_keypoints_idx[0])]
    return mac_keypoints, mod_keypoints
