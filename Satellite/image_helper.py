import cv2
import os
from osgeo import gdal

def cv_make_quadrants(img):
    #dividing the img using cv2
    #img=cv2.imread(img_str)
    quadrants = []
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    left1 = img[:, :width_cutoff]
    right1 = img[:, width_cutoff:]
    img = cv2.rotate(left1, cv2.ROTATE_90_CLOCKWISE)
    height = img.shape[0]
    width = img.shape[1]
    width_cutoff = width // 2
    l2 = img[:, :width_cutoff]
    l1 = img[:, width_cutoff:]
    l2 = cv2.rotate(l2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    l1 = cv2.rotate(l1, cv2.ROTATE_90_COUNTERCLOCKWISE)

    img = cv2.rotate(right1, cv2.ROTATE_90_CLOCKWISE)
    # start vertical devide image
    height = img.shape[0]
    width = img.shape[1]
    # Cut the image in half
    width_cutoff = width // 2
    r4 = img[:, :width_cutoff]
    r3 = img[:, width_cutoff:]
    # finish vertical devide image
    # rotate image to 90 COUNTERCLOCKWISE
    r4 = cv2.rotate(r4, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotate image to 90 COUNTERCLOCKWISE
    r3 = cv2.rotate(r3, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # save
    quadrants.append(l1)
    quadrants.append(r3)
    quadrants.append(l2)
    quadrants.append(r4)

 
    return quadrants

def cv_make_quadsofquads(img_quad):
    cv_img_oct=[]
    for oct_i in img_quad:
        cv_img_oct.extend(cv_make_quadrants(oct_i))
    return cv_img_oct


def stitch_parts(cv_img_oct,top1,top2,bottom1,bottom2):

    img_h1=cv2.hconcat([cv_img_oct[top1],cv_img_oct[top2]])
    img_h2=cv2.hconcat([cv_img_oct[bottom1],cv_img_oct[bottom2]])

    return cv2.vconcat([img_h1,img_h2])

def stitchsatellite(names):
    vrt1 = gdal.BuildVRT("top.vrt",[names[0],names[1],names[2]])
    gdal.Translate("top.jp2",vrt1)

    vrt2 = gdal.BuildVRT("middle.vrt",[names[3],names[4],names[5]])
    gdal.Translate("middle.jp2",vrt2)

    vrt3 = gdal.BuildVRT("bottom.vrt",[names[6],names[7],names[8]])
    gdal.Translate("bottom.jp2",vrt3)

    top_files =['top.jp2','middle.jp2','bottom.jp2']

    vrtd = gdal.BuildVRT(str("sat.vrt"),top_files)
    gdal.Translate(str("sat.jp2"),vrtd)

    return "sat.jp2"



def stitched_outputs(img):
    cv_img_quad=cv_make_quadrants(cv2.imread(img))
    
    cv_img_oct=cv_make_quadsofquads(cv_img_quad)


    list_cv=[cv_img_oct[0]] #top left corner 

    diag_top=stitch_parts(cv_img_oct,1,4,3,6) # area between the top diagonals
    list_cv.append(diag_top)

    list_cv.append(cv_img_oct[5])   # top right corner


    diag_left=stitch_parts(cv_img_oct,1,4,3,6) # Left area between diagonals
    list_cv.append(diag_left)

    center=stitch_parts(cv_img_oct,3,6,9,12)  #center image
    list_cv.append(center)

    diag_right=stitch_parts(cv_img_oct,6,7,12,13) # right area between diagonals
    list_cv.append(diag_right)

    list_cv.append(cv_img_oct[15])   # bottom right corner

    diag_bottom=stitch_parts(cv_img_oct,9,12,11,14) # bottom area between diagonals
    list_cv.append(diag_bottom)


    list_cv.append(cv_img_oct[10])   # bottom left corner


    return list_cv






    
