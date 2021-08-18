import os
import shutil
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import Functions as helper
import image_helper as imghelper
from osgeo import gdal, osr
from datetime import date
import sys
import logging
import pandas as pd
from zipfile import ZipFile
import wget

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    macCoords = pd.read_csv('Data/MacConnellCoords.csv')
    # Set spatial reference:
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(26986)
    logging.info('Spatial reference set to EPSG:26986.')
    # generate SIFT
    sift = cv2.SIFT_create()
    # brute force feature matching of descriptors to match keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    filenames = []
    refFn = None
    x_scaled = 600
    y_scaled = 600
    maxGCPs = 25
    # read config and set parameters
    mac = helper.read_yaml('config.yaml')['MACCONNELL']
    if mac['USE'] is True:
        inp = mac['INPUT']
        out = mac['OUTPUT']

        refCode = inp['REF_CODE']
        macPath = inp['MAC_PATH']
        satPath = inp['SAT_PATH']
        initials = inp['INITIALS']
        maxGCPs = inp['MAX_GCPS']
        avail_sat=inp['SAT_AVAILABLE']
        GCPOut = out['GCP_OUTPUT_LOC']
        refOut = out['REF_OUTPUT_LOC']
        target_epsg = out['TARGET_EPSG']

        GCP_fname = f'{GCPOut}{refCode}_GCPs_{initials}_{date.today().strftime("%Y%m%d")}.txt'
        if os.path.exists(GCP_fname):
                logging.warning(f'{GCP_fname} already exists...Overwriting.')
        else:
            logging.info(f'{GCP_fname} created.')
        orig_fn = None
        found = False
        for fn in macCoords['Filename']:
            if refCode == fn.split('\\')[-1].split('-')[-3] and fn.split('\\')[-1][-3:] == 'tif':
                temp = fn[30:].replace('\\', '/')
                orig_fn = f'{macPath}{temp}'
                name=orig_fn.split("/")[-1][:-4]
                orig_fn = f'{fn}'
                if found:
                    logging.error(f'Found more than one TIFF file with code {refCode} in folder {macPath}.')
                    #sys.exit(1)
                found = True
        if orig_fn is None:
            logging.error(f'Could not find an unreferenced MacConnell image with code {refCode} in folder {macPath}.')
            sys.exit(1)
        output_fn = f'{refOut}{name}.reference.tif'
        logging.info(f'The MacConnell Image is: {orig_fn}')
        filenames.append((orig_fn, output_fn, GCP_fname))

        tile_nameslist=[]
        if avail_sat:
            df = pd.read_csv('Data/Mapping.csv')
            tiles = df.loc[df['MacFile'] == orig_fn] \
                [['Tile1', 'Tile2', 'Tile3', 'Tile4', 'Tile5', 'Tile6', 'Tile7', 'Tile8', 'Tile9']]
            

        else:
            #Downloading the Satellite tiles on the fly
            link="http://download.massgis.digital.mass.gov/images/coq2019_15cm_jp2/"

            df = pd.read_csv('Data/Mapping.csv')
            tiles = df.loc[df['MacFile'] == orig_fn] \
                [['Tile1', 'Tile2', 'Tile3', 'Tile4', 'Tile5', 'Tile6', 'Tile7', 'Tile8', 'Tile9']]
            for p in tiles.values[0]:
                name=str(p)[-15:]
                name_str=name.replace('.jp2','.zip')
                tile_nameslist.append(name_str)

            logging.info("Started Downloading the satellite tiles")
            count=0

            for file in tile_nameslist:
                file_name=file.split('.')[0]
                os.system("mkdir "+str(satPath)+file_name )
                wget.download(link+file,str(satPath)+file_name+"\\"+file)

                with ZipFile(str(satPath)+file_name+"\\"+file, 'r') as zipObj:
                    zipObj.extractall(str(satPath)+file_name+"\\")
                os.system("del "+str(satPath)+file_name+"\\"+file)
                tiles.values[0][count]=str(satPath)+file_name+"\\"+file_name+".jp2"
                count+=1


    else:
        inp = helper.read_yaml('config.yaml')['OTHER']['INPUT']
        out = helper.read_yaml('config.yaml')['OTHER']['OUTPUT']
        x_scaled = inp['X_SCALE']
        y_scaled = inp['Y_SCALE']
        # set spatial reference
        sr.ImportFromEPSG(inp['EPSG'])
        logging.info(f'Spatial reference set to EPSG:{inp["EPSG"]}.')
        refFn = inp['MAC_PATH']
        avail_sat=inp['SAT_AVAILABLE']
        satPath = inp['SAT_PATH'] 
        maxGCPs = inp['MAX_GCPS']
        DPI = inp['DPI']
        ransacThresh = inp['RANSAC_THRESHOLD']
        blockSize = inp['BLOCK_SIZE']
        sobK = inp['SOBEL_K']
        harK = inp['HARRIS_K']
        GCPOut = out['GCP_OUTPUT_LOC']
        refOut = out['REF_OUTPUT_LOC']
    

        
    #dividing the image to 9 parts
    list_cv= imghelper.stitched_outputs(orig_fn)
    
    # define the new image scale
    x_scaled = 600
    y_scaled = 600


    #Merging the 9 satellite tiles into single tile
    logging.info(f'Merging all the 9 satellite tiles to single tile')
    merged_tile=imghelper.stitchsatellite(tiles.values[0])
    mod_img = gdal.Open(merged_tile, gdal.GA_ReadOnly)
    mod_width=mod_img.RasterXSize
    mod_height=mod_img.RasterXSize
    offset_dst_corners=[[0,0],[mod_width/3,0],[2*mod_width/3,0],
                        [0,mod_height/3],[mod_width/3,mod_height/3],[2*mod_width/3,mod_height/3],
                        [0,2*mod_height/3],[mod_width/3,2*mod_height/3],[2*mod_width/3,2*mod_height/3]]

    mac_img=gdal.Open(orig_fn)
    mac_width=mac_img.RasterXSize
    mac_height=mac_img.RasterYSize
    #adjusting offsets
   
    offset_src_corners = [[0,0],[mac_width/4,0],[3*mac_width/4,0],
                        [0,mac_height/4],[mac_width/4,mac_height/4],[mac_width/2,mac_height/4],
                        [0,3*mac_height/4],[mac_width/4,mac_height/2],[3*mac_width/4,3*mac_height/4]]

    with open(GCP_fname, 'w') as f:

        # Create a copy of the original file and save it as the output filename:
        shutil.copy(orig_fn, output_fn)
        # Open the output file for writing
        ds = gdal.Open(output_fn, gdal.GA_Update)
        logging.info(f'{output_fn} created.')

        #creating empty lists to combine matches across all combinations of Macconnell to satellite
        combined_matches=[]
        #combined_matches_mask=[]
        combined_src=[]
        combined_dst=[]


        total, n = 0, 0
        for q, tile in enumerate(tiles.values[0]):

            if tile is not None:
                logging.info(f'Started tile {q+1}')
                # get corresponding quadrant
                mac = list_cv[q]

                # read mod
                mod = cv2.imread(tile)
                # store scaling ratio
                mac_resize = [x_scaled / mac.shape[0], y_scaled / mac.shape[1]]
                mod_resize = [x_scaled / mod.shape[0], y_scaled / mod.shape[1]]
                
                # resize the images
                mac = cv2.resize(mac, (x_scaled, y_scaled))
                mod = cv2.resize(mod, (x_scaled, y_scaled))
                
                #preprocess the images and 
                grayMac,grayMod=helper.preprocess_images(mac,mod)
                logging.debug('Images preprocessed.')

                # generate keypoints from Harris Corner Detection
                keypointsMac, keypointsMod = helper.compute_harris_corner_keypoints(grayMac, grayMod)
                logging.debug('Keypoints detected with Harris Corner Detection.')
                # generate SIFT
                sift = cv2.SIFT_create()

                # generate keypoints and their descriptors
                descriptorsMac = sift.compute(grayMac, keypointsMac)[1]
                descriptorsMod = sift.compute(grayMod, keypointsMod)[1]
                logging.debug('Descriptors computed with SIFT.')

                # brute force feature matching of descriptors to match keypoints
                bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
                #                              query            train
                matches = sorted(bf.match(descriptorsMac, descriptorsMod), key=lambda x: x.distance)

                top_matches = matches[:-int((len(matches) / 4) * 3)]
                combined_matches.extend(top_matches)

                # take all the points present in top_matches, find src and dest pts
                src_pts = np.float32([keypointsMac[m.queryIdx].pt for m in top_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypointsMod[m.trainIdx].pt for m in top_matches]).reshape(-1, 1, 2)

                # apply RANSAC
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                # value is 1 for the inliers from RANSAC
                matchesMask = mask.ravel().tolist()

                scaled_src=[]
                scaled_dst=[]
                index=[]
                id=0
                for i in range(0,len(src_pts)):
                    if id==maxGCPs:
                        break
                    if matchesMask[i]==1:
                        id+=1#print(src_pts[i][0])
                        index.append(i)
                        scaled_src.append([float(int(i / j)+x) for i, j,x in zip(src_pts[i][0], mac_resize,offset_src_corners[q])])
                        scaled_dst.append([float(int(i / j)+x) for i, j,x in zip(dst_pts[i][0], mod_resize,offset_dst_corners[q])])


                id=0
                for m in index:
                    keypointsMac[top_matches[m].queryIdx].pt=(scaled_src[id][0],scaled_src[id][1])
                    keypointsMod[top_matches[m].trainIdx].pt=(scaled_dst[id][0],scaled_dst[id][1])
                    id+=1
                

                combined_src.extend(scaled_src)
                combined_dst.extend(scaled_dst)
                #combined_matches_mask.extend(matchesMask)

        top_matches = sorted(combined_matches, key=lambda x: x.distance)
        #final_matches_mask = [x for _,x in sorted(zip(combined_matches, combined_matches_mask), key=lambda pair: pair[0].distance)]
        
        final_src = combined_src
        final_dst = combined_dst

        gcps=[]
        logging.info(f'Writing GCPs to {GCP_fname} and applying them to {output_fn}...')
        ds=gdal.Open(r'sat.jp2')
        prj=ds.GetProjection()
        srs=osr.SpatialReference(wkt=prj)

        # For all the matches in ransac find the pixel to coord for mac and mod
        for i in range(0, len(final_src)):
            # Converting the pixel to coordinates in corresponding  for modern img
            CRSXmod,CRSYmod = helper.pixel2coord(mod_img,final_dst[i][0],final_dst[i][1])

            # Converting Modern image coordinates to Maconell Image 
            mac_latmod, mac_lonmod = helper.mod2mac(CRSXmod, CRSYmod,srs.GetAttrValue('AUTHORITY',1),target_epsg)

            # add the GCP
            gcps.append(gdal.GCP(mac_lonmod, mac_latmod, 0, int(final_src[i][0]), int(final_src[i][1])))


            # calculate the inch coords for a GCP based on the pixel coords
            ovX, ovY = helper.pixel2inches(cv2.imread(orig_fn), final_src[i][0], final_src[i][1])
            # write the GCPs to the text file
            ovX = format(np.round(ovX, 8), ".8f")
            ovY = format(np.round(ovY, 8), ".8f")

            f.write(f'{ovX}\t{ovY}\t{np.round(mac_latmod, 8)}\t{np.round(mac_lonmod, 8)}\n')
        logging.debug('All GCPs written and applied.')

            
    # Apply the GCPs to the open output file:
    ds.SetGCPs(gcps, sr.ExportToWkt())
    srs = sr.ExportToWkt()
    gdal.Warp(output_fn, ds, options=gdal.WarpOptions(polynomialOrder=2, srcSRS=srs, dstSRS=srs))
    logging.info('Completed!')
    os.system('del *.vrt')
