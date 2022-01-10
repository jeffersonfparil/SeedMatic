### import libraries
import os
import numpy as np
import pandas as pd
import skimage
from skimage import io
from skimage.feature import canny
import cv2
import matplotlib; matplotlib.use('Agg') ### for headless execution
# import matplotlib; matplotlib.use('TkAgg') ### for plotting in interactive mode
from matplotlib import pyplot as plt

### plot image
def fun_image_show(image, cmap=['viridis','gray','Pastel1'][0], title="", width=5, height=5, dpi=100.0, show_axes=True, save=False):
    ######################################
    ### TEST:
    # image = io.imread('res/Lr-germination-5.JPG')
    # fun_image_show(image)
    # fun_image_show(image[:,:,0])
    # fun_image_show(image[:,:,0], cmap="Pastel1")
    # fun_image_show([io.imread('res/Lr-germination-5.JPG'), io.imread('res/Lr-germination-6.JPG')])
    # fun_image_show([io.imread('res/Lr-germination-5.JPG')[:,:,0], io.imread('res/Lr-germination-6.JPG')[:,:,1]])
    # fun_image_show([io.imread('res/Lr-germination-5.JPG')[:,:,0], io.imread('res/Lr-germination-6.JPG')[:,:,1]], cmap=["gray", "Pastel2"])
    ######################################
    if isinstance(image, np.ndarray):
        image = [image]
    ncol = int(np.ceil(np.sqrt(len(image))))
    nrow = int(np.ceil(len(image) / ncol))
    fig = plt.figure(figsize=(width*ncol, height*nrow), dpi=dpi)
    for i in range(len(image)):
        fig.add_subplot(nrow, ncol, i+1)
        if isinstance(title, list):
            plt.title(title[i], fontsize=width)
        else:
            plt.title(title, fontsize=width)
        nDims = len(image[i].shape)
        if nDims==3:
            plt.imshow(image[i])
        elif nDims==2:
            try:
                plt.imshow(image[i], cmap=cmap[i])
            except:
                plt.imshow(image[i], cmap=cmap)
        else:
            return(1)
        if show_axes == False:
            plt.axis("off")
    if save:
        plt.ioff()
    else:
        try:
            plt.show(block=False)
        except:
            plt.show()

### flatten RGB images, i.e. transform the 3D array into a 2D array by grayscale transformation of extracting one of the 3 colour channels
def fun_image_flatten(image, vec_coef=[0.2125, 0.7154, 0.0721], plot=False):
    ########################################################################
    ### TEST:
    # image = io.imread('res/Lr-germination-5.JPG')
    # # vec_coef = [0.2125, 0.7154, 0.0721] ### grayscale transformation
    # # vec_coef = [1.0, 0.0, 0.0] ### extract red channel
    # vec_coef = [0.0, 1.0, 0.0] ### extract green channel
    # plot = True
    # fun_image_flatten(image, vec_coef, plot)
    ########################################################################
    ### for each pixel: multiply each colour channel value with the coefficients,
    ### and take the sum
    X = np.sum(image *  vec_coef, 2)
    # X = np.uint8(np.sum(image *  vec_coef, 2)) ### converting image array to uint8 reduces the sensitity!
    if plot: fun_image_show(X)
    return(X)

### multiple erosions or dilation operations
def fun_multi_erosion_dilation(image, type=["erosion", "dilation"][0], times=1):
    X = np.copy(image)
    for i in range(times):
        if type=="erosion":
            if len(X.shape)==2:
                X = skimage.morphology.erosion(X)
            else:
                for i in range(3): X[:,:,i] = skimage.morphology.erosion(X[:,:,i])
        elif type=="dilation":
            if len(X.shape)==2:
                X = skimage.morphology.dilation(X)
            else:
                for i in range(3): X[:,:,i] = skimage.morphology.dilation(X[:,:,i])
        else:
            print("Please choose type: 'erosion' or 'dilation'.")
    return(X)

### detect the circular plate in the middle of the image
def fun_detect_plate(image, plate_shape="round", vec_radius_or_height_limit=[1000,2000], vec_width_limit=None, scale_factor=0.05, n_steps=5, central_round_plate=True, plot=False, debug=False):
    #################################################################
    ### TEST:
    # # image = io.imread('res/At-germination-no-marker-1.JPG')
    # # image = io.imread('res/At-germination-with-marker-2.JPG')
    # plate_shape = "round"
    # vec_radius_or_height_limit = [1000, 2000]
    # image = io.imread('res/Lr-germination-5.JPG')
    # image = io.imread('res/Lr-germination-5.JPG')
    # plate_shape = "rectangular"
    # vec_radius_or_height_limit = [1500, 2000]
    # vec_width_limit = None
    # scale_factor = 0.05
    # n_steps = 5
    # central_round_plate = False
    # plot = True
    ################################################################
    ### flatten image to grayscale
    flattened = fun_image_flatten(image)
    ### rescale to reduce image resolution and decrease computation times for edge detection and hough circle or polygon detection
    rescaled = skimage.transform.rescale(flattened, scale_factor, anti_aliasing=False)
    ### initialise plate image
    plate = np.copy(image)
    ##################################################################
    if plate_shape=="rectangular":
        ##############################################################
        ### rectangular plates
        if vec_width_limit == None:
            vec_width_limit = vec_radius_or_height_limit
        height_min = np.round(np.array(vec_radius_or_height_limit)*scale_factor).astype(int)[0]
        height_max = np.round(np.array(vec_radius_or_height_limit)*scale_factor).astype(int)[1]
        width_min = np.round(np.array(vec_width_limit)*scale_factor).astype(int)[0]
        width_max = np.round(np.array(vec_width_limit)*scale_factor).astype(int)[1]
        edges = cv2.adaptiveThreshold(skimage.util.img_as_ubyte(np.uint8(rescaled)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
        contours,_=cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        vec_x = []; vec_y = []
        vec_w = []; vec_h = []
        vec_area = []
        for i in range(len(contours)):
                # i = 123
                cnt = contours[i]
                x,y,w,h = cv2.boundingRect(cnt)
                test = ( (h > height_min) and (w < height_max) and
                        (w > width_min) and (h < width_max) )
                if test:
                    vec_x.append(x); vec_y.append(y)
                    vec_w.append(w); vec_h.append(h)
                    vec_area.append(w*h)
                    # cnt_image = cv2.drawContours(np.zeros_like(np.zeros((rescaled.shape[0:2]))), contours, i, color=255, thickness=-1)
                    # fun_image_show(cv2.rectangle(rescaled, (x, y), (x+w, y+h), (255, 255, 255), -1))
        ### exit with unchanged image if we did not detecrt a matching contour
        if len(vec_area)==0:
            print("No matching plate contours found!")
            print("Please consider modifying the plate radius or width and/or height limits.")
            plate = np.copy(image)
            mask = np.ones(plate.shape, dtype=np.uint8)
            x = int(plate.shape[1]/2)
            y = int(plate.shape[0]/2)
            w = int(plate.shape[1])
            h = int(plate.shape[0])
            return(plate, mask, x, y, w, h)
        ### find the rectangle closest to the centre and with the smallest area
        vec_x = np.array(vec_x); vec_y = np.array(vec_y)
        vec_w = np.array(vec_w); vec_h = np.array(vec_h)
        vec_area = np.array(vec_area)
        vec_dx = abs((vec_x+(vec_w/2)) - (edges.shape[1]/2))
        vec_dy = abs((vec_y+(vec_h/2)) - (edges.shape[0]/2))
        vec_minimise = vec_dx + vec_dy + vec_area
        idx = np.arange(0, len(vec_x), 1)[np.min(vec_minimise)==vec_minimise][0]
        ### create the mask
        x = vec_x[idx]; y = vec_y[idx]
        w = vec_w[idx]; h = vec_h[idx]
        mask = np.zeros(rescaled.shape, dtype=np.uint8)
        mask = cv2.rectangle(mask, (x, y), (x+w, y+h), (255, 255, 255), -1)
        mask = mask > 0
        ### rescale the mask into the original size
        mask = skimage.transform.rescale(mask, 1/scale_factor, anti_aliasing=False)
        mask = mask.astype(bool)
        ### isolate the plate
        plate[np.invert(mask),:] = [0,0,0]
        ##############################################################
    elif plate_shape=="round":
        ##############################################################
        ### round plates
        ### find edges
        edges = canny(rescaled, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
        ### compute the range or circle radi to test
        radius_min = np.round(np.array(vec_radius_or_height_limit)*scale_factor).astype(int)[0]
        radius_max = np.round(np.array(vec_radius_or_height_limit)*scale_factor).astype(int)[1]
        step = ((radius_max - radius_min)/n_steps).astype(int)
        vec_radii = np.arange(radius_min, radius_max, step)
        ### detect circles
        circles = skimage.transform.hough_circle(edges, vec_radii)
        ### identify the 5 best fitting circles
        n_top_circles = 5
        accums, vec_x, vec_y, vec_r = skimage.transform.hough_circle_peaks(circles, vec_radii, total_num_peaks=n_top_circles)
        #### plot the top 5 circles
        if debug:
            for i in range(n_top_circles):
                ### create the mask
                try:
                    vec_rows, vec_cols = skimage.draw.disk((vec_y[i], vec_x[i]), vec_r[i], shape=rescaled.shape)
                except:
                    vec_rows, vec_cols = skimage.draw.circle(vec_y[i], vec_x[i], vec_r[i], rescaled.shape)
                mask = np.zeros(rescaled.shape, dtype=bool)
                mask[vec_rows, vec_cols] = True
                ### rescale the mask into the original size
                mask = skimage.transform.rescale(mask, 1/scale_factor, anti_aliasing=False)
                mask = mask.astype(bool)
                ### isolate the plate
                plate = np.copy(image)
                plate[np.invert(mask),:] = [0,0,0]
                fun_image_show(plate)
        if central_round_plate:
            ### find the circle closest to the centre and with the smallest radius
            vec_dx = abs(vec_x - (edges.shape[1]/2))
            vec_dy = abs(vec_y - (edges.shape[0]/2))
            vec_minimise = vec_dx + vec_dy + vec_r
            idx = np.arange(0, n_top_circles, 1)[np.min(vec_minimise)==vec_minimise][0]
        else:
            idx = 0
        ### create the mask
        try:
            vec_rows, vec_cols = skimage.draw.disk((vec_y[idx], vec_x[idx]), vec_r[idx], shape=rescaled.shape)
        except:
            vec_rows, vec_cols = skimage.draw.circle(vec_y[idx], vec_x[idx], vec_r[idx], rescaled.shape)
        mask = np.zeros(rescaled.shape, dtype=bool)
        mask[vec_rows, vec_cols] = True
        x = int(vec_x[idx]/scale_factor)
        y = int(vec_y[idx]/scale_factor)
        w = int(vec_r[idx]/scale_factor) ### radius
        h = w ### radius
        ### rescale the mask into the original size
        mask = skimage.transform.rescale(mask, 1/scale_factor, anti_aliasing=False)
        mask = mask.astype(bool)
        ### isolate the plate
        plate[np.invert(mask),:] = [0,0,0]
        ##############################################################
    else:
        print("Uknown plate shape: " + plate_shape + ".")
        print("Please select: 'round' or 'rectangular'.")
    ### plot
    if plot:
        fun_image_show([image, plate])
    ### output
    return(plate, mask, x, y, w, h)

### crop to include only the plate
def fun_image_crop(image, x0, x1, y0, y1, plot=False):
    ############################
    ### TEST:
    # image = io.imread('res/Lr-germination-5.JPG')
    # x0 = 1200
    # x1 = 3100
    # y0 = 1900
    # y1 = 3800
    # plot=True
    # fun_image_crop(image, x0, x1, y0, y1, plot)
    ############################
    X = np.copy(image[y0:y1, x0:x1]) ### where ys are the rows and xs are the columns
    if plot: fun_image_show(X)
    if X.size == 0:
        print("Incorrect cropping coordinates! Returning uncropped image.")
        return(image)
    return(X)

### colour correction or white-balancing
def fun_colour_correct(image, vec_RGB_mode_expected=[45, 55, 75], remove_black=True, plot=False):
    ##############################
    ### TEST:
    # fname = "res/At-germination-with-marker-2.JPG"
    # plate, mask_plate, y, x, r = fun_detect_plate(io.imread(fname))
    # image = fun_image_crop(plate, x0=x-r, x1=x+r, y0=y-r, y1=y+r)
    # vec_RGB_mode_expected = [45, 55, 75]
    # remove_black = True
    ##############################
    ### calculate the frequencies of each pixel value across the 3 channels
    vec_val, vec_freq = np.unique(image.reshape(-1, 3), axis=0, return_counts=True)
    ### remove black if we are masked out unwantend areas as black pixels
    if remove_black:
        idx = np.sum(vec_val!=[0,0,0], axis=1)==0
        vec_freq[idx] = [0]
    ### identify the most common pixel values at the R,G, and B channels
    vec_RGB_mode_observed = vec_val[np.argmax(vec_freq)]
    ### correction factor
    correction_factor = vec_RGB_mode_expected / vec_RGB_mode_observed
    ### convert image to range from 0 to 1 then multiply the correction factor then clip to range from 0 to 1 then convert to uint8 ranging from 0 to 255
    X = skimage.img_as_ubyte((image/255 * correction_factor).clip(0, 1))
    ### plot
    if plot:
        fun_image_show([image, X])
    ### output
    return(X, vec_RGB_mode_observed)

### compute the segmentation threshold using one of 7 algorithms
def fun_threshold_compute(flatImage, algo=["isodata", "li", "mean", "minimum", "otsu", "triangle", "yen"][0]):
    ##########################################################################################################
    ### TEST:
    # image = io.imread('res/Lr-germination-5.JPG')
    # flatImage = fun_image_flatten(image, [0.0, 1.0, 0.0], False)
    # fun_threshold_compute(flatImage)
    # fun_threshold_compute(flatImage, algo="triangle")
    ##########################################################################################################
    if (algo=="isodata"):
        try:
            out = skimage.filters.threshold_isodata(flatImage)
        except:
            out = 0.0
    elif (algo=="li"):
        try:
            out = skimage.filters.threshold_li(flatImage)
        except:
            out = 0.0
    elif (algo=="mean"):
        try:
            out = skimage.filters.threshold_mean(flatImage)
        except:
            out = 0.0
    elif (algo=="minimum"):
        try:
            out = skimage.filters.threshold_minimum(flatImage)
        except:
            out = 0.0
    elif (algo=="otsu"):
        try:
            out = skimage.filters.threshold_otsu(flatImage)
        except:
            out = 0.0
    elif (algo=="triangle"):
        try:
            out = skimage.filters.threshold_triangle(flatImage)
        except:
            out = 0.0
    elif (algo=="yen"):
        try:
            out = skimage.filters.threshold_yen(flatImage)
        except:
            out = 0.0
    else:
        return(1)
    return(out)

### select the best threshold among the thresholding algorithms
### given a maximum detection proportion threshold: maxFrac
def fun_threshold_optim(flatImage, maxFrac=0.10, plot=False):
    #############################################
    ### TEST:
    # image = io.imread('res/Lr-germination-5.JPG')
    # flatImage = fun_image_flatten(image, [0.0, 1.0, 0.0], False)
    # maxFrac = 0.10
    # plot = True
    # fun_threshold_optim(flatImage, maxFrac, plot)
    #############################################
    nElements = flatImage.shape[0] * flatImage.shape[1]
    vec_algos = np.array(["isodata", "li", "mean", "minimum", "otsu", "triangle", "yen"])
    vec_fracRetained = np.array([])
    for algo in vec_algos:
        x = fun_threshold_compute(flatImage, algo)
        fracRetained = np.sum(flatImage >= x) / nElements
        vec_fracRetained = np.append(vec_fracRetained, fracRetained)
    idx = vec_fracRetained < maxFrac
    ### if none of the algorithms pass the maxFrac threshold then use the algorithm which minimised the fraction of pixels retained
    if np.sum(idx) == 0: idx = vec_fracRetained == np.min(vec_fracRetained)
    ### choose the least conservative algorithm
    algo = vec_algos[idx][vec_fracRetained[idx]==np.max(vec_fracRetained[idx])][0]
    threshold = fun_threshold_compute(flatImage, algo)
    if plot:
        skimage.filters.thresholding.try_all_threshold(flatImage)
        try:
            plt.show(block=False)
        except:
            plt.show()
    return(threshold, algo)

### detect the seedlings from coloured photographs
### i.e. the seeds/cotyledon/s, hypocotyl/shoots, and radicle/roots
def fun_mask_seedlings(image, dic_flattenTypes={"grayscale":[0.2125, 0.7154, 0.0721], "red":[1.0, 0.0, 0.0], "green":[0.0, 1.0, 0.0]}, dic_fracThesholds={"grayscale":0.10, "red":0.10, "green":0.10}, blur_threshold=2000, plot=False):
    ########################################################################
    ### TEST:
    # # rawImage = io.imread('res/Lr-germination-5.JPG')
    # # x0, x1, y0, y1, crop_labeled = find_marker_positions(rawImage)
    # rawImage = io.imread('res/Lr-germination-5.JPG')
    # marker_vec_range_hue = [0.51, 0.66] ## light-blue post-it note markers
    # marker_vec_range_sat = [0.00, 1.00]
    # marker_vec_range_val = [0.75, 1.00]
    # x0, x1, y0, y1, crop_labeled = find_marker_positions(rawImage, marker_vec_range_hue, marker_vec_range_sat, marker_vec_range_val, marker_min_area, plot=False)
    # image = fun_image_crop(rawImage, x0, x1, y0, y1)
    # image = fun_remove_markers(image, marker_vec_range_hue, marker_vec_range_sat, marker_vec_range_val)
    # dic_flattenTypes = {"grayscale":[0.2125, 0.7154, 0.0721], "red":[1.0, 0.0, 0.0], "green":[0.0, 1.0, 0.0]}
    # dic_fracThesholds = {"grayscale":0.10, "red":0.10, "green":0.10}
    # blur_threshold = 2000
    # plot = True
    # fun_mask_seedlings(image, dic_flattenTypes, dic_fracThesholds, plot)
    ########################################################################
    ### define the mask using various flattening types and their corresponsing fractional thresholds
    mat_mask = np.zeros(image.shape[0:2], dtype=bool)
    for flattenType in dic_flattenTypes:
        ### flatten image
        vec_coef = dic_flattenTypes[flattenType]
        flatImage = fun_image_flatten(image, vec_coef)
        ### define seed detection thresholds
        maxFrac = dic_fracThesholds[flattenType]
        threshold, _ = fun_threshold_optim(flatImage, maxFrac)
        ### generate the mask
        mask = flatImage >= threshold
        ### update the mask matrix
        mat_mask += mask
        # ### test each mask
        # X = np.copy(image); X[np.invert(mask)] = 0; fun_image_show(X)
    ### dilate if blurry else erode to remove artefacts
    var_laplace = cv2.Laplacian(fun_image_flatten(image), cv2.CV_64F).var()
    if var_laplace < blur_threshold:
        mat_mask = np.uint8(fun_multi_erosion_dilation(mat_mask, type="dilation", times=1)*255)
    else:
        ### remove tiny artefacts by erosion
        mat_mask = np.uint8(fun_multi_erosion_dilation(mat_mask, type="erosion", times=1)*255)
    ### mask-out the background-ish (artefacts may still be present)
    X = np.copy(image)
    X[mat_mask==0] = 0
    ### visualise masked image
    if plot: fun_image_show(X)
    ### return the masked image
    return(X, mat_mask, var_laplace)

### detect the green shoots and calculate the total projected shoot area
def fun_detect_green_shoots(seedlings, shoot_area_limit=[100, 10000], vec_green_hue_limit=[0.15, 0.35], vec_green_sat_limit=[0.3, 1.0], vec_green_val_limit=[0.3, 1.0], plot=False, plot_masks=False, shoot_axis_ratio_min_diff=0.3, debug=False):
    ##################################################################
    ### TEST:
    # # rawImage = io.imread('res/Lr-germination-5.JPG')
    # rawImage = io.imread('res/Lr-germination-6.JPG')
    # marker_vec_range_hue = [0.51, 0.66] ## light-blue post-it note markers
    # marker_vec_range_sat = [0.00, 1.00]
    # marker_vec_range_val = [0.75, 1.00]
    # plate, mask, x, y, w, h = fun_detect_plate(rawImage, plate_shape="round")
    # image = fun_image_crop(plate, x0=x-w, x1=x+w, y0=y-h, y1=y+h)
    # dic_flattenTypes = {"grayscale":[0.2125, 0.7154, 0.0721], "red":[1.0, 0.0, 0.0], "green":[0.0, 1.0, 0.0]}
    # dic_fracThesholds = {"grayscale":0.10, "red":0.10, "green":0.10}
    # seedlings, mask_seedlings, var_laplace = fun_mask_seedlings(image, dic_flattenTypes, dic_fracThesholds)
    # # fun_image_show([image, seedlings])
    # shoot_area_limit=[500, 10000]
    # vec_green_hue_limit = [60/360, 140/360]
    # vec_green_sat_limit = [0.25, 1.00]
    # vec_green_val_limit = [0.25, 1.00]
    # plot = True
    # plot_masks = False
    # shoot_axis_ratio_min_diff=0.3
    # ### histograms
    # plt.hist(np.ravel(seedlings[:, :, 0]))
    # plt.show()
    # plt.hist(np.ravel(seedlings[:, :, 1]))
    # plt.show()
    # plt.hist(np.ravel(seedlings[:, :, 2]))
    # plt.show()
    ##################################################################
    ### derive the greeness flat image: green / (red+green+blue) from the seedlings image
    flatImage_greenness = np.uint8((seedlings[:,:,1] / (np.sum(seedlings,2) + 1e-10))*255)
    ### define the greenness threshold, i.e at 0.35 G/(R+G+B) ratio we deem the pixel green enough
    threshold_green = 0.35 * 255
    ### need to improve this greenness threshold to decrease false positive and false negatives!
    ### probably use the HSV colour space?
    ### create the greenness mask
    mask_greenness = flatImage_greenness >= threshold_green
    ### remove non-green promary and secondary colours using hue filtering
    ### using the range of +- 0.01 of toroidal [60, 180, 240, 300, 360]/360
    seedlings_HSV = skimage.color.rgb2hsv(seedlings)
    mask = np.uint8(mask_greenness)
    mask_shoots = np.uint8(np.zeros(mask.shape))
    ### extract shoots image
    shoots = np.uint8(np.zeros(seedlings.shape))
    # shoots = np.copy(seedlings)
    # shoots[mask==0] = 0
    ### copy the input seedlings image
    shoots_appended = np.copy(seedlings)
    ### extract each putative shoot and filter the most likely real shoots using:
    ###  minimum and maximum expected shoot area [DEFAULT: shoot_area_limit=[100, 10000]], as well as
    ### minimum green hue value [DEFAULT: 0.20] == yellow-greenish
    total_projected_shoot_area = 0.0
    contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ### @@@ where: mode =  cv2.RETR_TREE == 3: retrieves all of the contours and reconstructs a full hierarchy of nested contours. 
    ### @@@        method == cv2.CHAIN_APPROX_SIMPLE == 2: compresses horizontal, vertical, and diagonal segments and leaves only their end points. For example, an up-right rectangular contour is encoded with 4 points.
    ### sort contours by decreasing area, so that we don't have to iterate across all of them, i.e. once we hit our minimim aera threshold we break out of the for-loop
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    counter = 0
    min_green_hue = vec_green_hue_limit[0]; max_green_hue = vec_green_hue_limit[1]
    min_green_sat = vec_green_sat_limit[0]; max_green_sat = vec_green_sat_limit[1]
    min_green_val = vec_green_val_limit[0]; max_green_val = vec_green_val_limit[1]
    for i in range(len(contours)):
        # i = 123
        cnt = contours[i]
        moments = cv2.moments(cnt)
        area = moments["m00"] # area = cv2.contourArea(cnt)
        if area < shoot_area_limit[0]:
            break
        elif area > shoot_area_limit[1]:
            continue
        ### calculate mean hue, saturation, and value
        cnt_image = cv2.drawContours(np.zeros_like(np.zeros((seedlings.shape[0:2]))), contours, i, color=255, thickness=-1)
        # plt.hist(seedlings_HSV[cnt_image==255,0]); plt.show()
        idx = cnt_image==255
        mean_hue = np.round(np.mean(seedlings_HSV[idx, 0]),2)
        mean_sat = np.round(np.mean(seedlings_HSV[idx, 1]),2)
        mean_val = np.round(np.mean(seedlings_HSV[idx, 2]),2)
        ### fit an ellipse to the contour and calculate the ratio between the major and minor axes
        (x,y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(cnt)
        axis_ratio = minorAxisLength / (minorAxisLength + majorAxisLength)
        ### determine if the contour fits within the expected size, colour and rough shape
        test = ( (area >= shoot_area_limit[0]) and (area <= shoot_area_limit[1]) and
                 (mean_hue >= min_green_hue) and (mean_hue <= max_green_hue) and
                 (mean_sat >= min_green_sat) and (mean_sat <= max_green_sat) and
                 (mean_val >= min_green_val) and (mean_val <= max_green_val) and
                 (abs(0.5-axis_ratio) <= shoot_axis_ratio_min_diff) )
                #  (majorAxisLength <= shoot_area_limit[0]) )
        if debug:
            test = True
        if test:
            # print(str(majorAxisLength))
            # print(str(shoot_area_limit[1]))
            total_projected_shoot_area += area
            counter += 1
            ### append mask for shoots
            shoots[idx, :] = seedlings[idx, :]
            mask_shoots[idx] = 255
            ### append bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(shoots_appended, (x, y), (x+w, y+h), (255, 0, 0), 2)
            ### append text
            cv2.putText(shoots_appended, str(counter) + ": " + str(area) + "; Hue=" + str(mean_hue) + "; Saturation=" + str(mean_sat) + "; Value=" + str(mean_val), (x+int(w/2),y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    ### plot
    if plot:
        fun_image_show(shoots_appended)
    if plot_masks:
        fun_image_show([mask_greenness, shoots_appended, mask_shoots, shoots], title=["Greenness", "Original image with the detected shoots", "Shoots mask", "Shoots"])
    ### output
    return(shoots, mask_shoots, shoots_appended, total_projected_shoot_area, counter)

### count seeds and seedlings
def fun_detect_seeds_seedlings(seedlings, mask_seedlings, mask_shoots, seed_area_limit=[100, 5000], vec_seed_hue_limit=[0.10, 0.20], vec_seed_sat_limit=[0.50, 1.00], vec_seed_val_limit=[0.50, 1.00], seed_axis_ratio_min_diff=0.2, plot=False, debug=False):
    ##################################################################
    ### TEST:
    # fname = "/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-with-marker-29-51.jpg"
    # plot = True
    # marker_vec_range_hue = [0.51, 0.66] ## light-blue post-it note markers
    # marker_vec_range_sat = [0.00, 1.00]
    # marker_vec_range_val = [0.75, 1.00]
    # marker_min_area = 10000
    # dic_flattenTypes = {"grayscale":[0.2125, 0.7154, 0.0721], "red":[1.0, 0.0, 0.0], "green":[0.0, 1.0, 0.0]}
    # dic_fracThesholds = {"grayscale":0.20, "red":0.20, "green":0.20}
    # seed_area_limit=[100, 5000]
    # vec_green_hue_limit = [0.15, 0.35]
    # vec_green_sat_limit = [0.3, 1.0]
    # vec_green_val_limit = [0.3, 1.0]
    # rawImage = io.imread(fname)
    # rawImage = fun_white_balance(rawImage, "white_patch", perc=99)
    # x0, x1, y0, y1, crop_labeled = find_marker_positions(rawImage, marker_vec_range_hue, marker_vec_range_sat, marker_vec_range_val, marker_min_area, plot=False)
    # image = fun_image_crop(rawImage, x0, x1, y0, y1)
    # image = fun_remove_markers(image, marker_vec_range_hue, marker_vec_range_sat, marker_vec_range_val)
    # seedlings, mask_seedlings = fun_mask_seedlings(image, dic_flattenTypes, dic_fracThesholds, plot=False)
    # shoot_area_limit=[500, 10000]
    # vec_green_hue_limit = [60/360, 140/360]
    # vec_green_sat_limit = [0.25, 1.00]
    # vec_green_val_limit = [0.25, 1.00]
    # shoots, mask_shoots, shoots_appended, total_projected_shoot_area, counter = fun_detect_green_shoots(seedlings, shoot_area_limit=shoot_area_limit, vec_green_hue_limit=vec_green_hue_limit, vec_green_sat_limit=vec_green_sat_limit, vec_green_val_limit=vec_green_val_limit, plot=False, plot_masks=False, debug=False)
    # seed_area_limit = [100, 1000]
    # vec_seed_hue_limit = [50/360, 90/360]
    # vec_seed_sat_limit = [0.30, 1.00]
    # vec_seed_val_limit = [0.40, 1.00]
    # seed_axis_ratio_min_diff = 0.2
    ###############################################################
    ### clone the image in HSV colour space
    seedlings_HSV = skimage.color.rgb2hsv(seedlings)
    seeds = np.uint8(np.zeros(seedlings.shape))
    mask_seeds = np.uint8(np.zeros(seedlings.shape[0:2]))
    ### clone the seedlings image for appending the labels
    seeds_appended = np.copy(seedlings)
    ### remove shoots from the seedlings mask
    mask_seedlings_less_shoots = np.copy(mask_seedlings)
    mask_seedlings_less_shoots[mask_shoots==255] = 0.0
    ### extract putative seeds and seedlings contours
    contours, heirarchy = cv2.findContours(mask_seedlings_less_shoots, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ### sort contours by decreasing area, so that we don't have to iterate across all of them, i.e. once we hit our minimim aera threshold we break out of the for-loop
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    counter = 0
    for i in range(len(contours)):
        # i = 234
        cnt = contours[i]
        moments = cv2.moments(cnt)
        area = moments["m00"] # area = cv2.contourArea(cnt)
        if area < seed_area_limit[0]:
            break
        elif area > seed_area_limit[1]:
            continue
        ### mean HSV
        ### calculate mean hue for the ith contour
        cnt_image = cv2.drawContours(np.zeros_like(np.zeros((seedlings.shape[0:2]))), contours, i, color=255, thickness=-1)
        idx = cnt_image==255
        mean_hue = np.round(np.mean(seedlings_HSV[idx,0]),2)
        mean_sat = np.round(np.mean(seedlings_HSV[idx,1]),2)
        mean_val = np.round(np.mean(seedlings_HSV[idx,2]),2)
        try:
            (x,y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(cnt)
            axis_ratio = minorAxisLength/(minorAxisLength+majorAxisLength)
        except:
            axis_ratio = 0.0
        test = ( (area >= seed_area_limit[0]) and (area <= seed_area_limit[1]) and
                 (mean_hue >= vec_seed_hue_limit[0]) and (mean_hue <= vec_seed_hue_limit[1]) and
                 (mean_sat >= vec_seed_sat_limit[0]) and (mean_sat <= vec_seed_sat_limit[1]) and
                 (mean_val >= vec_seed_val_limit[0]) and (mean_val <= vec_seed_val_limit[1]) and
                 (abs(0.5-axis_ratio) <= seed_axis_ratio_min_diff) )
        if debug:
            test = True
        if test:
            counter += 1
            # add contour into the mask
            seeds[idx, :] = seedlings[idx, :]
            mask_seeds[idx] = 255
            ### append bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(seeds_appended, (x, y), (x+w, y+h), (255, 0, 0), 2)
            ### append text
            cv2.putText(seeds_appended, str(counter) + ": " + str(area) + "; Hue=" + str(mean_hue) + "; Saturation=" + str(mean_sat) + "; Value=" + str(mean_val), (x+int(w/2),y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    ### plot
    if plot:
        fun_image_show([seeds, seeds_appended, mask_seeds])
    ### output
    return(seeds, mask_seeds, seeds_appended, counter)

### shoot emergence frac
def fun_frac_shoot_emergence(fname, dir_output=".", write_out=True, plot_out=True, suffix_out="",
                     blur_threshold=2000,
                     plate_shape="round",
                     vec_plate_radius_or_height_limit=[1000,2000],
                     vec_plate_width_limit=None,
                     central_round_plate=True,
                     vec_RGB_mode_expected=[45, 55, 75],
                     dic_flattenTypes={"grayscale":[0.2125, 0.7154, 0.0721], "red":[1.0, 0.0, 0.0], "green":[0.0, 1.0, 0.0]},
                     dic_fracThesholds={"grayscale":0.10, "red":0.10, "green":0.10},
                     shoot_area_limit=[100, 10000],
                     vec_green_hue_limit=[0.15, 0.35],
                     vec_green_sat_limit=[0.3, 1.0],
                     vec_green_val_limit=[0.3, 1.0], 
                     seed_area_limit=[100, 5000],
                     vec_seed_hue_limit=[0.10, 0.20],
                     vec_seed_sat_limit=[0.50, 1.00],
                     vec_seed_val_limit=[0.50, 1.00],
                     shoot_axis_ratio_min_diff=0.3,
                     seed_axis_ratio_min_diff=0.2,
                     compute_areas = False,
                     debug=False):
    ##################################################################
    ### TEST:
    # # fname = "/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-with-marker-29-51.jpg"
    # # fname = "res/At-germination-with-marker-2.JPG"
    # # fname = "/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-with-marker-48-51.jpg"
    # # fname = "/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-with-marker-45-45.jpg"
    # # fname = "/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-with-marker-0-49.jpg"
    # # fname = "/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-no-marker-46-49.jpg"
    # # fname = "/home/jeff/Documents/seedGermCV/res/Arabidopsis/At-no-marker-46-47.jpg"
    # fname = "/home/jeff/Downloads/weedomics_large_files/1.c.2.-Avadex/Batch-A-20211012/TEST-seedGermCV-D10/1.c.2-PLATE_3-PHOTO_1-DAY_10.jpg"
    # fname = "/home/jeff/Downloads/weedomics_large_files/1.c.2.-Avadex/Batch-A-20211012/TEST-seedGermCV-D10/1.c.2-PLATE_17-PHOTO_1-DAY_08.jpg"
    # import matplotlib; matplotlib.use('TkAgg') ### for plotting in interactive mode
    # from matplotlib import pyplot as plt
    # dir_output="."
    # write_out = False
    # plot_out = False
    # suffix_out = ""
    # blur_threshold = 2000
    # plate_shape="round"
    # # vec_plate_radius_or_height_limit = [1000,2000]
    # vec_plate_radius_or_height_limit = [800,1500]
    # vec_plate_width_limit = None
    # central_round_plate = False
    # # vec_RGB_mode_expected = [45, 55, 75]
    # vec_RGB_mode_expected = None
    # dic_flattenTypes = {"grayscale":[0.2125, 0.7154, 0.0721], "red":[1.0, 0.0, 0.0], "green":[0.0, 1.0, 0.0]}
    # dic_fracThesholds = {"grayscale":0.20, "red":0.20, "green":0.20}
    # shoot_area_limit=[100, 1000000]
    # vec_green_hue_limit = [60/360, 140/360]
    # vec_green_sat_limit = [0.25, 1.00]
    # vec_green_val_limit = [0.25, 1.00]
    # seed_area_limit = [100, 1000000]
    # # vec_seed_hue_limit = [50/360, 115/360]
    # vec_seed_hue_limit = [0/360, 45/360]
    # # vec_seed_sat_limit = [0.25, 1.00]
    # vec_seed_sat_limit = [0.30, 1.0]
    # # vec_seed_val_limit = [0.40, 1.00]
    # vec_seed_val_limit = [0.00, 1.00]
    # shoot_axis_ratio_min_diff=0.5
    # seed_axis_ratio_min_diff=0.2
    # debug = False
    ###############################################################
    ### image extension name or filename suffix
    extension_name = fname.split(".")[-1]
    ### output filenames
    fname_base = os.path.basename(fname)
    if dir_output==".":
        dir_output = os.path.dirname(fname)
    if suffix_out=="":
        suffix_out=""
    else:
        suffix_out = "-" + suffix_out
    fname_out_csv = os.path.join(dir_output, fname_base.replace("." + extension_name, "-germination_data" + suffix_out + ".csv"))
    fname_out_jpg = os.path.join(dir_output, fname_base.replace("." + extension_name, "-segmentation" + suffix_out + ".jpg"))
    ### load image
    rawImage = io.imread(fname)
    ### isolate plate
    plate, mask_plate, x, y, w, h = fun_detect_plate(rawImage, plate_shape=plate_shape, vec_radius_or_height_limit=vec_plate_radius_or_height_limit, vec_width_limit=vec_plate_width_limit, scale_factor=0.05, n_steps=5, central_round_plate=central_round_plate, plot=False)
    ### crop to include only the plate
    x0 = x - w; y0 = y - h
    x1 = x + w; y1 = y + h
    plate = fun_image_crop(plate, x0, x1, y0, y1, plot=False)
    ### colour correction
    if vec_RGB_mode_expected is None:
        image = plate
    else:
        image, vec_RGB_mode_observed = fun_colour_correct(plate, vec_RGB_mode_expected=vec_RGB_mode_expected, remove_black=True, plot=False)
    # ### sharpen if blurry
    # var_laplace = cv2.Laplacian(fun_image_flatten(image), cv2.CV_64F).var()
    # sharpen = var_laplace < blur_threshold
    # if sharpen:
    #     image = fun_sharpen(image, k=5)
    ### detect seedlings
    seedlings, mask_seedlings, var_laplace = fun_mask_seedlings(image=image, dic_flattenTypes=dic_flattenTypes, dic_fracThesholds=dic_fracThesholds, blur_threshold=blur_threshold, plot=False)
    ### detect green shoots
    shoots, mask_shoots, shoots_labeled, shoot_area, n_shoots = fun_detect_green_shoots(seedlings=seedlings, shoot_area_limit=shoot_area_limit, vec_green_hue_limit=vec_green_hue_limit, vec_green_sat_limit=vec_green_sat_limit, vec_green_val_limit=vec_green_val_limit, shoot_axis_ratio_min_diff=shoot_axis_ratio_min_diff, debug=debug)
    ### detect seeds
    seeds, mask_seeds, seeds_labeled, n_seeds = fun_detect_seeds_seedlings(seedlings=seedlings, mask_seedlings=mask_seedlings, mask_shoots=mask_shoots, seed_area_limit=seed_area_limit, vec_seed_hue_limit=vec_seed_hue_limit, vec_seed_sat_limit=vec_seed_sat_limit, vec_seed_val_limit=vec_seed_val_limit, seed_axis_ratio_min_diff=seed_axis_ratio_min_diff, debug=debug)
    # ### mean hue of detected putative seeds
    # max_seed_hue = 75/360
    # seeds_HSV = skimage.color.rgb2hsv(seeds)
    # idx = np.sum(mask_seeds > 0)
    # if idx > 0:
    #     seeds_mean_hue = np.mean(seeds_HSV[mask_seeds>0,0])
    # else:
    #     seeds_mean_hue = 0.0
    # ### if the identified sheets ware actually misidentified shoots
    # ### then adjust the seed hue limit into the narrow yellow band: 50 to 60 degrees
    # # print("Mean seed hue: "+ str(seeds_mean_hue))
    # if seeds_mean_hue > max_seed_hue:
    #     print("Putative seeds are above max hue: " + str(round(seeds_mean_hue,4)) + " > " + str(round(max_seed_hue,4)))
    #     vec_seed_hue_limit = [50/360, 60/360]
    #     seeds, mask_seeds, seeds_labeled, n_seeds = fun_detect_seeds_seedlings(seedlings=seedlings, mask_seedlings=mask_seedlings, mask_shoots=mask_shoots, seed_area_limit=seed_area_limit, vec_seed_hue_limit=vec_seed_hue_limit, vec_seed_sat_limit=vec_seed_sat_limit, vec_seed_val_limit=vec_seed_val_limit, debug=debug)
    ### merge seeds and shoots mask
    mask_seeds_shoots = mask_seeds | mask_shoots
    ### detect the contours
    contours, heirarchy = cv2.findContours(mask_seeds_shoots, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ### count
    n_seeds_seedlings = len(contours) ### potential to lose shoots and seeds contours here resulting to n_seeds_seedlings < n_shoots + n_seeds
    ### save shoot and root areas into a csv file
    r_germ = n_shoots/(n_seeds_seedlings+1e-10)
    col_ID = fname_base.replace("."+extension_name, "")
    if not(compute_areas):
        arr_germination = np.array([(col_ID, n_shoots, n_seeds_seedlings, round(r_germ,5))], dtype=[('ID', 'U256'), ('Number of shoots', 'f4'), ('Number of seeds and seedlings', 'f4'), ('Germination rate', 'f4')])
    else:
        seeds_roots_area = sum(np.ravel(mask_seedlings)>0) - shoot_area
        arr_germination = np.array([(col_ID, n_shoots, n_seeds_seedlings, round(r_germ,5), shoot_area, seeds_roots_area)], dtype=[('ID', 'U256'), ('Number of shoots', 'f4'), ('Number of seeds and seedlings', 'f4'), ('Germination rate', 'f4'), ('Shoots area', 'f4'), ('Seeds and roots area', 'f4')])
    if write_out:
        try:
            np.savetxt(fname=fname_out_csv, X=arr_germination, delimiter=',', fmt=['%s', '%f', '%f', '%f'], comments='')
        except:
            np.savetxt(fname=fname_out_csv, X=arr_germination, delimiter=',', fmt=['%s', '%f', '%f', '%f', '%f', '%f'], comments='')
    ### save image segmentation output
    arr_images = [rawImage, image, shoots_labeled, shoots, seeds, mask_seeds_shoots]
    vec_image_labels = ["Raw image", "Plate\n(V(Laplace):"+str(round(var_laplace))+")", "Shoots Labeled", "Shoots ("+str(n_shoots)+")", "Seeds ("+str(n_seeds)+")", "Mask Seeds and Shoots\n(Approx. Germ. Rate: "+str(n_shoots)+"/"+str(n_seeds_seedlings)+"="+str(round(r_germ,5))+")"]
    if plot_out:
        fun_image_show(arr_images,
                        title=vec_image_labels,
                        width=5, height=5, save=plot_out)
        plt.savefig(fname_out_jpg)
        plt.close()
    ### output
    return(arr_germination)

### seed dimensions
def fun_seed_dimensions(fname, seed_area_limit=[5000, np.Infinity], max_convex_hull_deviation=500, plot=False, dir_output=".", suffix_out="", write_out=False, plot_out=False, plot_width=5, plot_height=5):
    #############################################
    ### TEST
    # fname = 'res/At-seeds-Col_1-03.JPG'
    # # fname = 'res/At-seeds-Oy_0-04.JPG'
    # seed_area_limit = [5000, np.Infinity]
    # max_convex_hull_deviation = 500
    # plot=True; dir_output="."; suffix_out=""; write_out=False; plot_out=False; plot_width=5; plot_height=5
    #############################################
    ### image extension name or filename suffix
    extension_name = fname.split(".")[-1]
    ### output filenames
    fname_base = os.path.basename(fname)
    if dir_output==".":
        dir_output = os.path.dirname(fname)
    if suffix_out=="":
        suffix_out=""
    else:
        suffix_out = "-" + suffix_out
    fname_out_csv = os.path.join(dir_output, fname_base.replace("." + extension_name, "-germination_data" + suffix_out + ".csv"))
    fname_out_jpg = os.path.join(dir_output, fname_base.replace("." + extension_name, "-segmentation" + suffix_out + ".jpg"))
    ### load image
    image = io.imread(fname)
    ### flatten the RGB image into grayscale
    flatImage = fun_image_flatten(image, vec_coef=[0.2125, 0.7154, 0.0721])
    ### detect edges of the seeds
    edges = cv2.adaptiveThreshold(skimage.util.img_as_ubyte(np.uint8(flatImage)),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
    ### revert
    edges = abs(edges - 255)
    ### trying to close holes along the edge of the seeds
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    ### erode and dilate to remove artefacts and restore eroded edges
    edges = fun_multi_erosion_dilation(
                fun_multi_erosion_dilation(edges, type="erosion"),
            type="dilation")
    ### find initial countours prioir to coutour filling
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ### fill contours while removing countours that are too small
    for cnt in contours:
        cv2.fillPoly(edges, pts =[cnt], color=(255,255,255));
    ### find the contours of the flattened image with the filled seed contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ### iterate across contours and find the seed areas and dimensions
    vec_area_length_width = []
    for i in range(len(contours)):
        # i = 8127
        # print(i)
        cnt = contours[i]
        ### moments
        moments = cv2.moments(cnt)
        area = moments["m00"]
        ### filter by area
        if (area < seed_area_limit[0]) or (area > seed_area_limit[1]):
            continue
        ### filter by coinvexity, i.e. to remove overlapping seeds
        hull = cv2.convexHull(cnt, returnPoints=False)
        try:
            defects = cv2.convexityDefects(cnt,hull)
        except:
            continue
        X = np.reshape(defects, (len(defects), 4))
        mean_deviation = round(np.mean(X[:,3]))
        if mean_deviation > max_convex_hull_deviation:
            continue
        ### fit an ellipse to the contour and calculate the ratio between the major and minor axes
        (x,y), (minorAxisLength, majorAxisLength), angle = cv2.fitEllipse(cnt)
        axis_ratio = minorAxisLength / (minorAxisLength + majorAxisLength)
        ### annotate image
        ### append bounding rectangle
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        ### append text
        label = str(i) + ": " + str(round(area)) + "; " + str(round(majorAxisLength)) + "; " + str(round(minorAxisLength))
        cv2.putText(image, label, (x+int(w/2),y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        ### append seed area and dimensions
        vec_area_length_width.append([fname, area, majorAxisLength, minorAxisLength])
    ### plot
    if plot:
        fun_image_show([edges, image], title=["Edges", "Annotated"])
    ### output
    OUT = pd.DataFrame(vec_area_length_width, columns=["ID", "area", "length", "width"])
    if write_out:
        OUT.to_csv(fname_out_csv, index=False)
    ### save image segmentation output
    if plot_out:
        fun_image_show([edges, image],
                        title=["Edges", "Annotated"],
                        width=plot_width, height=plot_height, save=plot_out)
        plt.savefig(fname_out_jpg)
        plt.close()
    ### return
    return(OUT)
