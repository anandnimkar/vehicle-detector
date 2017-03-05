import numpy as np
import cv2
import matplotlib.image as mpimg
from skimage.feature import hog

# Feature Extraction using Color Space Transforms, Spatial Binning, Color Histograms, and Histogram of Oriented Gradients (HOG)

def convert_cspace(img, cspace):
    # apply color conversion if other than 'RGB'
    if cspace != 'RGB' and cspace in ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']:
        return cv2.cvtColor(img, eval('cv2.COLOR_RGB2' + cspace))
    elif cspace == 'RGB':
        return np.copy(img)
    else:
        raise ValueError("cspace must be one of 'RGB', 'HSV', 'LUV', 'HLS', 'YUV', or 'YCrCb'")

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
    

def extract_features(imgs, cspace='RGB', spatial_size=(32,32), hist_bins=32, hist_range=(0, 256), 
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for f in imgs:
        
        # Read in each one by one and convert to the appropriate color space
        image = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) # read in using cv2 to keep the 0 to 255 scaling for png images
        image = convert_cspace(image, cspace)

        # Apply bin_spatial() to get spatial color features
        spatial_features = bin_spatial(image, size=spatial_size)
        
        # Apply color_hist() to get color histogram features
        hist_features = color_hist(image, nbins=hist_bins, bins_range=hist_range)
                                    
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(image.shape[2]):
                hog_features.append(get_hog_features(image[:,:,channel], orient, pix_per_cell, cell_per_block, 
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True)
        
        # Append the new feature vector to the features list
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    
    # Return list of feature vectors
    return features
