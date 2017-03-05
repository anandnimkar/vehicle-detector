import time
import datetime
import glob
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler


# Load Data

vehicle_image_fnames = glob.glob('./dataset/vehicles/**/*.png')
non_vehicle_image_fnames = glob.glob('./dataset/non-vehicles/**/*.png')
print('Number of vehicle images: {}'.format(len(vehicle_image_fnames)))
print('Number of non-vehicle images: {}'.format(len(non_vehicle_image_fnames)))


# Feature Extraction using Color Space Transforms, Spatial Binning, Color Histograms, and Histogram of Oriented Gradients (HOG)

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

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
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

def extract_features(imgs, cspace='BGR', spatial_size=(32,32), hist_bins=32, hist_range=(0, 256), 
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL'):
    # Create a list to append feature vectors to
    features = []
    
    # Iterate through the list of images
    for file in imgs:
        
        # Read in each one by one
        image = cv2.imread(file)
        
        # apply color conversion if other than 'BGR'
        if cspace != 'BGR' and cspace in ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']:
            feature_image = cv2.cvtColor(image, eval('cv2.COLOR_BGR2' + cspace))
        elif cspace == 'BGR':
            feature_image = np.copy(image)
        else:
            raise ValueError("cspace must be one of 'BGR', 'HSV', 'LUV', 'HLS', 'YUV', or 'YCrCb'")
            
        # Apply bin_spatial() to get spatial color features
        bs = bin_spatial(image, size=spatial_size)
        
        # Apply color_hist() to get color histogram features
        cs = color_hist(image, nbins=hist_bins, bins_range=hist_range)
                                    
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], orient, pix_per_cell, cell_per_block, 
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, pix_per_cell, cell_per_block, 
                                            vis=False, feature_vec=True)
        
        # Append the new feature vector to the features list
        features.append(np.concatenate((bs, cs, hog_features)))
    
    # Return list of feature vectors
    return features


# Training a Support Vector Machine (SVM) Classifier

def pipeline_v1(vehicles, non_vehicles, params=None, save=False):

    params = params or {
        # color space
        'cspace': 'YCrCb', # Can be BGR, HSV, LUV, HLS, YUV, or YCrCb
        
        # spatial binning params
        'spatial_size': (24, 24),
    
        # color histogram params
        'hist_bins': 32,
        'hist_range': (0, 256),

        # HOG params
        'orient': 9,
        'pix_per_cell': 8,
        'cell_per_block': 2,
        'hog_channel':'ALL' # Can be 0, 1, 2, or "ALL"
    }
    
    t=time.time()
    vehicle_features = extract_features(vehicles, **params)
    non_vehicle_features = extract_features(non_vehicles, **params)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')
    
    # Create an array stack of feature vectors
    X = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)                        
    
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(vehicle_features)), np.zeros(len(non_vehicle_features))))

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

    print('Using:', params['orient'], 'orientations, ', params['pix_per_cell'],
          'pixels per cell, and', params['cell_per_block'] ,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    
    # Use a linear SVC 
    svc = LinearSVC()
    
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    
    # Check the score of the SVC
    acc = round(svc.score(X_test, y_test), 4)
    print('Test Accuracy of SVC = ', acc)
    
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
    
    if save:
        now = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        pickle_fname = ('./saved_models/{}|test_acc={}|train_samples={}|test_samples={}.pickle'
                        .format(now, acc, len(y_train), len(y_test)))
        with open(pickle_fname, 'wb') as f:
            pickle.dump({
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'X_scaler': X_scaler,
                'svc': svc,
                'params': params
            }, f, pickle.HIGHEST_PROTOCOL)
            print('Saved model and params to {}'.format(pickle_fname))

# Train, test, and save model
pipeline_v1(vehicle_image_fnames, non_vehicle_image_fnames, save=True)


