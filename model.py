import time
import datetime
import glob
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from feature import extract_features


def pipeline_v1(vehicles, non_vehicles, params=None, save=False):
    """
    This function performs feature engineering, trains a Linear SVC, and optionally saves the fitted model.
    """
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

# Load the data
vehicle_image_fnames = glob.glob('./dataset/vehicles/**/*.png')
non_vehicle_image_fnames = glob.glob('./dataset/non-vehicles/**/*.png')
print('Number of vehicle images: {}'.format(len(vehicle_image_fnames)))
print('Number of non-vehicle images: {}'.format(len(non_vehicle_image_fnames)))            

# Train, test, and save model
pipeline_v1(vehicle_image_fnames, non_vehicle_image_fnames, save=True)


