import cv2
import numpy as np
from skimage.feature import hog
from data_exploration import get_data
from parameters import load_param


def create_hog_desc(shape=(64, 64), orient=9, pix_per_cell=8, cell_per_block=2):
    cell_size = (pix_per_cell, pix_per_cell)  # h x w in pixels
    block_size = (cell_per_block, cell_per_block)  # h x w in cells
    nbins = orient  # number of orientation bins

    # winSize is the size of the image cropped to an multiple of the cell size
    return cv2.HOGDescriptor(_winSize=(shape[1] // cell_size[1] * cell_size[1],
                                       shape[0] // cell_size[0] * cell_size[0]),
                             _blockSize=(block_size[1] * cell_size[1],
                                         block_size[0] * cell_size[0]),
                             _blockStride=(cell_size[1], cell_size[0]),
                             _cellSize=(cell_size[1], cell_size[0]),
                             _nbins=nbins)


def get_hog_features_cv(img, hog_desc):
    return hog_desc.compute(img)[:, 0]


# Define a function to return HOG features and visualization
# Return 'features, hog_image' if vis == True
# Return 'features' if vis == False
def get_hog_features(img, orient=9, pix_per_cell=8, cell_per_block=2,
                     vis=False, feature_vec=True):
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               transform_sqrt=True,
               block_norm='L2-Hys',
               visualise=vis, feature_vector=feature_vec)


# Define a function to compute binned color features
def bin_spatial(img, size=(16, 16)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to extract features image
def extract_features(img, param, hog_desc=None):
    features = []
    # apply color conversion if other than 'BGR'
    image = cv2.resize(img, param['img_size'])
    color_space = param['color_space']
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else:
        feature_image = np.copy(image)

    if param['spatial_feat']:
        spatial_features = bin_spatial(feature_image, size=param['spatial_size'])
        features.append(spatial_features)
    if param['hist_feat']:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=param['hist_bins'])
        features.append(hist_features)
    if param['hog_feat']:
        # Call get_hog_features() with vis=False, feature_vec=True
        hog_channel = param['hog_channel']
        orient = param['orient']
        pix_per_cell = param['pix_per_cell']
        cell_per_block = param['cell_per_block']
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                if param['use_opencv_hog'] and hog_desc is not None:
                    hog_features.append(get_hog_features_cv(feature_image[:, :, channel], hog_desc))
                else:
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            if param['use_opencv_hog'] and hog_desc is not None:
                hog_features = get_hog_features_cv(feature_image[:, :, hog_channel], hog_desc)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)

    return np.concatenate(features)


# Define a function to extract features from a list of images
def extract_features_from_files(img_files, param):
    # Create a list to append feature vectors to
    features = []
    if param['use_opencv_hog']:
        hog_desc = create_hog_desc(param['img_size'], param['orient'], param['pix_per_cell'], param['cell_per_block'])
    else:
        hog_desc = None
    # Iterate through the list of images
    for file in img_files:
        image = cv2.imread(file)
        features.append(extract_features(image, param, hog_desc))
    # Return list of feature vectors
    return features


def main():
    car_files, noncar_files = get_data()
    param = load_param()
    print('car images = ', len(car_files))
    print('noncar images = ', len(noncar_files))
    img = cv2.imread(car_files[0])

    features = extract_features(img, param)
    print(len(features))
    return


if __name__ == "__main__":
    main()
