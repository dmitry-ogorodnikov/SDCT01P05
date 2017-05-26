import cv2
import numpy as np
import glob2
from extract_features import extract_features, create_hog_desc
from training import get_model
from scipy.ndimage.measurements import label
from parameters import load_param
from time import process_time


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels, scale, thick=6):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, (int(bbox[0][0] * scale), int(bbox[0][1] * scale)),
                      (int(bbox[1][0] * scale), int(bbox[1][1] * scale)), (0, 0, 255), thick)
    # Return the image
    return img


def get_candidates(labels):
    # Iterate through all detected cars
    candidates = []
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        if (bbox[1][0] - bbox[0][0]) * 3 >= (bbox[1][1] - bbox[0][1]):
            candidates.append(bbox)
    # Return the candidates
    return candidates


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, param, hog_desc=None):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], param['img_size'])
        # 4) Extract features for that window using single_img_features()
        features = extract_features(test_img, param, hog_desc)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # prediction = int(clf.decision_function(test_features) > 0.7)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start_stop[0] = 0
    if x_start_stop[1] is None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] is None:
        y_start_stop[0] = 0
    if y_start_stop[1] is None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def multiscale_search(img, svc, scaler, param, scale, hog_desc=None):
    hot_windows = []
    all_windows = []

    windows_size = [(int(64 / scale), int(64 / scale)), (int(128 / scale), int(128 / scale))]
    y_start_stop = [[int(400 / scale), int(600 / scale)], [int(400 / scale), int(600 / scale)]]
    xy_overlap = [(0.75, 0.75), (0.75, 0.75)]

    for i in range(0, 2):
        windows = slide_window(img, xy_overlap=xy_overlap[i], xy_window=windows_size[i],
                               y_start_stop=y_start_stop[i])
        all_windows += [windows]
        hot_windows += search_windows(img, windows, svc, scaler, param, hog_desc)

    return hot_windows, all_windows


def additional_search(img, candidates, svc, scaler, param, scale, hog_desc=None):
    hot_windows = []
    windows_size = (32, 32)
    xy_overlap = (0.85, 0.85)
    offset = int(24 / scale)
    for candidate in candidates:
        y_start_stop = [max(candidate[0][1] - offset, int(400 / scale)),
                        min(candidate[1][1] + offset, int(600 / scale))]
        x_start_stop = [max(candidate[0][0] - offset, 0), min(candidate[1][0] + offset, img.shape[1])]
        windows = slide_window(img, xy_overlap=xy_overlap, xy_window=windows_size,
                               y_start_stop=y_start_stop, x_start_stop=x_start_stop)
        hot_windows += search_windows(img, windows, svc, scaler, param, hog_desc)
    return hot_windows


def main():
    svc, scaler = get_model()
    img_files = glob2.glob('./test_images/*.jpg')
    param = load_param()
    hog_desc = create_hog_desc(param['img_size'], param['orient'], param['pix_per_cell'], param['cell_per_block'])
    scale = 2
    counter = 0
    for img_file in img_files:
        temp = cv2.imread(img_file)
        img = cv2.resize(temp, (int(temp.shape[1] / scale), int(temp.shape[0] / scale)))

        hot_windows, all_windows = multiscale_search(img, svc, scaler, param, scale, hog_desc)

        heat = np.zeros_like(img[:, :, 0]).astype(np.float)
        heat = add_heat(heat, hot_windows)
        heat = cv2.GaussianBlur(heat, (11, 11), 0)

        labels = label(heat)
        candidates = get_candidates(labels)
        hot_windows += additional_search(img, candidates, svc, scaler, param, scale, hog_desc)

        hot_img = draw_boxes(img, hot_windows, thick=2)
        filename = 'output_images/detection_example' + str(counter) + '.png'
        counter += 1
        cv2.imwrite(filename, hot_img)
        cv2.imshow('Hot windows', hot_img)
        cv2.waitKey()
        #heat = add_heat(heat, hot_windows)

        #heatmap = np.clip(heat, 0, 255)
        #heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
        # plt.imshow(heatmap)
        # plt.show()
        # print(heatmap.shape)
        #labels = label(heatmap)
        #draw_img = draw_labeled_bboxes(np.copy(temp), labels, scale)

        '''draw_img = np.copy(img)
        print(all_windows)
        for i in range(len(all_windows)):
            draw_img = draw_boxes(img, all_windows[i])
            cv2.imshow('Result', draw_img)
            cv2.waitKey()'''
        # hot_img = draw_boxes(img, hot_windows, thick=2)

        # cv2.imshow('Heatmap', heatmap)
        # cv2.imshow('Result', draw_img)
        # cv2.waitKey()
    return


if __name__ == "__main__":
    main()
