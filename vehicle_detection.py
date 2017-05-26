import cv2
import numpy as np
from extract_features import create_hog_desc
from training import get_model
from scipy.ndimage.measurements import label
from parameters import load_param
from searching import multiscale_search, add_heat, apply_threshold, \
    draw_labeled_bboxes, additional_search, get_candidates
from moviepy.editor import VideoFileClip
import collections

svc, scaler = get_model()
param = load_param()
hog_desc = create_hog_desc(param['img_size'], param['orient'], param['pix_per_cell'], param['cell_per_block'])
heatmaps = collections.deque(maxlen=10)
scale = 2


def process_image(image):
    # global tracking_car
    img = cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hot_windows, _ = multiscale_search(img, svc, scaler, param, scale, hog_desc)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = cv2.GaussianBlur(heat, (11, 11), 0)

    labels = label(heat)
    candidates = get_candidates(labels)
    hot_windows = additional_search(img, candidates, svc, scaler, param, scale, hog_desc)

    heat = add_heat(heat, hot_windows)
    heatmap = apply_threshold(heat, 5)

    heatmaps.append(heatmap)
    heatmap = sum(heatmaps)
    heatmap = apply_threshold(heatmap, 50)
    heatmap = np.clip(heatmap, 0, 255)
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)
    labels = label(heatmap)
    return draw_labeled_bboxes(image, labels, scale, 2)


def main():
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    white_clip.write_videofile("output_project_video.mp4", audio=False)

    '''video_file = 'project_video.mp4'
    video_capture = cv2.VideoCapture(video_file)

    while video_capture.isOpened():
        ret_val, img = video_capture.read()
        if ret_val:
            draw_img = process_image(img)
            cv2.imshow('Result', draw_img)
            cv2.waitKey(1)
        else:
            break'''
    return


if __name__ == "__main__":
    main()
