import pickle

color_space = 'YCrCb'
spatial_size = (16, 16)
hist_bins = 16
orient = 9
pix_per_cell = 4
cell_per_block = 2
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True
use_opencv_hog = True
img_size = (32, 32)


def save_param():
    # Save parameters to a pickle file
    dist_pickle = {'color_space': color_space, 'spatial_size': spatial_size,
                   'hist_bins': hist_bins, 'orient': orient,
                   'pix_per_cell': pix_per_cell, 'cell_per_block': cell_per_block,
                   'hog_channel': hog_channel, 'spatial_feat': spatial_feat,
                   'hist_feat': hist_feat, 'hog_feat': hog_feat,
                   'use_opencv_hog': use_opencv_hog, 'img_size': img_size}
    pickle.dump(dist_pickle, open('param.p', 'wb'))
    return


def load_param():
    param = pickle.load(open('param.p', 'rb'))
    return param


def main():
    save_param()
    return


if __name__ == "__main__":
    main()
