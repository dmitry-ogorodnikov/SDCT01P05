import glob2
from random import shuffle
import pickle


def split_data():
    img_car_files = glob2.glob('./vehicles/GTI_Far/*.png')
    img_car_files += glob2.glob('./vehicles/GTI_Left/*.png')
    img_car_files += glob2.glob('./vehicles/GTI_MiddleClose/*.png')
    img_car_files += glob2.glob('./vehicles/GTI_Right/*.png')
    img_car_files += glob2.glob('./vehicles/KITTI_extracted/*.png')
    print(len(img_car_files))

    img_noncar_files = glob2.glob('./non-vehicles/Extras/*.png')
    img_noncar_files += glob2.glob('./non-vehicles/GTI/*.png')
    print(len(img_noncar_files))

    shuffle(img_car_files)
    shuffle(img_noncar_files)

    print(img_car_files[0])
    print(img_noncar_files[0])

    # Save filenames to a pickle file
    dist_pickle = {'car': img_car_files, 'noncar': img_noncar_files}
    pickle.dump(dist_pickle, open('data.p', 'wb'))
    return


def get_data():
    param = pickle.load(open('data.p', 'rb'))
    return param['car'], param['noncar']


def main():
    split_data()
    return


if __name__ == "__main__":
    main()
