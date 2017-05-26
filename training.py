from extract_features import extract_features_from_files
from data_exploration import get_data
from parameters import load_param
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib


def training():
    param = load_param()
    car_files, noncar_files = get_data()
    car_features = extract_features_from_files(car_files, param)
    print('car_features size :', len(car_features))
    noncar_features = extract_features_from_files(noncar_files, param)
    print('noncar_features size :', len(noncar_features))

    # Create an array stack of feature vectors
    x = np.vstack((car_features, noncar_features)).astype(np.float64)
    # Fit a per-column scaler
    x_scaler = RobustScaler().fit(x)
    # Apply the scaler to X
    scaled_x = x_scaler.transform(x)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))
    x_train, x_test, y_train, y_test = train_test_split(
        scaled_x, y, test_size=0.2)
    print('x_train shape is :', x_train.shape)
    print('x_test shape is :', x_test.shape)
    # Use a linear SVC
    svc = LinearSVC()
    # svc = SVC()
    svc.fit(x_train, y_train)

    # Save svc and scaler to a pickle file
    dist_pickle = {'svc': svc, 'scaler': x_scaler}
    joblib.dump(dist_pickle, 'model.p')
    print('Accuracy of SVC = ', svc.score(x_test, y_test))


def get_model():
    model = joblib.load('model.p')
    return model['svc'], model['scaler']


def main():
    training()
    return


if __name__ == "__main__":
    main()
