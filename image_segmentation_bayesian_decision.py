import numpy as np
import scipy.io
import matplotlib.pyplot as plt


def create_feature_array_from_data(data):
    """
    Generate feature data (2nd largest value (in magnitude) of the DCT of data) Discrete Cosine Transform
    If data is (n,64), output will be (n,)
    :param data: numpy array (n,64)
    :return: features: numpy array (n,)
    """

    assert isinstance(data, np.ndarray)
    assert data.shape[1] == 64, "DCT has a feature length of 64"

    features = np.argmax(np.abs(data[:, 1:]), axis=1)
    features +=1  # argmax returns 0 indexed things, so 1 of features get maxed to 0
    # Feature is the index of the second largest absolute value of the vector

    assert features.shape == (data.shape[0],)

    return features


def plot_histogram(feature_0, feature_1, title="P(X|Y) Frequency of Occurrence histogram", normalize=False):
    """
    Plot histograms in a single plot  for two histogram data
    Format: feature_0  = (x, bins)
          : feature_1  = (y, bins)
    normalize : If set to True, will normalize histogram values
    :return:
    """
    assert isinstance(feature_0, np.ndarray)
    assert isinstance(feature_1, np.ndarray)
    assert isinstance(title, str)

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    _ = ax[0].hist(feature_0, bins=63, range=(1, 64), density=normalize)
    _ = ax[1].hist(feature_1, bins=63, range=(1, 64), density=normalize)

    # Note for histogram
    # For bins [1, 2, 3, 4],
    #   the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3).
    #       The last bin, however, is [3, 4], which includes 4.

    _ = ax[0].title.set_text("Y=0 Grass")
    _ = ax[1].title.set_text("Y=1 Cheetah")

    fig.suptitle(title)
    plt.show()


def create_likelihood_estimation(data_filename):
    """
    Given filename for .mat file containing Training and Test data, create likelihood estimation functions for both classes
    Estimate P(X|Y) from Training data for both Y=1 and Y=0
    :param data_filename:
    :return:
    """
    assert isinstance(data_filename, str)

    data_dict = scipy.io.loadmat(data_filename)

    data_array_class_0 = data_dict["TrainsampleDCT_BG"]   # Grass, class 0, background
    data_array_class_1 = data_dict["TrainsampleDCT_FG"]   # Cheetah, class 1, foreground

    feature_class_0 = create_feature_array_from_data(data_array_class_0)
    feature_class_1 = create_feature_array_from_data(data_array_class_1)

    # Plot histogram to check occurrence
    plot_histogram(feature_class_0, feature_class_1, normalize=False)

    # Plot normalized histogram
    plot_histogram(feature_class_0, feature_class_1, title="P(X|Y) Normalized histogram", normalize=True)


if __name__ == "__main__":
    create_likelihood_estimation("TrainingSamplesDCT_8")
