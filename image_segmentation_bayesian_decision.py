import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2


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
    features += 1  # argmax returns 0 indexed things, so 1 of features get maxed to 0
    # Feature is the index of the second largest absolute value of the vector

    assert features.shape == (data.shape[0],)

    return features


def plot_histogram(feature_0, feature_1, title="P(X|Y) Frequency of Occurrence histogram", normalize=False, show=False):
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
    prob_class_0, bins_0, _ = ax[0].hist(feature_0, bins=63, range=(1, 64), density=normalize)
    prob_class_1, _, _ = ax[1].hist(feature_1, bins=63, range=(1, 64), density=normalize)

    # Note for histogram
    # For bins [1, 2, 3, 4],
    #   the first bin is [1, 2) (including 1, but excluding 2) and the second [2, 3).
    #       The last bin, however, is [3, 4], which includes 4.

    _ = ax[0].title.set_text("Y=0 Grass")
    _ = ax[1].title.set_text("Y=1 Cheetah")

    fig.suptitle(title)
    if show:
        plt.show()

    if normalize:
        return (prob_class_0, prob_class_1), bins_0


def create_posterior_probability(data_filename):
    """
    Given filename for .mat file containing Training and Test data, create likelihood estimation functions for both classes
    Estimate P(X|Y) from Training data for both Y=1 and Y=0
    :param data_filename:
    :return:
    """
    assert isinstance(data_filename, str)

    data_dict = scipy.io.loadmat(data_filename)

    data_array_class_0 = data_dict["TrainsampleDCT_BG"]  # Grass, class 0, background
    data_array_class_1 = data_dict["TrainsampleDCT_FG"]  # Cheetah, class 1, foreground

    feature_class_0 = create_feature_array_from_data(data_array_class_0)
    feature_class_1 = create_feature_array_from_data(data_array_class_1)

    # Plot histogram to check occurrence
    plot_histogram(feature_class_0, feature_class_1, normalize=False)

    # Plot normalized histogram
    probs, bins = plot_histogram(feature_class_0, feature_class_1, title="P(X|Y) Normalized histogram", normalize=True)

    likelihood_class_0, likelihood_class_1 = probs

    assert math.isclose(sum(likelihood_class_0), 1) \
           and math.isclose(sum(likelihood_class_1), 1), "Normalized probabilities should sum to 1"

    assert len(bins) == 64  # All bin left edges and the last bin right edge

    prior_class_0 = len(feature_class_0) / (len(feature_class_0) + len(feature_class_1))
    prior_class_1 = 1 - prior_class_0

    posterior_class_0, posterior_class_1 = likelihood_class_0 * prior_class_0, likelihood_class_1 * prior_class_1
    # P(Y|X) = (P(X|Y) * P(Y))/ P(X) , since P(X) is common for both, ignore that for comparisons

    posterior_predictions = np.greater(posterior_class_1, posterior_class_0)  # if class_1 has higher probability,
    # evaluates to 1 if not 0
    posterior_probability = {key: val for (key, val) in zip(bins[:-1], posterior_predictions)}  # Lookup table

    return posterior_probability


if __name__ == "__main__":
    posterior_probability = create_posterior_probability("TrainingSamplesDCT_8")

    test_image_name = "cheetah.bmp"
    test_image = cv2.imread(test_image_name)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)  # convert to GScale, cv2 reads images by default as BGR

    assert len(test_image.shape) == 2

