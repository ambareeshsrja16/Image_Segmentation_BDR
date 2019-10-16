import math
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import cv2


class ImageSegmentation:

    def __init__(self, training_data="TrainingSamplesDCT_8"):
        assert isinstance(training_data, str)

        self.posterior_probability = None  # To be evaluated in create_posterior_probability method
        self.data_filename = training_data

    @staticmethod
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


    @staticmethod
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


    def create_posterior_probability(self):
        """
        Given filename for .mat file containing Training and Test data, create likelihood estimation functions for both classes
        Estimate P(X|Y) from Training data for both Y=1 and Y=0
        :return:
        """

        data_dict = scipy.io.loadmat(self.data_filename)

        data_array_class_0 = data_dict["TrainsampleDCT_BG"]  # Grass, class 0, background
        data_array_class_1 = data_dict["TrainsampleDCT_FG"]  # Cheetah, class 1, foreground

        feature_class_0 = self.create_feature_array_from_data(data_array_class_0)
        feature_class_1 = self.create_feature_array_from_data(data_array_class_1)

        # Plot histogram to check occurrence
        self.plot_histogram(feature_class_0, feature_class_1, normalize=False)

        # Plot normalized histogram
        probs, bins = self.plot_histogram(feature_class_0, feature_class_1, title="P(X|Y) Normalized histogram", normalize=True)

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

        self.posterior_probability = {key: val for (key, val) in zip(bins[:-1], posterior_predictions)}  # Lookup table


    def get_prediction_on_block(self, block, f=8):
        """
        Gets a numpy array (a block of pixels from the image of size f*f and predicts if that block belongs to 0 or 1
        :param f: filter size
        :param block:
        :return: y (0 or 1)
        """
        assert isinstance(block, np.ndarray)
        assert block.size == f**2
        assert self.posterior_probability, "Call method create_posterior_probability() to fill up posterior_probability dictionary"

        # OpenCV equivalent of MATLAB function dct2
        dct_block = cv2.dct(block)

        ## START FROM HERE,
        # perform dct on elements,
        # get zigzagged elmeents from static method
        # Generate feature (write function for that

        return y

    def do_segmentation(self, test_image_name = "cheetah.bmp"):
        """

        :param test_image_name:
        :return:
        """

        assert isinstance(test_image_name, str)

        test_image = cv2.imread(test_image_name)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)  # convert to GScale, cv2 reads images by default as BGR

        assert len(test_image.shape) == 2

        # Stride of 1, Therefore for SAME padding, p1+p2  = f - 1, where p1=top, p2=bottom, f=filter
        # cv2.BORDER_DEFAULT pads the border with reflection padding(but doesn't repeat the border pixels themselves)
        # https://answers.opencv.org/question/50706/border_reflect-vs-border_reflect_101/
        f = 8  # filter
        p1, p2 = 3, 4  # padding on left/right or top/bottom
        padded_test_image = cv2.copyMakeBorder(test_image, p1, p2, p1, p2, borderType=cv2.BORDER_DEFAULT)
        segmentation_mask = np.zeros_like(test_image)

        R, C = segmentation_mask.shape

        for i in range(R):
            for j in range(C):
                segmentation_mask[i,j] = self.get_prediction_on_block(padded_test_image[i:i+f, j:j+f].copy(), f)
                # pass a copy of the numpy array view, so that original is unaffected

    @staticmethod
    def generate_zig_zig_pattern(input_array):
        """
        Generate zig zag pattern in 2-d array

        Example, for 8*8

        0   1   5   6  14  15  27  28

        2   4   7  13  16  26  29  42

        3   8  12  17  25  30  41  43

        9  11  18  24  31  40  44  53

        10  19  23  32  39  45  52  54

        20  22  33  38  46  51  55  60

        21  34  37  47  50  56  59  61

        35  36  48  49  57  58  62  63

        :return:  new_arr (R*C,) with zig_zag elements from the input_array
        """
        assert isinstance(input_array, np.ndarray)
        assert len(input_array) > 0

        R, C = input_array.shape

        new_arr = [0] * R * C
        count = 0

        direction = 1  # Initially 1 => 1 for upward diagonal (up, right), 0 for downward diagonal (down, left)
        move = [(1, -1), (-1, 1)]
        r, c = 0, 0  # pointers in current array, initialized to (0,0)

        check_bound_of_arr = lambda x, y: 0 <= x < R and 0 <= y < C  # check if pointers fall outside array

        while count < R * C:
            new_arr[count] = input_array[r, c]
            count += 1

            c_r, c_c = r + move[direction][0], c + move[direction][1]

            if check_bound_of_arr(c_r, c_c):
                r, c = c_r, c_c
            else:
                if direction:
                    c_r, c_c = r, c + 1
                    if not check_bound_of_arr(c_r, c_c):
                        c_r, c_c = r + 1, c
                else:
                    c_r, c_c = r + 1, c
                    if not check_bound_of_arr(c_r, c_c):
                        c_r, c_c = r, c + 1

                direction = (direction + 1) % 2  # change direction
                r, c = c_r, c_c

        return new_arr


if __name__ == "__main__":
    data_filename = "TrainingSamplesDCT_8"

    img_segment = ImageSegmentation(data_filename)
    img_segment.create_posterior_probability()





