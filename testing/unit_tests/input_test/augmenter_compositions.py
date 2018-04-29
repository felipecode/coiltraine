
from torchvision import transforms


from imgauggpu import Augmenter, AugmenterCPU, ToGPU, Dropout, CoarseDropout, GaussianBlur, Grayscale,\
    Sometimes, AdditiveGaussianNoise, ContrastNormalization
from imgauggpu import Add, Multiply
from imgaug.augmenters.arithmetic import Add as AddCPU
from imgaug.augmenters.arithmetic import Multiply as MultiplyCPU
from imgaug.augmenters.arithmetic import Dropout as DropoutCPU
from imgaug.augmenters.arithmetic import CoarseDropout as CoarseDropoutCPU
from imgaug.augmenters.color import Grayscale as GrayscaleCPU
from imgaug.augmenters.blur import GaussianBlur as GaussianBlurCPU
from imgaug.augmenters.arithmetic import AdditiveGaussianNoise as AdditiveGaussianNoiseCPU
from imgaug.augmenters.arithmetic import ContrastNormalization as ContrastNormalizationCPU


add_test_composition = [ToGPU(), Add(10), Add(-10)]
add_random_composition = [ToGPU(), Add((-20, 20))]

mult_test_composition = [ToGPU(), Multiply(1.3), Multiply(1/1.3)]
mult_random_composition = [ToGPU(), Multiply((0.7, 1.3))]


# The adding callable that is used to test augmentation.
add_test_augmenter = Augmenter(add_test_composition)
add_random_augmenter = Augmenter(add_random_composition)
add_random_augmenter_color = Augmenter([ToGPU(), Add((-30, 30), per_channel=0.2)])
add_random_augmenter_color_cpu = AugmenterCPU([AddCPU((-30, 30), per_channel=0.2)])

# The test callable for testing multiplication
mul_test_augmenter = Augmenter(add_test_composition)
mul_random_augmenter = Augmenter(mult_random_composition)
mul_random_augmenter_color = Augmenter([ToGPU(), Multiply((0.7, 1.3), per_channel=0.5)])
mult_random_color_sometimes = Augmenter([ToGPU(), Sometimes(Multiply((0.7, 1.3), per_channel=0.5), 1.0)])




# CPU Big mul/add chain
muladd_cpu = AugmenterCPU(
    [MultiplyCPU((0.95, 1.05))for _ in range(10)] + [AddCPU((-5, 5)) for _ in range(10)] )

# GPU Big mul/add chain
muladd_gpu = Augmenter( [ToGPU()] +
    [Multiply((0.95, 1.05)) for _ in range(10)] + [Add((-5, 5)) for _ in range(10)] )

# dropout test
dropout_random = Augmenter([ToGPU(), Dropout(0.2)])

dropout_random_color = Augmenter([ToGPU(), Dropout(0.2, per_channel=0.5)])

dropout_random_cpu = AugmenterCPU([DropoutCPU(0.2)])

# Coarse dropout


coarse_dropout_random = Augmenter([ToGPU(), CoarseDropout((0.0, 0.20), size_px=(16, 16))])

coarse_dropout_random_color = Augmenter([ToGPU(), CoarseDropout((0.0, 0.20), size_px=(16, 16), per_channel=0.5)])

coarse_dropout_random_cpu = AugmenterCPU([CoarseDropoutCPU((0.0, 0.20), size_px=(16, 16))])

# Random gausian blur

gaussian_blur_test =Augmenter([ToGPU(), GaussianBlur(sigma=(0.0, 3.0))])


gaussian_blur_test_cpu =AugmenterCPU([GaussianBlurCPU(sigma=(0.0, 3.0))])


# Add the grayscale kernel to work

grayscale_test = Augmenter([ToGPU(), Grayscale(alpha=(0.0, 1.0))])

grayscale_test_cpu = Augmenter([ GrayscaleCPU(alpha=(0.0, 1.0))])

# Test Additive gaussian noise

additive_gaussian_test = Augmenter([ToGPU(), AdditiveGaussianNoise(loc=0, scale=(0.0, 0.10*255))])

additive_gaussian_test_color = Augmenter([ToGPU(), AdditiveGaussianNoise(loc=0, scale=(0.0, 0.10*255), per_channel=0.5)])

additive_gaussian_test_cpu = AugmenterCPU([ AdditiveGaussianNoiseCPU(loc=0, scale=(0.0, 0.10*255))])

# Test the contrast normalization


contrast_normalization_test = Augmenter([ToGPU(), ContrastNormalization((0.5, 1.5))])

contrast_normalization_test_color = Augmenter([ToGPU(), ContrastNormalization((0.5, 1.5), per_channel=0.5)])

contrast_normalization_test_cpu = AugmenterCPU([ContrastNormalizationCPU((0.5, 1.5))])

