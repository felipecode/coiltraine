from input.augmenter import Augmenter
from input.augmenter import Add, Multiply


add_test_composition = [Add(10), Add(-10)]
add_random_composition = [Add(-7, 7), Add(-15, 15), Add(-30, 30)]

mult_test_composition = [Multiply(1.3), Multiply(1/1.3)]


# The adding callable that is used to test augmentation.
add_test_augmenter = Augmenter(add_test_composition)
add_random_augmenter = Augmenter(add_random_composition)

# The test callable for testing multiplication
mul_test_augmenter = Augmenter(add_test_composition)
