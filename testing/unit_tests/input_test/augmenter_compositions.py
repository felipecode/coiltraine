
from torchvision import transforms
from input.augmenter import Augmenter, ToGPU
from input.augmenter import Add, Multiply, AddCPU, MultiplyCPU


add_test_composition = [ToGPU(), Add(10), Add(-10)]
add_random_composition = [ToGPU(), Add((-20, 20))]

mult_test_composition = [ToGPU(), Multiply(1.3), Multiply(1/1.3)]
mult_random_composition = [ToGPU(), Multiply((0.7, 1.3))]
mult_random_composition_color = [ToGPU(), Multiply((0.7, 1.3), per_channel=0.5)]

# The adding callable that is used to test augmentation.
add_test_augmenter = Augmenter(add_test_composition)
add_random_augmenter = Augmenter(add_random_composition)

# The test callable for testing multiplication
mul_test_augmenter = Augmenter(add_test_composition)
mul_random_augmenter = Augmenter(mult_random_composition)
mul_random_augmenter_color = Augmenter(mult_random_composition_color)

# CPU Big mul/add chain
muladd_cpu = Augmenter([transforms.ToTensor()] +
    [MultiplyCPU((0.95, 1.05)) for _ in range(100)])# + [AddCPU((-5, 5)) for _ in range(50)] )

# GPU Big mul/add chain
muladd_gpu = Augmenter( [ToGPU()] +
    [Multiply((0.95, 1.05)) for _ in range(100)]) #+ [Add((-5, 5)) for _ in range(50)] )

