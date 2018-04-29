import torch

class ToGPU(object):


    def __call__(self, img):
        #img = img.transpose(0, 1)
        #img = img.transpose(1, 2)
        #img = img.transpose(2, 3)
        return torch.squeeze(img.cuda())



class Augmenter(object):
    # Here besides just applying the list, the class should also apply the scheduling
    # Based on the augmentation

    def __init__(self, composition):
        self.transforms = composition

    def __call__(self, iteration, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

