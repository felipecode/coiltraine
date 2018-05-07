from logger import coil_logger

class Loss(object):


    def __init__(self):
        pass

    # TODO: iteration control should go inside the logger, somehow
    def __call__(self, tensor, output_tensor):
        coil_logger.add_message('Loss', {
            "Iteration": 765,
            "LossValue": [0.232,  0.232,  0.332,  0.2322,  0.232,  0.232, 0.232]}
        )
        return tensor