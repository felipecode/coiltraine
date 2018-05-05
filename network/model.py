from logger import coil_logger

class Model(object):


    def __init__(self, model_definition):

        # TODO: MAKE THE model
        pass

    # TODO: iteration control should go inside the logger, somehow
    def __call__(self, tensor):

        # TODO: TRACK NANS OUTPUTS
        coil_logger.add_message('Model', {
            "Iteration": 765,
            "Output": [ 1.0,12.3,124.29]
        }
                                )
        return tensor