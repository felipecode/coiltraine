from logger import coil_logger
import torch.nn as nn


class CoILModel(nn.Module):


    def __init__(self, model_definition):
        super(CoILModel, self).__init__()
        # TODO: MAKE THE model
        pass

    # TODO: iteration control should go inside the logger, somehow

    def forward(self, x):

        # TODO: TRACK NANS OUTPUTS
        # TODO: Maybe change the name
        coil_logger.add_message('Model', {
            "Iteration": 765,
            "Output": [1.0, 12.3, 124.29]
        }
                                )
        return x


    def load_network(self, checkpoint):
        """
        Load a network for a given model definition .

        Args:
            checkpoint: The checkpoint that the user wants to add .



        """
        coil_logger.add_message('Loading', {
                    "Model": {"Loaded checkpoint: " + str(checkpoint) }

                })



        # TODO: implement



