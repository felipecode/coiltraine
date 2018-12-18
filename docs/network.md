Model Configurations
====================

#### Network

For the network we provide some abstraction to easily change
the architecture. A few key attributes can be changed

* MODEL TYPE: The type of model with some training strategy related to it,
 right now we just have  [coil-icra](https://arxiv.org/pdf/1710.02410.pdf)
  that has 4 subdivisions: percetion, measurements, join and branching.
* MODEL CONFIGURATION: Specify all the network

Example:

```
    MODEL_TYPE: 'coil-icra' # The type of model. Defines which modules the model has.
    MODEL_CONFIGURATION:  # Based on the MODEL_TYPE, we specify the structure
      perception:  # The module that process the image input, it ouput the number of classes
        res:
          name: 'resnet34'
          num_classes: 512

      measurements:  # The module the process the input float data, in this case speed_input
        fc:  # Easy to configure fully connected layer
          neurons: [128, 128] # Each position add a new layer with the specified number of neurons
          dropouts: [0.0, 0.0]
      join:  # The module that joins both the measurements and the perception
        fc:
          neurons: [512]
          dropouts: [0.0]
      speed_branch:  # The prediction branch speed branch
        fc:
          neurons: [256, 256]
          dropouts: [0.0, 0.5]
      branches:  # The output branches for the different possible directions ( Straight, Left, Right, None)
        number_of_branches: 4
        fc:
          neurons: [256, 256]
          dropouts: [0.0, 0.5]
    PRE_TRAINED: False  # If the weights are started with imagenet.
```

The perception can also be defined layer by layer
as it is defined on the [example](../configs/sample/coil_icra.yaml).


#### Loss

The loss can be selected between L2 or L1 losses. The loss
is created with a factory function based on the string passed
in the LOSS_FUNCTION configuration parameter.

For the loss, considering the conditional model, the weight
for each branch must be defined.
Further, the user must define the weights for each of the TARGET
variables produced by the network.

Sample:

```
    # Loss Parameters #
    BRANCH_LOSS_WEIGHT: [0.95, 0.95, 0.95, 0.95, 0.05] # how much each branch is weighted when computing loss
    LOSS_FUNCTION: 'L1' # The loss function used
    VARIABLE_WEIGHT: # how much each of the outputs specified on TARGETS are weighted for learning.
      Steer: 0.5
      Gas: 0.45
      Brake: 0.05
```

#### Optimizer

For the optimizer we have [adam](https://arxiv.org/abs/1412.6980) hardcoded.
You can configure the start learning rate and also a
interval where the base learning rate is reduced.


Sample:

```
    LEARNING_RATE: 0.0002  # First learning rate
    LEARNING_RATE_DECAY_INTERVAL: 75000 # Number of iterations where the learning rate is reduced
    LEARNING_RATE_DECAY_LEVEL: 0.5 # Th factor of reduction applied to the learning rate
```


