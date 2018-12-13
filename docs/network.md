Model Configurations
====================

#### Network

For the network we provide some abstraction to easily change
the architecture.
For the networks we have a concept of model type.
Inside a model type there is a certain training
strategy related to it. The one we use is called [model icra](https://arxiv.org/pdf/1710.02410.pdf) that has 4 subdivisions:
percetion, measurements, join and branching.

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
as it is defined on the example.




#### Loss



#### Optimizer

For the optimizer we have [adam](https://arxiv.org/abs/1412.6980) hardcoded.
You can configure the start learning rate and also a
interval where the base learning rate is reduced.


```
LEARNING_RATE: 0.0002  # First learning rate
LEARNING_RATE_DECAY_INTERVAL: 75000 # Number of iterations where the learning rate is reduced
LEARNING_RATE_DECAY_LEVEL: 0.5 # Th factor of reduction applied to the learning rate
```


