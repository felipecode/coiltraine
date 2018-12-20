COiLTRAiNE System Overview
==========================


![COIL Diagram](img/CoIL.png?raw=true )

The idea of the system is to, by using a single command, train several
experiments with different [model conditions](docs/network.md) and
[input data configurations](docs/input.md).

For a given [experiment configuration file](docs/configuration.md), the main executer function
can perform:  [training](docs/main_modules.md/#train), measurement of the [model prediction error](docs/main_modules.md/#validation) on some
dataset and evaluation of the model performance on a [driving benchmark](docs/main_modules.md/#drive).


The training, prediction error, and driving benchmarks are performed
simultaneusly on different process. To perform evaluation
the validation and driving modules wait for the
training part to produce checkpoints. All of this is controlled
by the [executer module](docs/executer.md).


During the execution, all the information is [logged and
printed](docs/logger.md) on the screen to summarize the experiment
 training or evaluation status to be checked by a single glance
 of a user.
