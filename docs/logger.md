Logger and Printer
==================


FIGURE SHOWING


The idea of the logger is that with a single
glance you could look to a set of experiments.

So the log is made during execution on ALL levels,
but also there is a interface for the user to
consult it at any moment.

The logger is global and accessible at any moment.


The experiments have a set of status for each part of
the logging. The general status is the last one that stopped.

* Not Started
* Loading
* Iterating
* Error
* Finished


# Training Messages

# Validation Messages

# Driving Messages


The folder organization for logging is as following:

* Root
    * Experiment Folder
        * Experiment Name
            * Logs



The messages are JSON

For our system the following message types exist.

Loading messages


When the agent is testing there are several mesages

1.Control Output
2.Image Input # TODO: a way to take images inputs very fast
3.Full measurements
4.Directions
5.Activation images.



