The idea of the logger is that with a single
glance you could look to a set of experiments.

So the log is made during execution on ALL levels,
but also there is a interface for the user to
consult it at any moment.

The logger is global and accessible at any moment.

(CHECK FOR THREAD SAFETY ISSUES)

The experiments have a set of status for each part of
the logging. The general status is the last one that stopped.

* To Run
* Running
* Error
* Finished


The folder organization for logging is as following:

* Root
    * Experiment Folder
        * Experiment Name
            * Logs



The messages are JSON

For our system the following message types exist.

Loading messages

Running Messages
From the running messages it has three types

1. Reading
2. Network
3. Optimization





