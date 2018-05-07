
import multiprocessing
# Import all the test libraries.
import sys
import os

from coil_core import train, validate, run_drive



# You could send the module to be executed and they could have the same interface.



def execute_train(gpu, exp_batch, exp_alias):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """

    p = multiprocessing.Process(target=train.execute, args=(gpu, exp_batch, exp_alias,))
    p.start()

def execute_validation(gpu, exp_batch, exp_alias):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """
    #if module_name not in set(["train","drive","evaluate"]):
    #    raise ValueError("Invalid module to execute")



    # The difference between train and validation is the
    p = multiprocessing.Process(target=validate.execute, args=(gpu, exp_batch, exp_alias, ))
    p.start()


#TODO: set before the dataset path as environment variables

def execute_drive(gpu, exp_batch, exp_alias, city_name):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.
        path: The path were the datasets are

    Returns:

    """

    p = multiprocessing.Process(target=run_drive.execute, args=(gpu, exp_batch, exp_alias, city_name,))
    p.start()




def folder_execute(folder,gpus,param):
    """
    On this mode the training software keeps all
    It forks a process to run the monitor over the training logs.
    Arguments
        param, prioritize training, prioritize test, prioritize
    """


    #TODO: it is likely that the monitorer classes is not actually necessary.

    #for all methods in the folder
    #    if monitorer.get_status(folder, methods) == "Finished" # TODO: should we call this logger or monitorer ??
    #        if not done or executing  get to the list


    #Allocate all the gpus
    #for i in gpu:
    #    for a process and
    #    execute()
    #Check
    pass



if __name__ == '__main__':

    execute_train("0", "eccv", "experiment_1")
    #execute_validation("1", "experiment_2", 'Town01')
    execute_drive("2", "eccv", "experiment_3", 'Town02')
