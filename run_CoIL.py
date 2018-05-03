
import multiprocessing
from importlib.machinery import SourceFileLoader


# You could send the module to be executed and they could have the same interface.
def execute(gpu, module_name, exp_alias):
    """

    Args:
        gpu: The gpu being used for this execution.
        module_name: The module name, if it is train, drive or evaluate
        exp_alias: The experiment alias, file name, to be executed.

    Returns:

    """
    #if module_name not in set(["train","drive","evaluate"]):
    #    raise ValueError("Invalid module to execute")


    module = SourceFileLoader(module_name,'testing/unit_tests/structural_test/multiprocessing_test/'+module_name +'.py')
    module = module.load_module()
    p = multiprocessing.Process(target=module.execute, args=(gpu, exp_alias,))
    p.start()



    # Execute:
    # python3 main/<task> gpu path

    # The dataset is set inside the configuration file, however the path is manually set.




def folder_execute(folder,gpus,param):
    """
    On this mode the training software keeps all
    It forks a process to run the monitor over the training logs.
    Arguments
        param, prioritize training, prioritize test, prioritize
    """


    #TODO: it is likely that the monitorer classes is not actually necessary.

    #for all methods in the folder
    #    logger.check_if_done() # TODO: should we call this logger or monitorer ??
    #        if not done or executing  get to the list


    #Allocate all the gpus
    #for i in gpu:
    #    for a process and
    #    execute()
    #Check
    pass



if __name__ == '__main__':

    execute("0", "test_train", "experiment_1")
    execute("1", "test_train", "experiment_2")
    execute("2", "test_train", "experiment_3")