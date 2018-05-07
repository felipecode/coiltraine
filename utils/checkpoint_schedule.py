import os


def maximun_checkpoint_reach():



    if self._current_checkpoint_number >= len(self._checkpoint_schedule):
        return True
    else:
        return False



def next_check_point_ready():
    """
    Looks at every checkpoint file in the folder. And for each of
    then tries to find the one that matches EXACTLY with the one in the schedule

    :return:
    """

    """
    checkpoint_files = sorted(os.listdir(self._config_input.models_path))
    for f in checkpoint_files:

        match = re.search('model.ckpt-(\d+)', f)
        if match:
            checkpoint_number = match.group(1)

            if int(checkpoint_number) == (self._checkpoint_schedule[self._current_checkpoint_number]):
                self._checkpoint_number_to_test = str(self._checkpoint_schedule[self._current_checkpoint_number])

                return True
    logging.info('Checkpoint Not Found, Will wait for %d' % self._checkpoint_schedule[self._current_checkpoint_number] )
    """

    return False


def get_current_checkpoint():

    return 'first'

def get_next_checkpoint():
    """
    Look at the last checkpoint saved, and get the next on by looking at the scheduler
    Returns:

    """

    return self._current_checkpoint_number



def get_test_name():

    return str(self._checkpoint_number_to_test)

def finish_model():
    """
    Increment and go to the next model

    :return None:

    """
    self._current_checkpoint_number += 1


def is_iteration_for_saving():


    return True