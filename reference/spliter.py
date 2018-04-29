def partition_keys_by_steering(steerings, keys):
    # print len(steerings)
    # print steerings
    max_steer = min(0.6, max(steerings))  # SUPER HACKY MEGA WARNING
    print
    'Max Steer'
    print
    max_steer
    min_steer = max(-0.5, min(steerings))
    print
    'Min Steer'
    print
    min_steer
    # print steerings

    steerinterval = (max_steer - min_steer) / len(self._steering_bins_perc)

    iter_value = min_steer + steerinterval
    iter_index = 0
    splited_keys = []
    # print 'len steerings'
    # print len(steerings)
    for i in range(0, len(steerings)):

        if steerings[i] >= iter_value:
            # We split

            splited_keys.append(keys[iter_index:i])
            iter_index = i
            iter_value = iter_value + steerinterval

            print
            'split on ', i
            print
            len(splited_keys)
            print
            len(splited_keys[-1])

    return splited_keys
