
#TODO: This file could be easily eliminated !
def coil_input_init():

    #Define the dataset. This structure is has the __get_item__ redefined in a way
    #that you can access the HDFILES positions from the root directory as a in a vector.

    #The transform is defined in order to already convert to tensor, not to GPU yet due
    # to pytorch limitations

    dataset = CoILDataset(global_conf.root_directory, transform=transform.ToTensor())


    # Creates the sampler, this part is responsible for managing the keys. It divides
    # all keys depending on the measurements and produces a set of keys for each bach.
    sampler = CoILSampler(dataset.measurements)

    # The data loader is the multi threaded module from pytorch that release a number of
    # workers to get all the data.
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=120,
                                              shuffle=False, num_workers=12, pin_memory=True)
    # By instanciating the augmenter we get a callable that augment images and transform them
    # into tensors.
    augment = Augmenter()


def the_loop():


    for data in data_loader:

        sensors, labels = data
        # At this point the data was augmented and was send to GPU, the data loader is
        # working in  parallel to upload many juicy images to the memory.

        sensors_augmented = augment(iteration, sensors)

        # All the magic should happen here after.
        forward_pass(data_augmented,labels)

        network.backprop()
        optimizer.do_step()



