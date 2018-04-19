





# What do we define as a parameter what not.





def check_integrity():

    # Check the entire execution sample for integrity.
    pass


# The main function maybe we could call it with a default name
def execute():

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


    model = network.Model(Definition)

    criterion = network.Loss()

    optmizer = Optmizer()


    for g_conf.NUMBER_OF_ITERATIONS:

        input_data,labels = data_loader.next_batch()

        input_data = augment()

        output = model(input_data)

        if g_conf.compute_loss:
            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()


    # TODO: DO ALL THE AMAZING LOGGING HERE, as a way to very the status in paralell.
    # THIS SHOULD BE AN INTERELY PARALLEL PROCESS


def main()

    # PARALLEL PROCESS.

