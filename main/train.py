





# What do we define as a parameter what not.





def check_integrity():

    # Check the entire execution sample for integrity.
    pass


# The main function maybe we could call it with a default name
def execute():

    # But now this is pytorch


    model = network.Model(Definition)
    optmizer = Optmizer()

    criterion  = network.Loss(Type) # You could send it as a parameter here ...

    schedule = Scheduler()



    for g_conf.NUMBER_OF_ITERATIONS:

        input_data,labels = scheduler.next_batch()

        output = model(input_data)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
    # TODO: DO ALL THE AMAZING LOGGING HERE, as a way to very the status in paralell.
    # THIS SHOULD BE AN INTERELY PARALLEL PROCESS


def main()

    # PARALLEL PROCESS.

