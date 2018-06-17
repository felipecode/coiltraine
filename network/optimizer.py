def adjust_learning_rate(optimizer, num_iters):
    """
    Adjusts the learning rate every epoch based on the selected schedule
    """
    cur_iters = num_iters
    lr = 0.00002
    minlr = 0.0000001
    scheduler = "normal"
    decayinterval = 50000
    decaylevel = 0.5
    if scheduler == "normal":
        while cur_iters >= decayinterval:
            lr = lr * decaylevel
            cur_iters = cur_iters - decayinterval
        lr = max(lr, minlr)

    for param_group in optimizer.param_groups:
        print("New Learning rate is ", lr)
        param_group['lr'] = lr