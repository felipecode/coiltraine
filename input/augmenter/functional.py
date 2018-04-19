

def multiply(tensor, scalar):
    # We transpose the tensor to enable to multiply everything in one shot


    result = tensor * scalar.expand_as(tensor)

    return result


def add(tensor, scalar):

    return tensor+scalar.expand_as(tensor)