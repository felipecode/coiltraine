

def multiply(tensor, scalar):
    # We permute the tensor to enable to multiply everything in one shot
    if len(tensor.shape) > 2:  # If the dimension is bigger than 2 is safe to say...
        tensor = tensor.permute(3, 2, 1, 0)

    result = tensor * scalar.expand_as(tensor)
    if len(tensor.shape) > 2:
        result = result.permute(3, 2, 1, 0)

    return result


def add(tensor, scalar):

    return tensor+scalar.expand_as(tensor)