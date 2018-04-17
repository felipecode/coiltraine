

def multiply(tensor, scalar):

    return tensor*scalar.expand_as(tensor)