

def multiply(tensor, scalar):

    return tensor*scalar.expand_as(tensor)


def add(tensor, scalar):

    return tensor+scalar.expand_as(tensor)