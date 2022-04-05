
def get_tensor_dependencies(tensor):

    # If a tensor is passed in, get its op
    try:
        tensor_op = tensor.op
    except:
        tensor_op = tensor

    # Recursively analyze inputs
    dependencies = []
    for inp in tensor_op.inputs:
        new_d = get_tensor_dependencies(inp)
        non_repeated = [d for d in new_d if d not in dependencies]
        dependencies = dependencies + non_repeated

    # If we've reached the "end", return the op's name
    if len(tensor_op.inputs) == 0:
        dependencies = [tensor_op.name]

    # Return a list of tensor op names
    return dependencies
