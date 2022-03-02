def model_size(model, unit="MB"):
    """Computes the model's size"""
    # From here: https://discuss.pytorch.org/t/finding-model-size/130275

    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    if unit == "MB":
        d = 2**20
    elif unit == "KB":
        d = 2**10
    elif unit == "B":
        d = 1
    else:
        raise ValueError('Unit must be one of "MB", "KB" or "B"')

    size_all = (param_size + buffer_size) / d
    return round(size_all, 3)
