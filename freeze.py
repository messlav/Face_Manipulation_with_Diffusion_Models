def freeze_down5_block1_conv1(model):
    for param in model.down[5].block[1].conv1.parameters():
        param.requires_grad = False

    return model


freeze_function_mapping = {
    'freeze_down5_block1_conv1': freeze_down5_block1_conv1
}
