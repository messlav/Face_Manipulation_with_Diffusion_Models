import torch.nn as nn


def freeze_base(params_location):
    for param in params_location.parameters():
        param.requires_grad = False

def freeze_all(model):
    freeze_base(model, model)

def freeze_down5_block1_conv1(model):
     freeze_base(model.down[5].block[1].conv1)

def freeze_down0_block0_conv1(model):
     freeze_base(model.down[0].block[0].conv1)

def freeze_down(model):
     freeze_base(model.down)

def freeze_up(model):
     freeze_base(model.up)

def freeze_temb(model):
     freeze_base(model.temb)

def freeze_all_except_attn(model):
    for param in model.parameters():
        param.requires_grad = False

    for name, module in model.named_modules():
        if 'attn' in name and not isinstance(module, nn.ModuleList):
            for param in module.parameters():
                param.requires_grad = True

def freeze_attn(model):
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) and 'attn' in name:
            for param in module.parameters():
                param.requires_grad = False

def freeze_temb_all(model):
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) and 'temb' in name:
            for param in module.parameters():
                param.requires_grad = False

def freeze_reso_256_128(model):
    for i in range(2):
        for param in model.down[i].parameters():
            param.requires_grad = False

        for param in model.up[i].parameters():
            param.requires_grad = False

def freeze_reso_64_32(model):
    for i in range(2, 4):
        for param in model.down[i].parameters():
            param.requires_grad = False

        for param in model.up[i].parameters():
            param.requires_grad = False

def freeze_reso_16_8(model):
    for i in range(4, 6):
        for param in model.down[i].parameters():
            param.requires_grad = False

        for param in model.up[i].parameters():
            param.requires_grad = False


freeze_function_mapping = {
    'freeze_all': freeze_all,
    'freeze_down5_block1_conv1': freeze_down5_block1_conv1,
    'freeze_down0_block0_conv1': freeze_down0_block0_conv1,
    'freeze_down': freeze_down,
    'freeze_up': freeze_up,
    'freeze_temb': freeze_temb,
    'freeze_all_except_attn': freeze_all_except_attn,
    'freeze_attn': freeze_attn,
    'freeze_temb_all': freeze_temb_all,
    'freeze_reso_256_128': freeze_reso_256_128,
    'freeze_reso_64_32': freeze_reso_64_32,
    'freeze_reso_16_8': freeze_reso_16_8,
}
