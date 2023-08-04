#!/usr/bin/env python
# coding: utf-8

# QAT

import torch
from torch.ao.quantization import QuantStub, DeQuantStub


class Conv(torch.nn.Module):
    # Standard convolution
    default_act = torch.nn.SiLU()  # default activation

    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.act = self.default_act
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.dequant(x)
        return x

    def forward_fuse(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.dequant(x)
        return x
    
# create a model instance
model_fp32 = Conv()
model_fp32.eval()
print (model_fp32)
model_fp32.qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'bn']])
model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)

model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

res = model_int8(input_fp32)

