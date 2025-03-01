# import torch
# from torch import Tensor, nn
# from torch._overrides import has_torch_function, handle_torch_function
# import torch.nn.functional as F
#
#
# def linear(input, weight, bias=None):
#     # type: (Tensor, Tensor, Optional[Tensor]) -> Tensor
#     r"""
#     Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
#
#     Shape:
#
#         - Input: :math:`(N, *, in\_features)` where `*` means any number of
#           additional dimensions
#         - Weight: :math:`(out\_features, in\_features)`
#         - Bias: :math:`(out\_features)`
#         - Output: :math:`(N, *, out\_features)`
#     """
#     tens_ops = (input, weight)
#     if not torch.jit.is_scripting():
#         if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
#             return handle_torch_function(linear, tens_ops, input, weight, bias=bias)
#     if input.dim() == 2 and bias is not None:
#         # fused op is marginally faster
#         ret = torch.addmm(bias, input, weight.t())
#     else:
#         weight = F.normalize(weight,dim=1)
#         print("quanzhongdecanshu",weight.shape)
#         output = input.matmul(weight.t())
#         if bias is not None:
#             output += bias
#         ret = output
#     return ret
# class MyLinear(nn.Linear):
#     def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
#         super(MyLinear, self).__init__(in_features,out_features,bias)
#     def forward(self, input: Tensor) -> Tensor:
#         return linear(input,self.weight,self.bias)
