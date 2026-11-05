import correlation_cuda
import torch
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.autograd import Variable


class CorrelationFunction(Function):

    def __init__(self, pad_size=3, kernel_size=3, max_displacement=20, stride1=1, stride2=2, corr_multiply=1):
        super(CorrelationFunction, self).__init__()
        self.pad_size = pad_size
        self.kernel_size = kernel_size
        self.max_displacement = max_displacement
        self.stride1 = stride1
        self.stride2 = stride2
        self.corr_multiply = corr_multiply
        # self.out_channel = ((max_displacement/stride2)*2 + 1) * ((max_displacement/stride2)*2 + 1)

    @staticmethod
    def forward(ctx, input1, input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply):
        ctx.save_for_backward(input1, input2)
        ctx.params = (pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()
            output = input1.new()

            correlation_cuda.forward(input1, input2, rbot1, rbot2, output,
                                     pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply = ctx.params

        with torch.cuda.device_of(input1):
            rbot1 = input1.new()
            rbot2 = input2.new()

            grad_input1 = input1.new()
            grad_input2 = input2.new()

            correlation_cuda.backward(input1, input2, rbot1, rbot2, grad_output, grad_input1, grad_input2,
                                      pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply)

        return grad_input1, grad_input2, pad_size, kernel_size, max_displacement, stride1, stride2, corr_multiply


class Correlation(Module):
    def __init__(self, pad_size=0, kernel_size=0, max_displacement=0, stride1=1, stride2=2, corr_multiply=1):
        super(Correlation, self).__init__()
        self.pad_size = Variable(torch.tensor(pad_size, dtype=torch.float32), requires_grad=True)
        self.kernel_size = Variable(torch.tensor(kernel_size, dtype=torch.float32), requires_grad=True)
        self.max_displacement = Variable(torch.tensor(max_displacement, dtype=torch.float32), requires_grad=True)
        self.stride1 = Variable(torch.tensor(stride1, dtype=torch.float32), requires_grad=True)
        self.stride2 = Variable(torch.tensor(stride2, dtype=torch.float32), requires_grad=True)
        self.corr_multiply = Variable(torch.tensor(corr_multiply, dtype=torch.float32), requires_grad=True)

    def forward(self, input1, input2):
        # result = CorrelationFunction(self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)(input1, input2)
        result = CorrelationFunction.apply(input1, input2, self.pad_size, self.kernel_size, self.max_displacement,self.stride1, self.stride2, self.corr_multiply)

        return result