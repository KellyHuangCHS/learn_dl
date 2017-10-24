import numpy as np

class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return output

class ReluActivator(object):
    # 前向计算
    def forward(self, weight_input):
        return max(0, weight_input)

    # 计算导数
    def backward(self, output):
        return 1 if output > 0 else 0
