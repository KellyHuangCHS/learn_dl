import numpy as np
from activators import ReluActivator, IdentityActivator

class ConvLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, filter_number,
                 zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = \
            calculate_output_size(
                self.input_width, filter_width, zero_padding, stride
            )
        self.output_height = \
            calculate_output_size(
                self.input_height, filter_height, zero_padding, stride
            )
        # print((self.filter_number, self.output_height, self.output_width))
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,
                                      filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array, filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
            element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        '''
        计算传递到上一层的sensitivity map
        :param sensitivity_array: 本层的sensitivity map
        :param activator: 上一层的激活函数
        '''
        expanded_array = self.expand_sensitivity_array_map(sensitivity_array)
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        self.delta_array = self.create_delta_array()
        for f in range(self.filter_number):
            filter = self.filters[f]
            flipped_weights = np.array([np.rot90(i, 2) for i in filter.get_weights()])
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        expanded_array = self.expand_sensitivity_array_map(sensitivity_array)
        for f in  range(self.filter_number):
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            filter.bias_grad = expanded_array[f].sum()

    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)

    def expand_sensitivity_array_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        expanded_width = (self.input_width -
                          self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height -
                           self.filter_height + 2 * self.zero_padding + 1)
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:,i_pos,j_pos] = \
                    sensitivity_array[:,i,j]
        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))


def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)


# 对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

# 卷积的方法
def conv(input_array, kernel_array, output_array, stride, bias):
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                get_patch(input_array, i, j, kernel_width, kernel_height, stride) * kernel_array
            ).sum() + bias

# 获取卷积区域
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
            start_i : start_i + filter_height,
            start_j : start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:,
               start_i : start_i + filter_height,
               start_j : start_j + filter_width]

# 为数组增加zero padding
def padding(input_array, zp):
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            # 这3个参数都是浮点数
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            # 这个类型错误不知道是为什么
            padded_array = np.zeros((
                input_depth,
                input_height + 2.0 * zp,
                input_width + 2.0 * zp
            ))
            padded_array[:,
                zp : zp + input_height,
                zp : zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp
            ), dtype=np.float64)
            padded_array[
                zp : zp + input_height,
                zp : zp + input_width
            ] = input_array
            return padded_array

class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weight:\n%s\nbias:\n%s'%(repr(self.weights), repr((self.bias)))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

def init_test():
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],
         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
         [[1,0,2],
          [0,0,0],
          [1,2,1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],
         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],
         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weight = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],
         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],
         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl

def test():
    a, b, cl = init_test()
    cl.forward(a)
    print(cl.output_array)

def test_bp():
    a, b, cl = init_test()
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print(cl.filters[0])
    print(cl.filters[1])

def gradient_check():
    error_function = lambda o: o.sum()
    a, b, cl = init_test()
    cl.forward(a)
    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)
    cl.backward(a, sensitivity_array, IdentityActivator())
    epsilon = 10e-4
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d, i, j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weight[d, i, j] -= 2*epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)
                expect_grad = (err1 - err2)/(2*epsilon)
                cl.filters[0].weights[d, i, j] += epsilon
                print('weights(%d,%d,%d): expected - actural %f - %f' % (
                    d, i, j, expect_grad, cl.filters[0].weights_grad[d,i,j]))

class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width, filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width - filter_width) / self.stride + 1
        self.output_height = (input_height - filter_height) / self.stride + 1
        self.output_array = np.zeros((int(self.channel_number), int(self.output_height), int(self.output_width)))

    def forward(self, input_array):
        for d in range(int(self.channel_number)):
            for i in range(int(self.output_height)):
                for j in range(int(self.output_width)):
                    self.output_array[i][j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max())

    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(int(self.channel_number)):
            for i in range(int(self.output_height)):
                for j in range(int(self.output_width)):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l = get_max_index(patch_array)
                    # 此处是因为向量维度不匹配导致的错误
                    self.delta_array[d, j * self.stride - k, j * self.stride - l] = sensitivity_array

def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0,0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array [i,j] > max_value:
                max_value = array[i, j]
                max_i = i
                max_j = j
    return max_i, max_j

def init_pool_test():
    a = np.array(
        [[[1,1,2,4],
          [5,6,7,8],
          [3,2,1,0],
          [1,2,3,4]],
         [[0,1,2,3],
          [4,5,6,7],
          [8,9,0,1],
          [3,4,5,6]]], dtype=np.float64)

    b = np.array(
        [[[1,2],
          [2,4]],
         [[3,5],
          [8,2]]], dtype=np.float64)

    mpl = MaxPoolingLayer(4, 4, 2, 2, 2, 2)

    return a, b, mpl

def test_pool():
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print('input array:\n%s\noutput array:\n%s' % (a,
        mpl.output_array))

def test_pool_bp():
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print('input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (
        a, b, mpl.delta_array))