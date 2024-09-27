import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.nn import Cell

def test_my_softmax(target):
    z = np.array([1., 2., 3., 4.])
    a = target(z)
    softmax = nn.Softmax(axis=-1)
    atf = softmax(Tensor(z)).asnumpy()


    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"
    
    z = np.array([np.log(0.1)] * 10)
    a = target(z)
    softmax = nn.Softmax(axis=-1)
    atf = softmax(Tensor(z)).asnumpy()
 

    assert np.allclose(a, atf, atol=1e-10), f"Wrong values. Expected {atf}, got {a}"
    
    print("\033[92m All tests passed.")
    
def test_model(target, classes, input_size):
    # target.build(input_shape=(None,input_size))
    
    assert len(target) == 3, \
        f"Wrong number of layers. Expected 3 but got {len(target.layers)}"
    assert target[0].weight.shape[1] == input_size, \
        f"Wrong input shape. Expected [None,  {input_size}] but got [None, {target[0].weight.shape[1]}]"
    expected = [[nn.Dense, [None, 25], nn.ReLU()],
                [nn.Dense, [None, 15], nn.ReLU()],
                [nn.Dense, [None, classes], None]]

    for i,layer in enumerate(target):
        assert type(layer) == expected[i][0], \
            f"Wrong type in layer {i}. Expected {expected[i][0]} but got {type(layer)}"
        assert layer.weight.shape[0] == expected[i][1][-1], \
            f"Wrong number of units in layer {i}. Expected {expected[i][1]} but got [None,{layer.weight.shape[0]}]"
        if expected[i][2] is not None:
            assert isinstance(layer.activation, type(expected[i][2])), \
                f"Wrong activation in layer {i}. Expected {expected[i][2]} but got {type(layer.activation)}"

    print("\033[92mAll tests passed!")
    