'''simple customized rnn in blocks to test output'''
import numpy
import theano
from theano import tensor
from blocks.bricks import Identity
from blocks import initialization
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.bricks import Random, Initializable, MLP, Linear, Rectifier
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal, Uniform
from blocks.bricks.base import application, lazy
import theano.tensor as T

class FeedbackRNN(BaseRecurrent, Initializable):
    def __init__(self, **kwargs):
        super(FeedbackRNN, self).__init__(**kwargs)
        inits = {
            # 'weights_init': IsotropicGaussian(0.01),
            # 'biases_init': Constant(0.),
            'weights_init': Orthogonal(),
            'biases_init': IsotropicGaussian(),
        }
        self.mlp = MLP(activations=[Identity()], dims=[11, 1], name="mlp1", **inits)
        self.children = [self.mlp]

    @property
    def output_dim(self):
        return 1

    def get_dim(self, name):
        if name == 'b':
            return 1
        if name == 'inputs':
            return 10
        else:
            super(FeedbackRNN, self).get_dim(name)

    @recurrent(sequences=['inputs'], contexts=[], states=['b'], outputs=['b'])
    def apply(self, inputs, b=None):
        aa = T.concatenate([inputs, b], axis=0)
        b = self.mlp.apply(aa)
        return b

x = tensor.fmatrix('x')
feedback = FeedbackRNN()
feedback.initialize()
b = feedback.apply(inputs=x, cont=x)
f = theano.function([x], [b], allow_input_downcast=True)
# for states in f(numpy.ones((4, 10))):
#     print(states)
print(f(numpy.ones((4, 10))))