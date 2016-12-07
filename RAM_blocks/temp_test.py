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
import theano.tensor as T

theano.config.floatX = 'float32'
floatX = theano.config.floatX

class FeedbackRNN(BaseRecurrent):
    def __init__(self, **kwargs):
        super(FeedbackRNN, self).__init__(**kwargs)
        inits = {
            # 'weights_init': IsotropicGaussian(0.01),
            # 'biases_init': Constant(0.),
            'weights_init': Orthogonal(),
            'biases_init': IsotropicGaussian(),
        }
        self.mlp = MLP(activations=[Identity()], dims=[11, 2], name="mlp", **inits)
        self.children = [self.mlp]

    def get_dim(self, name):
        if name == 'b':
            return 1
        if name == 'inputs':
            return 1

    @recurrent(sequences=['inputs'], contexts=[],states=['b'],outputs=['b'])
    def apply(self, inputs, b=None):
        aa = T.concatenate([inputs, b[0]], axis=0)
        b = self.mlp.apply(aa)
        return b

x = tensor.lmatrix('x')
feedback = FeedbackRNN()
feedback.initialize()
b = feedback.apply(inputs=x)
f = theano.function([x], [b])
for states in f(numpy.ones((4, 10))):
    print(states)