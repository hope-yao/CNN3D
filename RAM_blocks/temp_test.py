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
            'weights_init': Constant(1.),
            'biases_init': Constant(0.),
        }
        self.mlp = MLP(activations=[Identity()], dims=[13, 2], name="mlp", **inits)
        self.children = [self.mlp]

    def get_dim(self, name):
        if name == 'b':
            return 1
        if name == 'c':
            return 1
        if name == 'inputs':
            return 1
        # if name == 'dummy':
        #     return 1

    @recurrent(sequences=['dummy'], contexts=['inputs'],states=['b'],outputs=['c'])
    def apply(self, inputs, dummy, b=tensor.ones((1,2))):
        inputs = T.concatenate([inputs, b], axis=1)
        c = self.mlp.apply(inputs)
        b = c
        return c

x = tensor.lmatrix('x') # batch * input_dim
feedback = FeedbackRNN()
feedback.initialize()

u = theano.tensor.zeros(4) #(self.n_iter, batch_size, 1)
b = feedback.apply(inputs=x, dummy=u)
f = theano.function([x], [b])

l = f(numpy.ones((1,11),dtype='int64'))
print(l)