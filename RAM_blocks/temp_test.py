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
    def __init__(self, n_itr, bs, input_len, **kwargs):
        super(FeedbackRNN, self).__init__(**kwargs)
        inits = {
            # 'weights_init': IsotropicGaussian(0.01),
            # 'biases_init': Constant(0.),
            'weights_init': Constant(1.),
            'biases_init': Constant(0.),
        }
        self.mlp = MLP(activations=[Identity()], dims=[13, 2], name="mlp", **inits)
        self.children = [self.mlp]

        self.n_itr = n_itr
        self.bs = bs
        self.input_len = input_len

    def get_dim(self, name):
        if name == 'b':
            return 3
        if name == 'c':
            return 3
        if name == 'inputs':
            return 3
        # if name == 'dummy':
        #     return 1

    @recurrent(sequences=['dummy'], contexts=['inputs'],states=['b'],outputs=['c','test1','test2'])
    def apply(self, inputs, dummy, b=None):
        if b == None:
            b = tensor.ones((self.bs,2)) # n_itr repeated as the first dimension automatically
            b = b * 2
        test1 =inputs
        test2 = b
        # test2 = T.concatenate([inputs, b], axis=2) #WHY  ValueError: Join argument "axis" is out of range (given input dimensions)
        # c = self.mlp.apply(inputs)
        c = inputs
        return c,test1,test2


a = tensor.ones((4,3 , 13))
b = tensor.ones((4,3 , 2))
c = T.concatenate([a,b],axis=2)
f = theano.function([a,b], [c])
a_ = numpy.ones((4,3 , 13),dtype=floatX)
b_ = numpy.ones((4,3 , 2),dtype=floatX)
c_ = f(a_,b_) ##WHY INT64??
print(numpy.asarray(a_).shape)
print(numpy.asarray(c_).shape)

n_itr = 4
bs = 3
input_len = 11
x = tensor.lmatrix('x') # batch * input_dim
feedback = FeedbackRNN(n_itr,bs,input_len)
feedback.initialize()

u = theano.tensor.zeros(n_itr,dtype=floatX) #(self.n_iter, batch_size, 1)
c,test1,test2 = feedback.apply(inputs=x, dummy=u)
f = theano.function([x], [c,test1,test2])

# c,inputs = f(numpy.ones((bs ,input_len),dtype='int64'))
c,test1,test2= f(numpy.ones((bs ,input_len),dtype='int64')) ##WHY INT64??
print(c)
print(test1.shape,test2.shape)
