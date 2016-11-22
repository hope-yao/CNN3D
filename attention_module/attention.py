#!/ysr/bin/env python 

from __future__ import division

import numpy as np

import theano 
import theano.tensor as T

from theano import tensor
from theano.compile.sharedvalue import shared
floatX = theano.config.floatX

#-----------------------------------------------------------------------------
        
def my_batched_dot(A, B):
    """Batched version of dot-product.

    For A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4] this
    is \approx equal to:

    for i in range(dim_1):
        C[i] = tensor.dot(A[i], B[i])

    Returns
    -------
        C : shape (dim_1 \times dim_2 \times dim_4)
    """
    print(A.shape, B.shape)
    C = A.dimshuffle([0,1,2,'x']) * B.dimshuffle([0,'x',1,2])
    return C.sum(axis=-2)


# -----------------------------------------------------------------------------

class ZoomableAttentionWindow(object):
    def __init__(self, channels, img_height, img_width, N):
        """A zoomable attention window for images.
        Parameters
        ----------
        channels : int
        img_heigt, img_width : int
            shape of the images
        N :
            $N \times N$ attention window size
        """
        self.channels = channels
        self.img_height = img_height
        self.img_width = img_width
        self.N = N

    def filterbank_matrices(self, center_y, center_x, delta, sigma):
        """Create a Fy and a Fx

        Parameters
        ----------
        center_y : T.vector (shape: batch_size)
        center_x : T.vector (shape: batch_size)
            Y and X center coordinates for the attention window
        delta : T.vector (shape: batch_size)
        sigma : T.vector (shape: batch_size)

        Returns
        -------
            FY : T.fvector (shape: )
            FX : T.fvector (shape: )
        """
        tol = 1e-4
        N = self.N

        rng = T.arange(N, dtype=floatX) - N / 2. + 0.5  # e.g.  [1.5, -0.5, 0.5, 1.5]

        muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x']) * rng
        muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x']) * rng

        a = tensor.arange(self.img_width, dtype=floatX)
        b = tensor.arange(self.img_height, dtype=floatX)

        FX = tensor.exp(-(a - muX.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
        FY = tensor.exp(-(b - muY.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
        FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        return FY, FX

    def read(self, images, center_y, center_x, delta, sigma):
        """Extract a batch of attention windows from the given images.
        Parameters
        ----------
        images : :class:`~tensor.TensorVariable`
            Batch of images with shape (batch_size x img_size). Internally it
            will be reshaped to a (batch_size, img_height, img_width)-shaped
            stack of images.
        center_y : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Distance between extracted grid points.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian readout kernel.
            Expected shape: (batch_size,)
        Returns
        -------
        windows : :class:`~tensor.TensorVariable`
            extracted windows of shape: (batch_size x N**2)
        """
        N = self.N
        channels = self.channels
        batch_size = images.shape[0]

        # Reshape input into proper 2d images
        I = images.reshape((batch_size * channels, self.img_height, self.img_width))

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        FY = T.repeat(FY, channels, axis=0)
        FX = T.repeat(FX, channels, axis=0)

        # apply to the batch of images
        W = my_batched_dot(my_batched_dot(FY, I), FX.transpose([0, 2, 1]))

        return W.reshape((batch_size, channels * N * N))

    def read_large(self, images, center_y, center_x):
        N = self.N
        channels = self.channels
        batch_size = images.shape[0]

        delta = T.ones([batch_size], 'float32')
        sigma = T.ones([batch_size], 'float32')


        # Reshape input into proper 2d images
        I = images.reshape( (batch_size*channels, self.img_height, self.img_width) )

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        FY = T.repeat(FY, channels, axis=0)
        FX = T.repeat(FX, channels, axis=0)

        # apply to the batch of images
        W = my_batched_dot(my_batched_dot(FY, I), FX.transpose([0,2,1]))
        # W = W.reshape((batch_size * channels, N, N))

        # Max hack: convert back to an image
        # II = my_batched_dot(my_batched_dot(FY.transpose([0, 2, 1]), W), FX)

        # return II.reshape((batch_size, channels*self.img_height*self.img_width))

        return W.reshape((batch_size, channels*N*N))

    def write(self, windows, center_y, center_x, delta, sigma):
        """Write a batch of windows into full sized images.
        Parameters
        ----------
        windows : :class:`~tensor.TensorVariable`
            Batch of images with shape (batch_size x N*N). Internally it
            will be reshaped to a (batch_size, N, N)-shaped
            stack of images.
        center_y : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        center_x : :class:`~tensor.TensorVariable`
            Center coordinates for the attention window.
            Expected shape: (batch_size,)
        delta : :class:`~tensor.TensorVariable`
            Distance between extracted grid points.
            Expected shape: (batch_size,)
        sigma : :class:`~tensor.TensorVariable`
            Std. dev. for Gaussian readout kernel.
            Expected shape: (batch_size,)
        Returns
        -------
        images : :class:`~tensor.TensorVariable`
            extracted windows of shape: (batch_size x img_height*img_width)
        """
        N = self.N
        channels = self.channels
        batch_size = windows.shape[0]

        # Reshape input into proper 2d windows
        W = windows.reshape((batch_size * channels, N, N))

        # Get separable filterbank
        FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

        FY = T.repeat(FY, channels, axis=0)
        FX = T.repeat(FX, channels, axis=0)

        # apply...
        I = my_batched_dot(my_batched_dot(FY.transpose([0, 2, 1]), W), FX)

        return I.reshape((batch_size, channels * self.img_height * self.img_width))

    def nn2att(self, l):
        """Convert neural-net outputs to attention parameters

        Parameters
        ----------
        layer : :class:`~tensor.TensorVariable`
            A batch of neural net outputs with shape (batch_size x 5)

        Returns
        -------
        center_y : :class:`~tensor.TensorVariable`
        center_x : :class:`~tensor.TensorVariable`
        delta : :class:`~tensor.TensorVariable`
        sigma : :class:`~tensor.TensorVariable`
        gamma : :class:`~tensor.TensorVariable`
        """
        center_y = l[:, 0]
        center_x = l[:, 1]
        log_delta = l[:, 2]
        log_sigma = l[:, 3]
        log_gamma = l[:, 4]

        delta = T.exp(log_delta)
        sigma = T.exp(log_sigma / 2.)
        gamma = T.exp(log_gamma).dimshuffle(0, 'x')

        # normalize coordinates
        center_x = (center_x + 1.) / 2. * self.img_width
        center_y = (center_y + 1.) / 2. * self.img_height
        delta = (max(self.img_width, self.img_height) - 1) / (self.N - 1) * delta

        return center_y, center_x, delta, sigma, gamma

# -----------------------------------------------------------------------------

class ZoomableAttentionWindow3d(object):
        def __init__(self, channels, img_height, img_width, img_depth, N):
            """A zoomable attention window for images.

            Parameters
            ----------
            channels : int
            img_heigt, img_width : int
                shape of the images
            N :
                $N \times N$ attention window size
            """
            self.channels = channels
            self.img_height = img_height
            self.img_width = img_width
            self.img_depth = img_depth
            self.N = N

        def read(self, images, center_y, center_x, delta, sigma):
            """Extract a batch of attention windows from the given images.

            Parameters
            ----------
            images : :class:`~tensor.TensorVariable`
                Batch of images with shape (batch_size x img_size). Internally it
                will be reshaped to a (batch_size, img_height, img_width)-shaped
                stack of images.
            center_y : :class:`~tensor.TensorVariable`
                Center coordinates for the attention window.
                Expected shape: (batch_size,)
            center_x : :class:`~tensor.TensorVariable`
                Center coordinates for the attention window.
                Expected shape: (batch_size,)
            delta : :class:`~tensor.TensorVariable`
                Distance between extracted grid points.
                Expected shape: (batch_size,)
            sigma : :class:`~tensor.TensorVariable`
                Std. dev. for Gaussian readout kernel.
                Expected shape: (batch_size,)

            Returns
            -------
            windows : :class:`~tensor.TensorVariable`
                extracted windows of shape: (batch_size x N**2)
            """
            N = self.N
            channels = self.channels
            batch_size = images.shape[0]

            # Reshape input into proper 2d images
            I = images.reshape((batch_size * channels, self.img_height, self.img_width))

            # Get separable filterbank
            FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

            FY = T.repeat(FY, channels, axis=0)
            FX = T.repeat(FX, channels, axis=0)

            # apply to the batch of images
            W = my_batched_dot(my_batched_dot(FY, I), FX.transpose([0, 2, 1]))

            return W.reshape((batch_size, channels, N, N))
        #
        # def read_large(self, images, center_x, center_y, center_z):
        #     N = self.N
        #     channels = self.channels
        #     batch_size = images.shape[0]
        #
        #     # Hope: get 3D gaussian filter, with truncation at the boundary of the filter
        #     F = tensor.zeros((self.img_height, self.img_width, self.img_depth))
        #     cx = T.cast(center_x[0]*self.img_height, 'int64')
        #     cy = T.cast(center_y[0]*self.img_width, 'int64')
        #     cz = T.cast(center_z[0]*self.img_depth, 'int64')
        #     if 0:
        #         posx = tensor.arange( T.cast(cx - N / 2, 'int64'),  T.cast(cx + N / 2 + 1, 'int64'))
        #         posy = tensor.arange( T.cast(cy - N / 2, 'int64'),  T.cast(cy + N / 2 + 1, 'int64'))
        #         posz = tensor.arange( T.cast(cz - N / 2, 'int64'),  T.cast(cz + N / 2 + 1, 'int64'))
        #         F = tensor.inc_subtensor(F[posx,posy,posz], 1)
        #         F = T.repeat(F, batch_size *channels, axis=0)
        #         return F.reshape((batch_size, channels * self.img_height * self.img_width))
        #     else:
        #         # posx = tensor.arange( 0, self.img_height )
        #         # posy = tensor.arange( 0, self.img_width )
        #         # posz = tensor.arange( 0, self.img_depth )
        #         # for posx in range(self.img_height):
        #         #     for posy in range(self.img_width):
        #         #         for posz in range(self.img_depth):
        #         #             sqr_dis = (cx-posx)**2+(cy-posy)**2+(cz-posz)**2
        #         #             sigma = 1.0
        #         #             vinc = tensor.exp(-sqr_dis/ 2. / sigma ** 2)
        #         #             F = tensor.inc_subtensor(F[posx,posy,posz],vinc)
        #         # tol = 1e-4
        #         # F = F / (F.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
        #         # FF = T.repeat(F, batch_size, axis=0)
        #         # FF = FF.reshape((batch_size, channels*self.img_depth*self.img_height*self.img_width))
        #         FF = tensor.ones((batch_size, channels*self.img_depth*self.img_height*self.img_width))*center_x
        #         return FF

        def filterbank_matrices(self, center_y, center_x, center_z, delta, sigma):
            tol = 1e-4
            N = self.N

            rng = T.arange(N, dtype=floatX) - N / 2. + 0.5  # e.g.  [1.5, -0.5, 0.5, 1.5]

            muX = center_x.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x']) * rng
            muY = center_y.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x']) * rng
            muZ = center_z.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x']) * rng

            a = tensor.arange(self.img_width, dtype=floatX)
            b = tensor.arange(self.img_height, dtype=floatX)
            c = tensor.arange(self.img_depth, dtype=floatX)

            FX = tensor.exp(-(a - muX.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
            FY = tensor.exp(-(b - muY.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
            FZ = tensor.exp(-(c - muZ.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
            FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
            FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
            FZ = FZ / (FZ.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

            return FY, FX, FZ

        def read_large(self, images, center_y, center_x, center_z):
            N = self.N
            channels = self.channels
            batch_size = images.shape[0]
            print(images.ndim)

            delta = T.ones([batch_size], 'float32')
            sigma = T.ones([batch_size], 'float32')

            # Reshape input into proper 3d images
            I = images.reshape((batch_size, self.img_height, self.img_width, self.img_depth))

            # Get separable filterbank
            FY, FX, FZ = self.filterbank_matrices(center_y, center_x, center_z, delta, sigma)

            FY = T.repeat(FY, channels, axis=0)
            FX = T.repeat(FX, channels, axis=0)
            FZ = T.repeat(FZ, channels, axis=0)

            # apply to the batch of images
            I1 = I.reshape((batch_size, self.img_height, self.img_width*self.img_depth))
            IY = my_batched_dot(FY, I1).reshape((batch_size, N, self.img_width, self.img_depth))
            I2 = IY.dimshuffle([0,1,3,2]).reshape((batch_size, N*self.img_depth, self.img_width))
            IX = my_batched_dot(I2,FX.transpose([0, 2, 1])).reshape((batch_size, N, self.img_depth, N)).dimshuffle([0,1,3,2])
            I3 = IX.dimshuffle([0,3,1,2]).reshape((batch_size, self.img_depth, N*N))
            IZ = my_batched_dot(FZ, I3).reshape((batch_size, N, N, N)).dimshuffle([0,2,3,1])
            #
            # Max hack: convert back to an image
            IYY = my_batched_dot(FY.transpose([0, 2, 1]), IZ.reshape((batch_size, N, N*N))).reshape((batch_size, self.img_height, N, N))
            I11 = IYY.dimshuffle([0,1,3,2]).reshape((batch_size, self.img_height*N, N))
            IXX = my_batched_dot(I11, FX).reshape((batch_size, self.img_height, N, self.img_width)).dimshuffle([0,1,3,2])
            I22 = IXX.dimshuffle([0,3,1,2]).reshape((batch_size, N, self.img_height*self.img_width))
            IZZ = my_batched_dot(FZ.transpose([0, 2, 1]), I22).reshape((batch_size, self.img_depth, self.img_height, self.img_width)).dimshuffle([0,2,3,1])

            return IZZ.reshape((batch_size, self.img_height*self.img_width*self.img_depth))

            # IYY = my_batched_dot(FY.transpose([0, 2, 1]), IY.reshape((batch_size, N, self.img_width*self.img_depth))).reshape((batch_size, self.img_height, self.img_width, self.img_depth))
            # return IYY.reshape((batch_size, self.img_height*self.img_width*self.img_depth))

            # return IZ.reshape((batch_size, N*N*N))

        def write(self, windows, center_y, center_x, delta, sigma):
            """Write a batch of windows into full sized images.

            Parameters
            ----------
            windows : :class:`~tensor.TensorVariable`
                Batch of images with shape (batch_size x N*N). Internally it
                will be reshaped to a (batch_size, N, N)-shaped
                stack of images.
            center_y : :class:`~tensor.TensorVariable`
                Center coordinates for the attention window.
                Expected shape: (batch_size,)
            center_x : :class:`~tensor.TensorVariable`
                Center coordinates for the attention window.
                Expected shape: (batch_size,)
            delta : :class:`~tensor.TensorVariable`
                Distance between extracted grid points.
                Expected shape: (batch_size,)
            sigma : :class:`~tensor.TensorVariable`
                Std. dev. for Gaussian readout kernel.
                Expected shape: (batch_size,)

            Returns
            -------
            images : :class:`~tensor.TensorVariable`
                extracted windows of shape: (batch_size x img_height*img_width)
            """
            N = self.N
            channels = self.channels
            batch_size = windows.shape[0]

            # Reshape input into proper 2d windows
            W = windows.reshape((batch_size * channels, N, N))

            # Get separable filterbank
            FY, FX = self.filterbank_matrices(center_y, center_x, delta, sigma)

            FY = T.repeat(FY, channels, axis=0)
            FX = T.repeat(FX, channels, axis=0)

            # apply...
            I = my_batched_dot(my_batched_dot(FY.transpose([0, 2, 1]), W), FX)

            return I.reshape((batch_size, channels * self.img_height * self.img_width))

        # Max hack
        def write_small(self, windows, center_y, center_x, sigma):
            N = self.N
            channels = self.channels
            batch_size = windows.shape[0]
            # Reshape input into proper 2d windows
            W = windows.reshape((batch_size * channels, N, N))
            # Get separable filterbank
            FY, FX = self.filterbank_matrices_small(center_y, center_x, sigma)
            FY = T.repeat(FY, channels, axis=0)
            FX = T.repeat(FX, channels, axis=0)
            # apply...
            I = my_batched_dot(my_batched_dot(FY.transpose([0, 2, 1]), W), FX)
            return I.reshape((batch_size, channels * self.img_height * self.img_width))

        def filterbank_matrices_small(self, center_y, center_x, sigma):
            tol = 1e-4
            N = self.N
            rng = T.arange(N, dtype=floatX) - N / 2. + 0.5  # e.g.  [1.5, -0.5, 0.5, 1.5]
            muX = center_x.dimshuffle([0, 'x']) + rng
            muY = center_y.dimshuffle([0, 'x']) + rng
            a = tensor.arange(self.img_width, dtype=floatX)
            b = tensor.arange(self.img_height, dtype=floatX)
            FX = tensor.exp(-(a - muX.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
            FY = tensor.exp(-(b - muY.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
            FX = FX / (FX.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
            FY = FY / (FY.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)
            return FY, FX

        def nn2att(self, l):
            """Convert neural-net outputs to attention parameters

            Parameters
            ----------
            layer : :class:`~tensor.TensorVariable`
                A batch of neural net outputs with shape (batch_size x 5)

            Returns
            -------
            center_y : :class:`~tensor.TensorVariable`
            center_x : :class:`~tensor.TensorVariable`
            delta : :class:`~tensor.TensorVariable`
            sigma : :class:`~tensor.TensorVariable`
            gamma : :class:`~tensor.TensorVariable`
            """
            center_y = l[:, 0]
            center_x = l[:, 1]
            log_delta = l[:, 2]
            log_sigma = l[:, 3]
            log_gamma = l[:, 4]

            delta = T.exp(log_delta)
            sigma = T.exp(log_sigma / 2.)
            gamma = T.exp(log_gamma).dimshuffle(0, 'x')

            # normalize coordinates
            center_x = (center_x + 1.) / 2. * self.img_width
            center_y = (center_y + 1.) / 2. * self.img_height
            delta = (max(self.img_width, self.img_height) - 1) / (self.N - 1) * delta

            return center_y, center_x, delta, sigma, gamma

        def nn2att_const_gamma(self, l):
            """Convert neural-net outputs to attention parameters

            Parameters
            ----------
            layer : :class:`~tensor.TensorVariable`
                A batch of neural net outputs with shape (batch_size x 5)

            Returns
            -------
            center_y : :class:`~tensor.TensorVariable`
            center_x : :class:`~tensor.TensorVariable`
            delta : :class:`~tensor.TensorVariable`
            sigma : :class:`~tensor.TensorVariable`
            gamma : :class:`~tensor.TensorVariable`
            """
            center_x = l[:, 0]
            center_y = l[:, 1]
            center_z = l[:, 2]
            # log_delta = l[:, 2]
            # log_sigma = l[:, 3]

            # delta = T.exp(log_delta)
            # sigma = T.exp(log_sigma / 2.)

            # normalize coordinates
            center_x = (center_x + 1.) / 2. * self.img_width
            center_y = (center_y + 1.) / 2. * self.img_height
            center_z = (center_z + 1.) / 2. * self.img_depth
            # delta = (max(self.img_width, self.img_height) - 1) / (self.N - 1) * delta

            return center_x, center_y, center_z


#=============================================================================

if __name__ == "__main__":
    dim = 2
    if dim == 2:
        from PIL import Image

        N = 40
        channels = 3
        height = 480
        width = 640

        # ------------------------------------------------------------------------
        att = ZoomableAttentionWindow(channels, height, width, N)

        I_ = T.matrix()
        center_y_ = T.vector()
        center_x_ = T.vector()
        delta_ = T.vector()
        sigma_ = T.vector()
        W_ = att.read(I_, center_y_, center_x_, delta_, sigma_)
        do_read = theano.function(inputs=[I_, center_y_, center_x_, delta_, sigma_],
                                  outputs=W_, allow_input_downcast=True)
        # W_ = att.read_large(I_, center_y_, center_x_)
        # do_read = theano.function(inputs=[I_, center_y_, center_x_],
        #                           outputs=W_, allow_input_downcast=True)


        W_ = T.matrix()
        center_y_ = T.vector()
        center_x_ = T.vector()
        delta_ = T.vector()
        sigma_ = T.vector()
        I_ = att.write(W_, center_y_, center_x_, delta_, sigma_)

        do_write = theano.function(inputs=[W_, center_y_, center_x_, delta_, sigma_],
                                   outputs=I_, allow_input_downcast=True)

        # ------------------------------------------------------------------------

        I = Image.open("cat.jpg")
        I = I.resize((640, 480))  # .convert('L')

        I = np.asarray(I).transpose([2, 0, 1])
        I = I.reshape((channels * width * height))
        I = I / 255.

        center_y = 200.5
        center_x = 330.5
        delta = 5.
        sigma = 2.


        def vectorize(*args):
            return [a.reshape((1,) + a.shape) for a in args]


        I, center_y, center_x, delta, sigma = \
            vectorize(I, np.array(center_y), np.array(center_x), np.array(delta), np.array(sigma))

        # import ipdb; ipdb.set_trace()

        W = do_read(I, center_y, center_x, delta, sigma)
        I2 = do_write(W, center_y, center_x, delta, sigma)


        def imagify(flat_image, h, w):
            image = flat_image.reshape([channels, h, w])
            image = image.transpose([1, 2, 0])
            return image / image.max()


        import pylab

        pylab.figure()
        pylab.gray()
        pylab.imshow(imagify(I, height, width), interpolation='nearest')

        pylab.figure()
        pylab.gray()
        pylab.imshow(imagify(W, N, N), interpolation='nearest')

        pylab.figure()
        pylab.gray()
        pylab.imshow(imagify(I2, height, width), interpolation='nearest')
        pylab.show(block=True)

        import ipdb;

        ipdb.set_trace()
    elif dim==3:
        N = 5
        channels = 1
        depth = 32
        height = 32
        width =  32

        #------------------------------------------------------------------------
        att = ZoomableAttentionWindow3d(channels, height, width, depth, N)

        I_ = T.matrix()
        center_y_ = T.vector()
        center_x_ = T.vector()
        center_z_ = T.vector()
        delta_ = T.vector()
        sigma_ = T.vector()
        W_ = att.read_large(I_, center_x_, center_y_, center_z_)

        do_read = theano.function(inputs=[I_, center_x_, center_y_, center_z_],
                                  outputs=W_)

        #------------------------------------------------------------------------
        # from fuel.datasets.hdf5 import H5PYDataset
        # train_set = H5PYDataset('../layer3D/shapenet10.hdf5', which_sets=('train',))
        # handle = train_set.open()
        # data = train_set.get_data(handle, slice(0, 1))
        # I = data[0].reshape(1,width*height*depth)
        # print((I.shape))

        center_x = [15]
        center_y = [15]
        center_z = [15]

        I = np.float32(np.ones((1,width*height*depth)))
        # I = np.float32(np.zeros((1, width, height, depth)))
        # I[0,15:16,:,:] = 1.
        # I = I.reshape((1,width*height*depth))
        W = do_read(I, center_x, center_x, center_z )
        # II = I.reshape((height, width, depth))
        # II = np.ones((height, width, depth))
        WW = W.reshape((height, width, depth))
        # WW = W.reshape((N, N, N))

        def viz2(V):
            V = V / np.max(V)

            x = y = z = t = []
            x1 = y1 = z1 = t1 = []
            x2 = y2 = z2 = t2 = []
            x3 = y3 = z3 = t3 = []
            for i in range(V.shape[0]):
                for j in range(V.shape[1]):
                    for k in range(V.shape[2]):
                        if V[i, j, k] != 0:
                            # V > 1e-1
                            x = x + [i]
                            y = y + [j]
                            z = z + [k]
                            t = t + [V[i, j, k]]
                            if i==15:
                                y1 = y1 + [j]
                                z1 = z1 + [k]
                                t1 = t1 + [V[i, j, k]]
                            if j==15:
                                x2 = x2 + [i]
                                z2 = z2 + [k]
                                t2 = t2 + [V[i, j, k]]
                            if k==15:
                                x3 = x3 + [i]
                                y3 = y3 + [j]
                                t3 = t3 + [V[i, j, k]]

            x = np.asarray(x)
            y = np.asarray(y)
            z = np.asarray(z)
            t = np.asarray(t)
            y1 = np.asarray(y1)
            z1 = np.asarray(z1)
            t1 = np.asarray(t1)
            x2 = np.asarray(x2)
            z2 = np.asarray(z2)
            t2 = np.asarray(t2)
            x3 = np.asarray(x3)
            y3 = np.asarray(y3)
            t3 = np.asarray(t3)

            fig = plt.figure()
            ax = fig.add_subplot(221, projection='3d')
            im = ax.scatter(x, y, z, c=t, marker='o', s=10)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            plt.xlim(0, V.shape[0])
            plt.ylim(0, V.shape[1])

            ax = fig.add_subplot(222)
            im = ax.scatter(y1, z1, c=t1, marker='o', s=30)
            ax.set_xlabel('Y Label')
            ax.set_ylabel('Z Label')
            plt.xlim(0, V.shape[0])
            plt.ylim(0, V.shape[1])

            ax = fig.add_subplot(223)
            im = ax.scatter(x2, z2, c=t2, marker='o', s=30)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Z Label')
            plt.xlim(0, V.shape[0])
            plt.ylim(0, V.shape[1])

            ax = fig.add_subplot(224)
            im = ax.scatter(x3, y3, c=t3, marker='o', s=30)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            plt.xlim(0, V.shape[0])
            plt.ylim(0, V.shape[1])

            cax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
            # cax = fig.add_axes([0.27, 0.8, 0.5, 0.05])
            fig.colorbar(im, cax=cax, orientation='vertical')

            plt.show()
            a = 1

        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        viz2(WW)
