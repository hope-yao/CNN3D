
# from theano.tensor.nnet import conv2d, conv3d
from theano.sandbox.cuda.dnn import dnn_conv, dnn_conv3d

from theano.tensor.nnet.abstract_conv import (AbstractConv2d_gradInputs, AbstractConv3d_gradInputs, AbstractConv3d,
                                              get_conv_output_shape)
from theano.tensor.signal.pool import pool_2d, pool_3d, Pool

from blocks.bricks import (Initializable, Feedforward, Sequence, Activation,
                           LinearLike)
from blocks.bricks.base import application, Brick, lazy
from blocks.roles import add_role, FILTER, BIAS
from blocks.utils import shared_floatx_nans
from theano import tensor
import numpy

class Convolutional3(LinearLike):
    """Performs a 2D convolution.
    Parameters
    ----------
    filter_size : tuple
        The height and width of the filter (also called *kernels*).
    num_filters : int
        Number of filters per channel.
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer. The filters are pooled over the
        channels.
    batch_size : int, optional
        Number of examples per batch. If given, this will be passed to
        Theano convolution operator, possibly resulting in faster
        execution.
    image_size : tuple, optional
        The height and width of the input (image or feature map). If given,
        this will be passed to the Theano convolution operator, resulting
        in possibly faster execution times.
    step : tuple, optional
        The step (or stride) with which to slide the filters over the
        image. Defaults to (1, 1).
    border_mode : {'valid', 'full'}, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Defaults to 'valid'.
    tied_biases : bool
        Setting this to ``False`` will untie the biases, yielding a
        separate bias for every location at which the filter is applied.
        If ``True``, it indicates that the biases of every filter in this
        layer should be shared amongst all applications of that filter.
        Defaults to ``True``.
    """
    # Make it possible to override the implementation of conv2d that gets
    # used, i.e. to use theano.sandbox.cuda.dnn.dnn_conv directly in order
    # to leverage features not yet available in Theano's standard conv2d.
    # The function you override with here should accept at least the
    # input and the kernels as positionals, and the keyword arguments
    # input_shape, subsample, border_mode, and filter_shape. If some of
    # these are unsupported they should still be accepted and ignored,
    # e.g. with a wrapper function that swallows **kwargs.
    conv2d_impl = staticmethod(dnn_conv3d)

    # Used to override the output shape computation for a given value of
    # conv2d_impl. Should accept 4 positional arguments: the shape of an
    # image minibatch (with 4 elements: batch size, number of channels,
    # height, and width), the shape of the filter bank (number of filters,
    # number of output channels, filter height, filter width), the border
    # mode, and the step (vertical and horizontal strides). It is expected
    # to return a 4-tuple of (batch size, number of channels, output
    # height, output width). The first element of this tuple is not used
    # for anything by this brick.
    get_output_shape = staticmethod(get_conv_output_shape)
    # def get_conv_output_shape(self, image_shape, kernel_shape,
    #                           border_mode, subsample,
    #                           filter_dilation=None):
    #     """
    #     This function compute the output shape of convolution operation.
    #
    #     Parameters
    #     ----------
    #     image_shape: tuple of int (symbolic or numeric) corresponding to the input
    #         image shape. Its four (or five) element must correspond respectively
    #         to: batch size, number of input channels, height and width (and
    #         possibly depth) of the image. None where undefined.
    #     kernel_shape: tuple of int (symbolic or numeric) corresponding to the
    #         kernel shape. Its four (or five) elements must correspond respectively
    #         to: number of output channels, number of input channels, height and
    #         width (and possibly depth) of the kernel. None where undefined.
    #     border_mode: string, int (symbolic or numeric) or tuple of int (symbolic
    #         or numeric). If it is a string, it must be 'valid', 'half' or 'full'.
    #         If it is a tuple, its two (or three) elements respectively correspond
    #         to the padding on height and width (and possibly depth) axis.
    #     subsample: tuple of int (symbolic or numeric). Its or three elements
    #         espectively correspond to the subsampling on height and width (and
    #         possibly depth) axis.
    #     filter_dilation: tuple of int (symbolic or numeric). Its two elements
    #         correspond respectively to the dilation on height and width axis.
    #
    #     Returns
    #     -------
    #     output_shape: tuple of int corresponding to the output image shape. Its
    #         four element must correspond respectively to: batch size, number of
    #         output channels, height and width of the image. None where undefined.
    #
    #     """
    #     bsize, imshp = image_shape[0], image_shape[2:]
    #     nkern, kshp = kernel_shape[0], kernel_shape[2:]
    #
    #     if filter_dilation is None:
    #         filter_dilation = numpy.ones(len(subsample), dtype='int')
    #
    #     if isinstance(border_mode, tuple):
    #         out_shp = tuple(self.get_conv_shape_1axis(
    #             imshp[i], kshp[i], border_mode[i],
    #             subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    #     else:
    #         out_shp = tuple(self.get_conv_shape_1axis(
    #             imshp[i], kshp[i], border_mode,
    #             subsample[i], filter_dilation[i]) for i in range(len(subsample)))
    #     return (bsize, nkern) + out_shp
    #
    # # filter dilation set by default to 1
    # # for compatibility with other tests.
    # def get_conv_shape_1axis(self, image_shape, kernel_shape, border_mode,
    #                          subsample, dilation=1):
    #     """
    #     This function compute the output shape of convolution operation.
    #
    #     Parameters
    #     ----------
    #     image_shape: int or None. Corresponds to the input image shape on a
    #         given axis. None if undefined.
    #     kernel_shape: int or None. Corresponds to the kernel shape on a given
    #         axis. None if undefined.
    #     border_mode: string or int. If it is a string, it must be
    #         'valid', 'half' or 'full'. If it is an integer, it must correspond to
    #         the padding on the considered axis.
    #     subsample: int. It must correspond to the subsampling on the
    #         considered axis.
    #     dilation: int. It must correspond to the dilation on the
    #         considered axis.
    #
    #     Returns
    #     -------
    #     out_shp: int corresponding to the output image shape on the
    #         considered axis. None if undefined.
    #
    #     """
    #     if None in [image_shape, kernel_shape, border_mode,
    #                 subsample, dilation]:
    #         return None
    #     # Implicit dilated kernel shape
    #     dil_kernel_shape = (kernel_shape - 1) * dilation + 1
    #     if border_mode == "half":
    #         pad = dil_kernel_shape // 2
    #     elif border_mode == "full":
    #         pad = dil_kernel_shape - 1
    #     elif border_mode == "valid":
    #         pad = 0
    #     else:
    #         pad = border_mode
    #         if pad < 0:
    #             raise ValueError("border_mode must be >= 0")
    #
    #     # In case of symbolic shape, we want to build the smallest graph
    #     # (image_shape + 2 * pad - dil_kernel_shape) // subsample + 1
    #     if pad == 0:
    #         out_shp = (image_shape - dil_kernel_shape)
    #     else:
    #         out_shp = (image_shape + 2 * pad - dil_kernel_shape)
    #     if subsample != 1:
    #         out_shp = out_shp // subsample
    #     out_shp = out_shp + 1
    #
    #     return out_shp

    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels, batch_size=None,
                 image_size=(None, None, None), step=(1, 1, 1), border_mode='valid',
                 tied_biases=True, **kwargs):
        super(Convolutional3, self).__init__(**kwargs)

        self.filter_size = filter_size
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.step = step
        self.border_mode = border_mode
        self.tied_biases = tied_biases

    def _allocate(self):
        W = shared_floatx_nans((self.num_filters, self.num_channels) +
                               self.filter_size, name='W')
        add_role(W, FILTER)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if getattr(self, 'use_bias', True):
            if self.tied_biases:
                b = shared_floatx_nans((self.num_filters,), name='b')
            else:
                # this error is raised here instead of during initializiation
                # because ConvolutionalSequence may specify the image size
                if self.image_size == (None, None) and not self.tied_biases:
                    raise ValueError('Cannot infer bias size without '
                                     'image_size specified. If you use '
                                     'variable image_size, you should use '
                                     'tied_biases=True.')

                b = shared_floatx_nans(self.get_dim('output'), name='b')
            add_role(b, BIAS)

            self.parameters.append(b)
            self.add_auxiliary_variable(b.norm(2), name='b_norm')

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Perform the convolution.
        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            A 4D tensor with the axes representing batch size, number of
            channels, image height, and image width.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A 4D tensor of filtered images (feature maps) with dimensions
            representing batch size, number of filters, feature map height,
            and feature map width.
            The height and width of the feature map depend on the border
            mode. For 'valid' it is ``image_size - filter_size + 1`` while
            for 'full' it is ``image_size + filter_size - 1``.
        """
        if self.image_size == (None, None):
            input_shape = None
        else:
            input_shape = (self.batch_size, self.num_channels)
            input_shape += self.image_size

        output = self.conv2d_impl(
            input_, self.W,
            # input_shape=input_shape,
            subsample=self.step,
            border_mode=self.border_mode
            # filter_shape=((self.num_filters, self.num_channels) +
            #               self.filter_size)
        )
        if getattr(self, 'use_bias', True):
            if self.tied_biases:
                output += self.b.dimshuffle('x', 0, 'x', 'x', 'x')
            else:
                output += self.b.dimshuffle('x', 0, 1, 2, 3)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return (self.num_channels,) + self.image_size
        if name == 'output':
            input_shape = (None, self.num_channels) + self.image_size
            kernel_shape = ((self.num_filters, self.num_channels) +
                            self.filter_size)
            out_shape = self.get_output_shape(input_shape, kernel_shape,
                                              self.border_mode, self.step)
            assert len(out_shape) == 5
            return out_shape[1:]
        return super(Convolutional3, self).get_dim(name)

    @property
    def num_output_channels(self):
        return self.num_filters


class ConvolutionalTranspose3(Convolutional3):
    """Performs the transpose of a 2D convolution.
    Parameters
    ----------
    num_filters : int
        Number of filters at the *output* of the transposed convolution,
        i.e. the number of channels in the corresponding convolution.
    num_channels : int
        Number of channels at the *input* of the transposed convolution,
        i.e. the number of output filters in the corresponding
        convolution.
    step : tuple, optional
        The step (or stride) of the corresponding *convolution*.
        Defaults to (1, 1).
    image_size : tuple, optional
        Image size of the input to the *transposed* convolution, i.e.
        the output of the corresponding convolution. Required for tied
        biases. Defaults to ``None``.
    unused_edge : tuple, optional
        Tuple of pixels added to the inferred height and width of the
        output image, whose values would be ignored in the corresponding
        forward convolution. Must be such that 0 <= ``unused_edge[i]`` <=
        ``step[i]``. Note that this parameter is **ignored** if
        ``original_image_size`` is specified in the constructor or manually
        set as an attribute.
    original_image_size : tuple, optional
        The height and width of the image that forms the output of
        the transpose operation, which is the input of the original
        (non-transposed) convolution. By default, this is inferred
        from `image_size` to be the size that has each pixel of the
        original image touched by at least one filter application
        in the original convolution. Degenerate cases with dropped
        border pixels (in the original convolution) are possible, and can
        be manually specified via this argument. See notes below.
    See Also
    --------
    :class:`Convolutional` : For the documentation of other parameters.
    Notes
    -----
    By default, `original_image_size` is inferred from `image_size`
    as being the *minimum* size of image that could have produced this
    output. Let ``hanging[i] = original_image_size[i] - image_size[i]
    * step[i]``. Any value of ``hanging[i]`` greater than
    ``filter_size[i] - step[i]`` will result in border pixels that are
    ignored by the original convolution. With this brick, any
    ``original_image_size`` such that ``filter_size[i] - step[i] <
    hanging[i] < filter_size[i]`` for all ``i`` can be validly specified.
    However, no value will be output by the transposed convolution
    itself for these extra hanging border pixels, and they will be
    determined entirely by the bias.
    """
    @lazy(allocation=['filter_size', 'num_filters', 'num_channels'])
    def __init__(self, filter_size, num_filters, num_channels,
                 original_image_size=None, unused_edge=(0, 0, 0),
                 **kwargs):
        super(ConvolutionalTranspose3, self).__init__(
            filter_size, num_filters, num_channels, **kwargs)
        self.original_image_size = original_image_size
        self.unused_edge = unused_edge

    @property
    def original_image_size(self):
        if self._original_image_size is None:
            if all(s is None for s in self.image_size):
                raise ValueError("can't infer original_image_size, "
                                 "no image_size set")
            if isinstance(self.border_mode, tuple):
                border = self.border_mode
            elif self.border_mode == 'full':
                border = tuple(k - 1 for k in self.filter_size)
            elif self.border_mode == 'half':
                border = tuple(k // 2 for k in self.filter_size)
            else:
                border = [0] * len(self.image_size)
            tups = zip(self.image_size, self.step, self.filter_size, border,
                       self.unused_edge)
            return tuple(s * (i - 1) + k - 2 * p + u for i, s, k, p, u in tups)
        else:
            return self._original_image_size

    @original_image_size.setter
    def original_image_size(self, value):
        self._original_image_size = value

    def conv2d_impl(self, input_, W, input_shape, subsample, border_mode,
                    filter_shape):
        # The AbstractConv2d_gradInputs op takes a kernel that was used for the
        # **convolution**. We therefore have to invert num_channels and
        # num_filters for W.
        # W = W.transpose(1, 0, 2, 3)
        W = W.transpose(1, 0, 2, 3, 4)
        imshp = (None,) + self.get_dim('output')
        kshp = (filter_shape[1], filter_shape[0]) + filter_shape[2:]
        return AbstractConv3d_gradInputs(
            imshp=imshp, kshp=kshp, border_mode=border_mode,
            subsample=subsample)(W, input_, self.get_dim('output')[1:])

    def get_dim(self, name):
        if name == 'output':
            return (self.num_filters,) + self.original_image_size
        return super(ConvolutionalTranspose3, self).get_dim(name)


class Pooling3(Initializable, Feedforward):
    """Base Brick for pooling operations.
    This should generally not be instantiated directly; see
    :class:`MaxPooling`.
    """
    @lazy(allocation=['mode', 'pooling_size'])
    def __init__(self, mode, pooling_size, step, input_dim, ignore_border,
                 padding, **kwargs):
        super(Pooling3, self).__init__(**kwargs)
        self.pooling_size = pooling_size
        self.mode = mode
        self.step = step
        self.input_dim = input_dim if input_dim is not None else (None,) * 4
        self.ignore_border = ignore_border
        self.padding = padding

    @property
    def image_size(self):
        return self.input_dim[-3:]

    @image_size.setter
    def image_size(self, value):
        self.input_dim = self.input_dim[:-3] + value

    @property
    def num_channels(self):
        return self.input_dim[0]

    @num_channels.setter
    def num_channels(self, value):
        self.input_dim = (value,) + self.input_dim[1:]

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        """Apply the pooling (subsampling) transformation.
        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            An tensor with dimension greater or equal to 2. The last two
            dimensions will be downsampled. For example, with images this
            means that the last two dimensions should represent the height
            and width of your image.
        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            A tensor with the same number of dimensions as `input_`, but
            with the last two dimensions downsampled.
        """
        output = pool_3d(input_, self.pooling_size, st=self.step,
                         mode=self.mode, padding=self.padding,
                         ignore_border=self.ignore_border)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'output':
            return tuple(Pool.out_shape(
                self.input_dim, self.pooling_size, st=self.step, ndim=3,
                ignore_border=self.ignore_border, padding=self.padding))

    @property
    def num_output_channels(self):
        return self.input_dim[0]


class MaxPooling3(Pooling3):
    """Max pooling layer.
    Parameters
    ----------
    pooling_size : tuple
        The height and width of the pooling region i.e. this is the factor
        by which your input's last two dimensions will be downscaled.
    step : tuple, optional
        The vertical and horizontal shift (stride) between pooling regions.
        By default this is equal to `pooling_size`. Setting this to a lower
        number results in overlapping pooling regions.
    input_dim : tuple, optional
        A tuple of integers representing the shape of the input. The last
        two dimensions will be used to calculate the output dimension.
    padding : tuple, optional
        A tuple of integers representing the vertical and horizontal
        zero-padding to be applied to each of the top and bottom
        (vertical) and left and right (horizontal) edges. For example,
        an argument of (4, 3) will apply 4 pixels of padding to the
        top edge, 4 pixels of padding to the bottom edge, and 3 pixels
        each for the left and right edge. By default, no padding is
        performed.
    ignore_border : bool, optional
        Whether or not to do partial downsampling based on borders where
        the extent of the pooling region reaches beyond the edge of the
        image. If `True`, a (5, 5) image with (2, 2) pooling regions
        and (2, 2) step will be downsampled to shape (2, 2), otherwise
        it will be downsampled to (3, 3). `True` by default.
    Notes
    -----
    .. warning::
        As of this writing, setting `ignore_border` to `False` with a step
        not equal to the pooling size will force Theano to perform pooling
        computations on CPU rather than GPU, even if you have specified
        a GPU as your computation device. Additionally, Theano will only
        use [cuDNN]_ (if available) for pooling computations with
        `ignure_border` set to `True`. You can ensure that the entire
        input is captured by at least one pool by using the `padding`
        argument to add zero padding prior to pooling being performed.
    .. [cuDNN] `NVIDIA cuDNN <https://developer.nvidia.com/cudnn>`_.
    """
    @lazy(allocation=['pooling_size'])
    def __init__(self, pooling_size, step=None, input_dim=None,
                 ignore_border=True, padding=(0, 0, 0),
                 **kwargs):
        super(MaxPooling3, self).__init__('max', pooling_size,
                                         step=step, input_dim=input_dim,
                                         ignore_border=ignore_border,
                                         padding=padding, **kwargs)

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Fix objects created before pull request #899.
        self.mode = getattr(self, 'mode', 'max')
        self.padding = getattr(self, 'padding', (0, 0, 0))
        self.ignore_border = getattr(self, 'ignore_border', False)


class AveragePooling3(Pooling3):
    """Average pooling layer.
    Parameters
    ----------
    include_padding : bool, optional
        When calculating an average, include zeros that are the
        result of zero padding added by the `padding` argument.
        A value of `True` is only accepted if `ignore_border`
        is also `True`. `False` by default.
    Notes
    -----
    For documentation on the remainder of the arguments to this
    class, see :class:`MaxPooling`.
    """
    @lazy(allocation=['pooling_size'])
    def __init__(self, pooling_size, step=None, input_dim=None,
                 ignore_border=True, padding=(0, 0, 0),
                 include_padding=False, **kwargs):
        mode = 'average_inc_pad' if include_padding else 'average_exc_pad'
        super(AveragePooling3, self).__init__(mode, pooling_size,
                                             step=step, input_dim=input_dim,
                                             ignore_border=ignore_border,
                                             padding=padding, **kwargs)


class ConvolutionalSequence3(Sequence, Initializable, Feedforward):
    """A sequence of convolutional (or pooling) operations.
    Parameters
    ----------
    layers : list
        List of convolutional bricks (i.e. :class:`Convolutional`,
        :class:`ConvolutionalActivation`, or :class:`Pooling` bricks).
        :class:`Activation` bricks that operate elementwise can also
        be included.
    num_channels : int
        Number of input channels in the image. For the first layer this is
        normally 1 for grayscale images and 3 for color (RGB) images. For
        subsequent layers this is equal to the number of filters output by
        the previous convolutional layer.
    batch_size : int, optional
        Number of images in batch. If given, will be passed to
        theano's convolution operator resulting in possibly faster
        execution.
    image_size : tuple, optional
        Width and height of the input (image/featuremap). If given,
        will be passed to theano's convolution operator resulting in
        possibly faster execution.
    border_mode : 'valid', 'full' or None, optional
        The border mode to use, see :func:`scipy.signal.convolve2d` for
        details. Unlike with :class:`Convolutional`, this defaults to
        None, in which case no default value is pushed down to child
        bricks at allocation time. Child bricks will in this case
        need to rely on either a default border mode (usually valid)
        or one provided at construction and/or after construction
        (but before allocation).
    tied_biases : bool, optional
        Same meaning as in :class:`Convolutional`. Defaults to ``None``,
        in which case no value is pushed to child :class:`Convolutional`
        bricks.
    Notes
    -----
    The passed convolutional operators should be 'lazy' constructed, that
    is, without specifying the batch_size, num_channels and image_size. The
    main feature of :class:`ConvolutionalSequence` is that it will set the
    input dimensions of a layer to the output dimensions of the previous
    layer by the :meth:`~bricks.Brick.push_allocation_config` method.
    The push behaviour of `tied_biases` mirrors that of `use_bias` or any
    initialization configuration: only an explicitly specified value is
    pushed down the hierarchy. `border_mode` also has this behaviour.
    The reason the `border_mode` parameter behaves the way it does is that
    pushing a single default `border_mode` makes it very difficult to
    have child bricks with different border modes. Normally, such things
    would be overridden after `push_allocation_config()`, but this is
    a particular hassle as the border mode affects the allocation
    parameters of every subsequent child brick in the sequence. Thus, only
    an explicitly specified border mode will be pushed down the hierarchy.
    """
    @lazy(allocation=['num_channels'])
    def __init__(self, layers, num_channels, batch_size=None,
                 image_size=(None, None, None), border_mode=None, tied_biases=None,
                 **kwargs):
        self.layers = layers
        self.image_size = image_size
        self.num_channels = num_channels
        self.batch_size = batch_size
        self.border_mode = border_mode
        self.tied_biases = tied_biases

        application_methods = [brick.apply for brick in layers]
        super(ConvolutionalSequence3, self).__init__(
            application_methods=application_methods, **kwargs)

    def get_dim(self, name):
        if name == 'input_':
            return ((self.num_channels,) + self.image_size)
        if name == 'output':
            last = len(self.layers) - 1
            while last >= 0:
                try:
                    return self.layers[last].get_dim(name)
                except ValueError:
                    last -= 1
            # The output shape of an empty ConvolutionalSequence or one
            # consisting only of Activations is the input shape.
            return self.get_dim('input_')
        return super(ConvolutionalSequence3, self).get_dim(name)

    def _push_allocation_config(self):
        num_channels = self.num_channels
        image_size = self.image_size
        for layer in self.layers:
            if isinstance(layer, Activation):
                # Activations operate elementwise; nothing to set.
                layer.push_allocation_config()
                continue
            if self.border_mode is not None:
                layer.border_mode = self.border_mode
            if self.tied_biases is not None:
                layer.tied_biases = self.tied_biases
            layer.image_size = image_size
            layer.num_channels = num_channels
            layer.batch_size = self.batch_size
            if getattr(self, 'use_bias', None) is not None:
                layer.use_bias = self.use_bias

            # Push input dimensions to children
            layer.push_allocation_config()

            # Retrieve output dimensions
            # and set it for next layer
            if None not in layer.image_size:
                output_shape = layer.get_dim('output')
                image_size = output_shape[1:]
            num_channels = layer.num_output_channels


class Flattener3(Brick):
    """Flattens the input.
    It may be used to pass multidimensional objects like images or feature
    maps of convolutional bricks into bricks which allow only two
    dimensional input (batch, features) like MLP.
    """
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return input_.flatten(ndim=2)


#
# convnet = Convolutional3(
#                     filter_size=(3,3,3),
#                     num_filters=2,
#                     num_channels=3,
#                     step=(1,1,1),
#                     border_mode='valid',
#                     weights_init = Constant(1),
#                     biases_init=Constant(0),
#                     name='conv3_0')
# convnet.initialize()
#
# dtensor5 = T.TensorType(floatX, (False,) * 5)
# x = dtensor5(name='input')
#
# h = convnet.apply(x)
# f = theano.function([x], h)
#
# a = np.zeros((2,3,5,5,5), dtype=floatX)
# a[0,0,:,:,:] = np.ones((5,5,5), dtype=floatX)
# a[1,1,:,:,:] = 3*np.ones((5,5,5), dtype=floatX)
# c = f(a)
# print(c.shape)
# print(c)

