import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear


def normalise_layer(tensor, scope=None, epsilon=1e-5):
    """
    Layer normalisation of a 2D tensor along its 2nd axis
    :param tensor: input tensor
    :param scope: scope of the tensor
    :param epsilon: avoid divide by zero
    :return: normalised layer
    """

    # Check if tensor is 2D
    assert len(tensor.get_shape()) == 2

    # Calculate mean and variance of tensor
    mean, variance = tf.nn.moments(tensor, axes=[1], keep_dims=True)

    # Ensure scope is a string
    if not isinstance(scope, str):
        scope = ''

    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))

    layer_norm_initial = (tensor - mean) / tf.sqrt(variance + epsilon)

    return layer_norm_initial * scale + shift


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


class MultiDimensionalLSTMCell(RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalisation
    state_is_tuple is always True
    """

    # Function when initialising object
    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    # Function when object is called
    def __call__(self, inputs, state, scope=None):
        """
        LSTM cell
        :param inputs: shape (batch, n)
        :param state: the states and hidden unit of the two cells
        :param scope: scope of the MDLSTM cell
        :return: hidden unit and state
        """

        with tf.variable_scope(scope or type(self).__name__):
            # c: hidden state
            # h: output
            c1, c2, h1, h2 = state

            # Returns: 2D tensor shape [batch, output_size]
            #       = sum_i(args[i] * W[i]), where W[i]s are newly created matrices
            # Change bias argument to False since LN will add bias via shift
            concat = _linear(args=[inputs, h1, h2], output_size=5 * self._num_units, bias=False)

            # TODO: what are these?
            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            # Add layer normalisation to each gate
            i = normalise_layer(tensor=i, scope='i/')
            j = normalise_layer(tensor=j, scope='j/')
            f1 = normalise_layer(tensor=f1, scope='f1/')
            f2 = normalise_layer(tensor=f2, scope='f2/')
            o = normalise_layer(tensor=o, scope='o/')

            # TODO: what is this?
            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) +
                     tf.nn.sigmoid(i) + self._activation(j))

            # Add layer norm in calculation of new hidden state
            new_h = self._activation(normalise_layer(new_c, scope='new_h/')) + tf.nn.sigmoid(o)
            new_state = LSTMStateTuple(c=new_c, h=new_h)

            return new_h, new_state


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


def get_up(t_, w_):
    """
    Function to get the sample skipping one row
    :param t_: time
    :param w_: width
    :return: time minus width
    """
    return t_ - tf.constant(w_)


def get_last(t_, w_):
    """
    Function to get the previous sample
    :param t_:
    :param w_:
    :return: time minus 1
    """
    return t_ - tf.constant(1)


def mdlstm_while_loop(rnn_size, input_data, window_shape, dims=None, scope_n='layer1'):
    """
    Implement naive MDLSTM
    :param rnn_size: hidden units
    :param input_data: data to process of shape [batch, h, w, channels]
    :param window_shape: shape of windows [height, width]
    :param dims: dimensions to reverse the input data, e.g.
        dims=[False, True, True, False] => True means reverse dimension
    :param scope_n: scope of this MDLSTM layer
    :return: output of LSTM of shape [batch, h/sh[0], w/sh[1], rnn_size]
        and inner states
    """

    with tf.variable_scope('MDLSTMCell-' + scope_n):
        # Create MDLSTM cell with selected size
        cell = MultiDimensionalLSTMCell(rnn_size)

        # Get shape of input (batch_size, x, y, channels)
        shape = input_data.get_shape().as_list()
        batch_size = shape[0]
        input_h = shape[1]
        input_w = shape[2]
        channels = shape[3]

        # Window size
        win_h = window_shape[0]
        win_w = window_shape[1]

        # Runtime batch size
        batch_size_runtime = tf.shape(input_data)[0]

        # If input cannot be exactly sampled by window, pad with zeros
        if input_h % win_h != 0:
            # Get offset size
            offset = tf.zeros(shape=[batch_size_runtime, input_h, win_w - (input_w % win_w), channels])

            # Concatenate Y dimension
            input_data = tf.concat(axis=2, values=[input_data, offset])

            # Get new shape
            shape = input_data.get_shape().as_list()

            # Update shape value
            input_w = shape[2]

        # Get the steps to perform in X and Y axis
        height, width = int(input_h / win_h), int(input_w / win_w)

        # Get the number of features (total number of input values per step)
        features = win_w * win_h * channels

        # Reshape input data to a tensor containing step indices and features inputs
        # Batch size is inferred from tensor size
        x = tf.reshape(input_data, shape=[batch_size_runtime, height, width, features])

        # Reverse selected dimensions
        if dims is not None:
            x = tf.reverse(x, dims=dims)

        # Reorder inputs to (height, width, batch_size, features)
        x = tf.transpose(x, perm=[1, 2, 0, 3])
        # Reshape to 1D tensor of shape (height * width * batch_size, features)
        x = tf.reshape(x, [-1, features])
        # Split tensor into height * width tensors of size (batch_size, features)
        x = tf.split(value=x, axis=0, num_or_size_splits=height * width)

        # Create input tensor array to use inside the loop
        inputs_ta = tf.TensorArray(dtype=tf.float32,
                                   size=height * width,
                                   name='input_ta')
        # Unstack input X in tensor array
        inputs_ta = inputs_ta.unstack(x)

        # Create input tensor array for the states
        states_ta = tf.TensorArray(dtype=tf.float32,
                                   size=height * width + 1,
                                   name='state_ta',
                                   clear_after_read=False)

        # Create tensor array for output
        outputs_ta = tf.TensorArray(dtype=tf.float32,
                                    size=height * width,
                                    name='output_ta')

        # Initial cell hidden states
        # Write to the end of array LSTMStateTuple filled with zeros
        states_ta = states_ta.write(index=height * width,
                                    value=LSTMStateTuple(
                                        c=tf.zeros(shape=[batch_size_runtime, rnn_size], dtype=tf.float32),
                                        h=tf.zeros(shape=[batch_size_runtime, rnn_size], dtype=tf.float32)
                                    ))

        # Initial index
        time = tf.constant(0)
        zero = tf.constant(0)

        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------
        # Body of while loop operation that applies on MDLSTM
        def loop_body(time_, outputs_ta_, states_ta_):
            # Current position <= width -> first row
            # and read the zero state added in row (height * width)
            # Else, get the sample located at a width distance
            state_up = tf.cond(pred=tf.less_equal(x=time_, y=tf.constant(width)),
                               true_fn=lambda: states_ta_.read(height * width),
                               false_fn=lambda: states_ta_.read(get_up(time_, width)))

            # If first step, read zero state
            # Else, read immediate last
            state_last = tf.cond(pred=tf.less(x=zero, y=tf.mod(time_, tf.constant(width))),
                                 true_fn=lambda: states_ta_.read(get_last(time_, width)),
                                 false_fn=lambda: states_ta_.read(height * width))

            # Build input state in both dimensions
            current_state = state_up[0], state_last[0], state_up[1], state_last[1]
            # Calculate output state and cell output
            out, state = cell(inputs=inputs_ta.read(time_), state=current_state)

            # Write output to tensor array
            outputs_ta_ = outputs_ta_.write(time_, out)
            # Save output state to tensor array
            states_ta_ = states_ta_.write(time_, state)

            # Return outputs and incremented time step
            return time_ + 1, outputs_ta_, states_ta_

        # Loop output condition. The index, given by time, should be less than
        # the total number of steps defined within the image
        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(x=time_, y=tf.constant(height * width))
        # ----------------------------------------------------------------------
        # ----------------------------------------------------------------------

        # Run the looped operation
        result, outputs_ta, states_ta = tf.while_loop(
            cond=condition,
            body=loop_body,
            loop_vars=[time, outputs_ta, states_ta],
            parallel_iterations=1
        )

        # Extract output tensors from the processed tensor array
        outputs = outputs_ta.stack()
        states = states_ta.stack()

        # Reshape outputs to match the shape of input
        y = tf.reshape(tensor=outputs,
                       shape=[height, width, batch_size_runtime, rnn_size])

        # Reorder the dimensions to match input
        y = tf.transpose(y, perm=[2, 0, 1, 3])

        # Reverse
        if dims is not None:
            y = tf.reverse(y, dims=dims)

        # Return output and inner states
        return y, states
