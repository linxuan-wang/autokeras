from abc import abstractmethod

from autokeras.constant import Constant
from autokeras.nn.graph import Graph
from autokeras.nn.layers import StubAdd, StubDense, StubReLU, get_conv_class, get_dropout_class, \
    get_global_avg_pooling_class, get_pooling_class, get_avg_pooling_class, get_batch_norm_class, StubDropout1d, \
    StubConcatenate


class NetworkGenerator:
    """The base class for generating a network.

    It can be used to generate a CNN or Multi-Layer Perceptron.

    Attributes:
        n_output_node: Number of output nodes in the network.
        input_shape: A tuple to represent the input shape.
    """

    def __init__(self, n_output_node, input_shape):
        """Initialize the instance.

        Sets the parameters `n_output_node` and `input_shape` for the instance.

        Args:
            n_output_node: An integer. Number of output nodes in the network.
            input_shape: A tuple. Input shape of the network.
        """
        self.n_output_node = n_output_node
        self.input_shape = input_shape

    @abstractmethod
    def generate(self, model_len, model_width):
        pass


class CnnGenerator(NetworkGenerator):
    """A class to generate CNN.

    Attributes:
          n_dim: `len(self.input_shape) - 1`
          conv: A class that represents `(n_dim-1)` dimensional convolution.
          dropout: A class that represents `(n_dim-1)` dimensional dropout.
          global_avg_pooling: A class that represents `(n_dim-1)` dimensional Global Average Pooling.
          pooling: A class that represents `(n_dim-1)` dimensional pooling.
          batch_norm: A class that represents `(n_dim-1)` dimensional batch normalization.
    """

    def __init__(self, n_output_node, input_shape):
        """Initialize the instance.

        Args:
            n_output_node: An integer. Number of output nodes in the network.
            input_shape: A tuple. Input shape of the network.
        """
        super(CnnGenerator, self).__init__(n_output_node, input_shape)
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        if len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.pooling = get_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):
        """Generates a CNN.

        Args:
            model_len: An integer. Number of convolutional layers.
            model_width: An integer. Number of filters for the convolutional layers.

        Returns:
            An instance of the class Graph. Represents the neural architecture graph of the generated model.
        """
        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        pooling_len = int(model_len / 4)
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        stride = 1
        for i in range(model_len):
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            output_node_id = graph.add_layer(self.batch_norm(graph.node_list[output_node_id].shape[-1]), output_node_id)
            output_node_id = graph.add_layer(self.conv(temp_input_channel,
                                                       model_width,
                                                       kernel_size=3,
                                                       stride=stride), output_node_id)
            # if stride == 1:
            #     stride = 2
            temp_input_channel = model_width
            if pooling_len == 0 or ((i + 1) % pooling_len == 0 and i != model_len - 1):
                output_node_id = graph.add_layer(self.pooling(), output_node_id)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        output_node_id = graph.add_layer(self.dropout(Constant.CONV_DROPOUT_RATE), output_node_id)
        output_node_id = graph.add_layer(StubDense(graph.node_list[output_node_id].shape[0], model_width),
                                         output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        graph.add_layer(StubDense(model_width, self.n_output_node), output_node_id)
        return graph


class MlpGenerator(NetworkGenerator):
    """A class to generate Multi-Layer Perceptron.
    """

    def __init__(self, n_output_node, input_shape):
        """Initialize the instance.

        Args:
            n_output_node: An integer. Number of output nodes in the network.
            input_shape: A tuple. Input shape of the network. If it is 1D, ensure the value is appended by a comma
                in the tuple.
        """
        super(MlpGenerator, self).__init__(n_output_node, input_shape)
        if len(self.input_shape) > 1:
            raise ValueError('The input dimension is too high.')

    def generate(self, model_len=None, model_width=None):
        """Generates a Multi-Layer Perceptron.

        Args:
            model_len: An integer. Number of hidden layers.
            model_width: An integer or a list of integers of length `model_len`. If it is a list, it represents the
                number of nodes in each hidden layer. If it is an integer, all hidden layers have nodes equal to this
                value.

        Returns:
            An instance of the class Graph. Represents the neural architecture graph of the generated model.
        """
        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        if isinstance(model_width, list) and not len(model_width) == model_len:
            raise ValueError('The length of \'model_width\' does not match \'model_len\'')
        elif isinstance(model_width, int):
            model_width = [model_width] * model_len

        graph = Graph(self.input_shape, False)
        output_node_id = 0
        n_nodes_prev_layer = self.input_shape[0]
        for width in model_width:
            output_node_id = graph.add_layer(StubDense(n_nodes_prev_layer, width), output_node_id)
            output_node_id = graph.add_layer(StubDropout1d(Constant.MLP_DROPOUT_RATE), output_node_id)
            output_node_id = graph.add_layer(StubReLU(), output_node_id)
            n_nodes_prev_layer = width

        graph.add_layer(StubDense(n_nodes_prev_layer, self.n_output_node), output_node_id)
        return graph


class ResNetGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super(ResNetGenerator, self).__init__(n_output_node, input_shape)
        # self.layers = [2, 2, 2, 2]
        self.in_planes = 64
        self.block_expansion = 1
        self.n_dim = len(self.input_shape) - 1
        if len(self.input_shape) > 4:
            raise ValueError('The input dimension is too high.')
        elif len(self.input_shape) < 2:
            raise ValueError('The input dimension is too low.')
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.adaptive_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        output_node_id = 0
        # output_node_id = graph.add_layer(StubReLU(), output_node_id)
        output_node_id = graph.add_layer(self.conv(temp_input_channel, model_width, kernel_size=3), output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(model_width), output_node_id)
        # output_node_id = graph.add_layer(self.pooling(kernel_size=3, stride=2, padding=1), output_node_id)

        output_node_id = self._make_layer(graph, model_width, 2, output_node_id, 1)
        model_width *= 2
        output_node_id = self._make_layer(graph, model_width, 2, output_node_id, 2)
        model_width *= 2
        output_node_id = self._make_layer(graph, model_width, 2, output_node_id, 2)
        model_width *= 2
        output_node_id = self._make_layer(graph, model_width, 2, output_node_id, 2)

        output_node_id = graph.add_layer(self.global_avg_pooling(), output_node_id)
        graph.add_layer(StubDense(model_width * self.block_expansion, self.n_output_node), output_node_id)
        return graph

    def _make_layer(self, graph, planes, blocks, node_id, stride):
        strides = [stride] + [1] * (blocks - 1)
        out = node_id
        for current_stride in strides:
            out = self._make_block(graph, self.in_planes, planes, out, current_stride)
            self.in_planes = planes * self.block_expansion
        return out

    def _make_block(self, graph, in_planes, planes, node_id, stride=1):
        out = graph.add_layer(self.batch_norm(in_planes), node_id)
        out = graph.add_layer(StubReLU(), out)
        residual_node_id = out
        out = graph.add_layer(self.conv(in_planes, planes, kernel_size=3, stride=stride), out)
        out = graph.add_layer(self.batch_norm(planes), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(planes, planes, kernel_size=3), out)

        residual_node_id = graph.add_layer(StubReLU(), residual_node_id)
        residual_node_id = graph.add_layer(self.conv(in_planes,
                                                     planes * self.block_expansion,
                                                     kernel_size=1,
                                                     stride=stride), residual_node_id)
        out = graph.add_layer(StubAdd(), (out, residual_node_id))
        return out


class DenseNetGenerator(NetworkGenerator):
    def __init__(self, n_output_node, input_shape):
        super().__init__(n_output_node, input_shape)
        # DenseNet Constant
        self.num_init_features = 64
        self.growth_rate = 32
        self.block_config = (6, 12, 24, 16)
        self.bn_size = 4
        self.drop_rate = 0
        # Stub layers
        self.n_dim = len(self.input_shape) - 1
        self.conv = get_conv_class(self.n_dim)
        self.dropout = get_dropout_class(self.n_dim)
        self.global_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.adaptive_avg_pooling = get_global_avg_pooling_class(self.n_dim)
        self.max_pooling = get_pooling_class(self.n_dim)
        self.avg_pooling = get_avg_pooling_class(self.n_dim)
        self.batch_norm = get_batch_norm_class(self.n_dim)

    def generate(self, model_len=None, model_width=None):
        if model_len is None:
            model_len = Constant.MODEL_LEN
        if model_width is None:
            model_width = Constant.MODEL_WIDTH
        graph = Graph(self.input_shape, False)
        temp_input_channel = self.input_shape[-1]
        # First convolution
        output_node_id = 0
        output_node_id = graph.add_layer(self.conv(temp_input_channel, model_width, kernel_size=7),
                                         output_node_id)
        output_node_id = graph.add_layer(self.batch_norm(num_features=self.num_init_features), output_node_id)
        output_node_id = graph.add_layer(StubReLU(), output_node_id)
        db_input_node_id = graph.add_layer(self.max_pooling(kernel_size=3, stride=2, padding=1), output_node_id)
        # Each DensebLock
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            db_input_node_id = self._dense_block(num_layers=num_layers, num_input_features=num_features,
                                                 bn_size=self.bn_size, growth_rate=self.growth_rate,
                                                 drop_rate=self.drop_rate,
                                                 graph=graph, input_node_id=db_input_node_id)
            num_features = num_features + num_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                db_input_node_id = self._transition(num_input_features=num_features,
                                                    num_output_features=num_features // 2,
                                                    graph=graph, input_node_id=db_input_node_id)
                num_features = num_features // 2
        # Final batch norm
        out = graph.add_layer(self.batch_norm(num_features), db_input_node_id)

        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.adaptive_avg_pooling(), out)
        # Linear layer
        graph.add_layer(StubDense(num_features, self.n_output_node), out)
        return graph

    def _dense_block(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, graph, input_node_id):
        block_input_node = input_node_id
        for i in range(num_layers):
            block_input_node = self._dense_layer(num_input_features + i * growth_rate, growth_rate,
                                                 bn_size, drop_rate,
                                                 graph, block_input_node)
        return block_input_node

    def _dense_layer(self, num_input_features, growth_rate, bn_size, drop_rate, graph, input_node_id):
        out = graph.add_layer(self.batch_norm(num_features=num_input_features), input_node_id)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1), out)
        out = graph.add_layer(self.batch_norm(bn_size * growth_rate), out)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1), out)
        out = graph.add_layer(self.dropout(rate=drop_rate), out)
        out = graph.add_layer(StubConcatenate(), (input_node_id, out))
        return out

    def _transition(self, num_input_features, num_output_features, graph, input_node_id):
        out = graph.add_layer(self.batch_norm(num_features=num_input_features), input_node_id)
        out = graph.add_layer(StubReLU(), out)
        out = graph.add_layer(self.conv(num_input_features, num_output_features, kernel_size=1, stride=1), out)
        out = graph.add_layer(self.avg_pooling(kernel_size=2, stride=2), out)
        return out





class AlexNetGenerator(NetworkGenerator):
        #Implementation of the AlexNet.
   
    def __init__(self, x, num_classes, weights_path='DEFAULT'):\
        #\"\"\"Create the graph of the AlexNet model.\n",
        """Args:
            x: Placeholder for the input tensor
            keep_prob: Dropout probability
            num_classes: Number of classes in the dataset
            skip_layer: List of names of the layer, that get trained from
                scratch
            weights_path: Complete path to the pretrained weight file, if it
                isn't in the same folder as this code\n",
        """
        # Parse input arguments into class variables\n",
        self.X = x
        self.NUM_CLASSES = num_classes
        
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path
    
        # Call the create function to build the computational graph of AlexNet
        self.create()

    def generate(self):
            #Create the network graph.
            # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
            conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='SAME', name='conv1')
            norm1 = lrn(conv1, 2, 1e-05, 0.75, name='norm1')
            pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')
    
            # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups\n",
            conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
            norm2 = lrn(conv2, 2, 1e-05, 0.75, name='norm2')
            pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')
            
            # 3rd Layer: Conv (w ReLu)\n",
            conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')
    
            # 4th Layer: Conv (w ReLu) splitted into two groups\n",
            conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')
    
            # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups\n",
            conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2,name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
            
            # 6th Layer: Flatten -> FC (w ReLu) -> Dropout\n",
            flattened = tf.reshape(pool5, [-1, 6*6*256])
            fc6 = fc(flattened, 6*6*256, 4096, name='fc6')
    
            # 7th Layer: FC (w ReLu) -> Dropout\n",
            fc7 = fc(fc6, 4096, 4096, name='fc7')
    
            # 8th Layer: FC and return unscaled activations\n",
            self.fc8 = fc(fc7, 4096, self.NUM_CLASSES, relu=False, name='fc8')
    
        # def load_initial_weights(self, session):
        #     """Load weights from file into network.
        #     As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        #     come as a dict of lists (e.g. weights['conv1'] is a list) and not as
        #     dict of dicts (e.g. weights['conv1'] is a dict with keys 'weights' &
        #     'biases') we need a special load function
        #     """
        #     # Load the weights into memory\n",
        #     weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()
    
        #     # list of all assignment operators\n",
        #     assign_list = []
    
        #     # Loop over all layer names stored in the weights dict\n",
        #     for op_name in weights_dict:
    
        #         # Check if layer should be trained from scratch\n",
        #         with tf.variable_scope(op_name, reuse=True):

        #                 # Assign weights/biases to their corresponding tf variable\n",
        #             for data in weights_dict[op_name]:
    
        #                     # Biases
        #                 if len(data.shape) == 1:
        #                     var = tf.get_variable('biases', trainable=False)
        #                     assign_list.append(var.assign(data))

        #                     # Weights\n",
        #                 else:
        #                     var = tf.get_variable('weights', trainable=False)
        #                     assign_list.append(var.assign(data))
    
        #     # create a group operator for all assignments\n",
        #     ret = tf.group(assign_list, name="load_weights")
        #     return ret
    def fc(x, num_in, num_out, name, relu=True):
        """Create a fully connected layer."""
        with tf.variable_scope(name) as scope:
        
            # Create tf variables for the weights and biases\n",
            weights = tf.get_variable('weights', shape=[num_in, num_out],
                                          trainable=True)
            biases = tf.get_variable('biases', [num_out], trainable=True)
        
            # Matrix multiply weights and inputs and add bias\n",
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)
        
        if relu:
            # Apply ReLu non linearity\n",
            relu = tf.nn.relu(act)
            return relu
        else:
            return ac

    def max_pool(x, filter_height, filter_width, stride_y, stride_x, name,
                 padding='SAME'):
        """Create a max pooling layer."""
        return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)
    
    
    def lrn(x, radius, alpha, beta, name, bias=1.0):
        """Create a local response normalization layer."""
        return tf.nn.local_response_normalization(x, depth_radius=radius,
                                                  alpha=alpha, beta=beta,
                                                 bias=bias, name=name)
