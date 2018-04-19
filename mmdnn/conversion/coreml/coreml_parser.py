# coding=utf-8
from coremltools.models.utils import load_spec as _load_spec
from coremltools.models import MLModel as _MLModel

import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.utils import *

from mmdnn.conversion.coreml.coreml_graph import CoreMLGraph

COREML_FILE = 'keras_mobilenet.mlmodel'

class CoreMLParser(Parser):

    @property
    def src_graph(self):
        return self.coreml_graph

    def __init__(self, model=None):
        super(CoreMLParser, self).__init__()
        spec = _load_spec(COREML_FILE)
        self.coreml_graph = CoreMLGraph(spec)
        self.coreml_graph.build()

    def gen_IR(self):
#        types = set()
        for layer in self.coreml_graph.topological_sort:
            current_node = self.coreml_graph.get_node(layer)
            node_type = current_node.type
            if node_type == 'temp_layer':
                continue
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)
            else:
                print("CoreMLParser has not supported operator [%s]." % (node_type))
                self.rename_UNKNOWN(current_node)
#                types.add(node_type)

#        print "======================================"
#        print types

    @staticmethod
    def tensor_shape_to_list(shapes):
        if isinstance(shapes, attr_value_pb2.AttrValue):
            return [dim.size for dim in shapes.shape.dim]
        else:
            ret = []
            for shape in shapes:
                this_one = [dim.size for dim in shape.dim]
                ret.append(this_one)
            return ret

    def _convert_padding(self, input_shape, IR_node, kernel_size):
        # TODO: Fused conv and pool with padding is different from defused operators
        if source_node.get_attr('padding') == 'VALID':
            dims = len(input_shape)
            assign_IRnode_values(IR_node, {'auto_pad': "VALID", 'pads': [0, 0] * dims})

        elif source_node.get_attr('padding') == 'SAME':
            padding = compute_tf_same_padding(
                input_shape,
                kernel_size,
                source_node.get_attr('strides'))
            assign_IRnode_values(IR_node, {'auto_pad': "SAME_LOWER", 'pads' : padding})
        else:
            assert False


    def _gen_IR_node(self, source_node, new_op=None):
        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.name
        IR_node.op = source_node.type if new_op == None else new_op
        self.convert_inedge(source_node, IR_node)
        return IR_node

    def rename_UNKNOWN(self, source_node):
        print("CoreMLParser has not supported operator [%s] with name [%s]."
              % (source_node.type, source_node.name))
        return

    def compute_tf_same_padding(input_shape, kernel_shape, strides, data_format='NHWC'):
        """ Convert [SAME] padding in tensorflow, keras to onnx pads,
            i.e. [x1_begin, x2_begin...x1_end, x2_end,...] """
        # print (input_shape)
        # print (kernel_shape)
        # print (strides)
        if data_format.startswith('NC'):
            # Not tested
            input_shape = input_shape[2:]
            remove_dim = len(strides) - len(input_shape)
            if remove_dim > 0:
                strides = strides[remove_dim::]

        else:
            input_shape = input_shape[1:-1]
            remove_dim = len(input_shape) - len(strides) + 1
            if remove_dim < 0:
                strides = strides[1:remove_dim]

        # print (input_shape)
        # print (kernel_shape)
        # print (strides)

        up_list = [0]
        down_list = [0]

        for idx in range(0, len(input_shape)):
            # kernel_shape[idx] = (kernel_shape[idx] - 1) * dilation_rate + 1
            output_shape = (input_shape[idx] + strides[idx] - 1) // strides[idx]
            this_padding = (output_shape - 1) * strides[idx] + kernel_shape[idx] - input_shape[idx]
            this_padding = max(0, this_padding)
            up_list.append(this_padding // 2)
            down_list.append(this_padding - this_padding // 2)

        # print ([0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0])
        # print ('-----------------------------------------------------')
        return [0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0]



    def rename_convolution(self, source_node):
        IR_node = self._gen_IR_node(source_node, 'Conv')
        conv = source_node.layer.convolution
        kwargs = {}
        self.set_weight(source_node.name, "weights", conv.weights)
        if conv.hasBias:
            kwargs['use_bias'] = True
            self.set_weight(source_node.name, "bias", conv.bias)
        # The IR strides is same to tensorflow
        kwargs['strides'] = [1] + list(conv.stride) + [1]
        # The IR kenel_shape format is HWCN
        kwargs['kernel_shape'] = list(conv.kernelSize)
        kwargs['kernel_shape'].append(conv.kernelChannels)
        kwargs['kernel_shape'].append(conv.outputChannels)
        # print kwargs
        assign_IRnode_values(IR_node, kwargs)

        if conv.WhichOneof('ConvolutionPaddingType') == 'Valid':
            dims = len(conv.weights.shape)
            assign_IRnode_values(IR_node, {'auto_pad': "VALID", 'pads': [0, 0] * dims})
            print input_node
            pass
        else:
            pass
            print "========================================="
            padding = compute_tf_same_padding(
                conv.weights.shape,
                kernel_size,
                source_node.get_attr('strides'))
            assign_IRnode_values(IR_node, {'auto_pad' : "SAME_LOWER", 'pads' : padding})


    def rename_pooling(self, source_node):
        pass

    def rename_padding(self, source_node):
        pass

    def rename_dropout(self, source_node):
        pass

    def rename_innerProduct(self, source_node):
        pass

    def rename_flatten(self, source_node):
        pass

    def rename_reshape(self, source_node):
        pass

    def rename_activation(self, source_node):
        pass

    def rename_add(self, source_node):
        pass

    def rename_concat(self, source_node):
        pass

    def rename_batchnorm(self, source_node):
        pass

    def rename_unary(self, source_node):
        pass

    def rename_softmax(self, source_node):
        pass

    def print_details(self):
        print '====================================='
        for layer_name in self.coreml_graph.layer_map:
            print layer_name


parser = CoreMLParser()
parser.gen_IR()
