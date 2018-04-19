from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from coremltools.proto.NeuralNetwork_pb2 import NeuralNetworkLayer


def _handle_scalar_feature(cm_value, doc_string=''):
    which_type = cm_value.type.WhichOneof('Type')
    onnx_type = _convert(which_type)
    onnx_shape = [1]
    return model_util.make_tensor_value_info(cm_value.name, onnx_type, onnx_shape, doc_string)


def _handle_multi_array_feature(cm_value, batch_size=1, doc_string=''):
    data_type = cm_value.type.multiArrayType.dataType
    onnx_type = _convert(data_type)
    onnx_shape = [batch_size]
    for shape_val in cm_value.type.multiArrayType.shape:
        onnx_shape.append(shape_val)
    return model_util.make_tensor_value_info(cm_value.name, onnx_type, onnx_shape, doc_string)


def _handle_dictionary_feature(cm_value, doc_string=''):
    key_type = cm_value.type.dictionaryType.WhichOneof('KeyType')
    onnx_key_type = _convert(key_type)
    onnx_value_type = onnx_proto.TensorProto.FLOAT
    map_type = model_util.make_map_value_info(cm_value.name, onnx_key_type, onnx_value_type, doc_string)
    return map_type

def _handle_image_feature(cm_value, batch_size=1, doc_string=''):
    # ONNX currently doesn't have image type, so we use tensor as images' representations.
    # One issue is that we are not able to add side information such as color space.
    onnx_type = onnx_proto.TensorProto.FLOAT
    if len(doc_string) > 0:
        doc_string = doc_string + (' ' if doc_string.endswith('.') else '. ')
    if cm_value.type.imageType.colorSpace == 10:
        onnx_shape = [batch_size, 1]
        doc_string = doc_string + 'Image(s) in gray scale. If there are N images, it is a 4-D tensor with shape [N, 1, H, W].'
    elif cm_value.type.imageType.colorSpace == 20:
        onnx_shape = [batch_size, 3]
        doc_string = doc_string + 'Image(s) in RGB format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the' \
                     'C-axis are red, green, and blue channels, respectively.'
    elif cm_value.type.imageType.colorSpace == 30:
        onnx_shape = [batch_size, 3]
        doc_string = doc_string + 'Image(s) in BGR format. It is a [N, C, H, W]-tensor. The 1st/2nd/3rd slices along the' \
                     'C-axis are blue, green, and red channels, respectively.'
    else:
        raise ValueError('Unsupported color space')
    onnx_shape.append(cm_value.type.imageType.height)
    onnx_shape.append(cm_value.type.imageType.width)
    return model_util.make_tensor_value_info(cm_value.name, onnx_type, onnx_shape, doc_string)





class CoreMLGraphNode(GraphNode):

    def __init__(self, layer):
        super(CoreMLGraphNode, self).__init__(layer)

    @property
    def name(self):
        return self.layer.name

    @property
    def type(self):
        type = self.layer.WhichOneof('layer')
        if type:
            return type
        else:
            return 'temp_layer'

    @property
    def keras_layer(self):
        return self.layer


class CoreMLGraph(Graph):

    def __init__(self, spec):
        super(CoreMLGraph, self).__init__(spec)
        self.spec = spec
        nn_type = spec.WhichOneof('Type')
        if nn_type == 'neuralNetwork':
            self.model = CoreMLGraph(spec.neuralNetwork)
        elif nn_type == 'neuralNetworkRegressor':
            self.model = CoreMLGraph(spec.neuralNetworkRegressor)
        elif nn_type == 'neuralNetworkClassifier':
            self.model = CoreMLGraph(spec.neuralNetworkClassifier)
        else:
            print('only support neural network conversion')

    def input(self):
        for graph_input in self.spec.description.input:
            input_tensor = make_value_info(graph_input, batch_size)


    def build(self):
        for i, layer in enumerate(self.model.layers):
            self.layer_map[layer.name] = CoreMLGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
            if layer.output is not None and len(layer.output) == 1:
                oneout = layer.output[0]
                if oneout not in self.layer_name_map:
                    self.layer_name_map[oneout] = layer.name

        for i, layer in enumerate(self.model.layers):
            for pred in layer.input:
                if pred not in self.layer_name_map:
                    #print "not in", pred
                    new_node = NeuralNetworkLayer()
                    new_node.name = pred
                    self.layer_map[pred] = CoreMLGraphNode(new_node)
                    self.layer_name_map[pred] = pred
                else:
                    pred = self.layer_name_map[pred]
                    #print "in:", pred
                self._make_connection(pred, layer.name)

        super(CoreMLGraph, self).build()
