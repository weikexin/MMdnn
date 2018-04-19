# coding=utf-8
from coremltools.models.utils import load_spec as _load_spec
from coremltools.models import MLModel as _MLModel

COREML_FILE = 'keras_mobilenet.mlmodel'

class CoreMLParser():

    @property
    def src_graph(self):
        return self.coreml_graph

    def __init__(self, model=None):
        self.spec = _load_spec(COREML_FILE)

    def parse(self):
        with open('mobilenet_coreml.struct','w') as f:
            for layer in self.spec.neuralNetwork.layers:
                print >> f, "======================================"
                print >> f, layer.name
                print >> f, layer.WhichOneof('layer')
                print >> f, layer.input
                print >> f, layer.output
parser = CoreMLParser()
parser.parse()
