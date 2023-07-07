from .evaluation_srcnn import main as evaluation_main
from .inference_srcnn import main as inference_main
from .training_srcnn import main as training_main

class SRCNNTrainer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        training_main(args=self.args)

class SRCNNEvaluator():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        evaluation_main(args=self.args)

class SRCNNInferencer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        inference_main(args=self.args)
