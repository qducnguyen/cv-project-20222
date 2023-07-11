from .evaluation_srgan import main as evaluation_main
from .inference_srgan import main as inference_main
from .training_srgan import main as training_main

class SRGANTrainer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        training_main(args=self.args)

class SRGANEvaluator():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        evaluation_main(args=self.args)

class SRGANInferencer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        return inference_main(args=self.args)
