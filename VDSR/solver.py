from .evaluation_vdsr import main as evaluation_main
from .inference_vdsr import main as inference_main
from .training_vdsr import main as training_main

class VDSRTrainer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        training_main(args=self.args)

class VDSREvaluator():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        evaluation_main(args=self.args)

class VDSRInferencer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        return inference_main(args=self.args)
