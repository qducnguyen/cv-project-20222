from .evaluation_unet import main as evaluation_main
from .inference_unet import main as inference_main
from .training_unet import main as training_main

class UNetTrainer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        training_main.main(args=self.args)

class UnetEvaluator():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        evaluation_main.main(args=self.args)

class UNetInferencer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        inference_main(args=self.args)
