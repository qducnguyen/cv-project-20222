from .evaluation_edsr import main as evaluation_main
from .inference_edsr import main as inference_main
from .training_edsr import main as training_main

class EDSRTrainer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        training_main.main(args=self.args)

class EDSREvaluator():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        evaluation_main.main(args=self.args)

class EDSRInferencer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        inference_main(args=self.args)
