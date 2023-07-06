from .evaluation_bicubic import main as evaluation_main
from .inference_bicubic import main as inference_main


class BicubicEvaluator():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        evaluation_main.main(args=self.args)

class BicubicInferencer():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        inference_main(args=self.args)
