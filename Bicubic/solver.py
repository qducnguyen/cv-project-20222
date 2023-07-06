from .evaluation_bicubic import main

class BicubicEvaluator():
    def __init__(self, args):
        self.args = args
    
    def run(self):
        main(args=self.args)