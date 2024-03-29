from RUSH_CV.utils import RunningAverage

class BaseEvaluation(RunningAverage):
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, Y, Y_pred, *args, **kwargs):
        # if len(Y.shape) == 4:
        #     for i in range(Y.shape[0]):
        #         self.total += self.scoring(Y[i], Y_pred[i], *args, **kwargs)
        # else:
        self.total += self.scoring(Y, Y_pred, *args, **kwargs)

        self.steps += 1

    def reset(self):
        self.steps = 0
        self.total = 0 

    def scoring(self, Y, Y_pred, *args, **kwargs):
        raise NotImplementedError

    def __call__(self):
        return self.total / float(self.steps) 
    
