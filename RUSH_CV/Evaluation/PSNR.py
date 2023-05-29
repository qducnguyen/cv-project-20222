import numpy as np

from RUSH_CV.Base.BaseEvaluation import BaseEvaluation

class PSNR(BaseEvaluation):
    def update(self, Y, Y_pred, *args, **kwargs):
        return 10. * np.log10(1. / np.mean((Y - Y_pred) ** 2))
    
