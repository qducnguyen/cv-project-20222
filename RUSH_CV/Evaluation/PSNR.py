import numpy as np

import torch
import numpy as np


from RUSH_CV.Base.BaseEvaluation import BaseEvaluation

class PSNR(BaseEvaluation):
    def scoring(self, Y, Y_pred, *args, **kwargs):
        if type(Y) is np.ndarray:
            return 10. * np.log10(1. / np.mean((Y - Y_pred) ** 2))
        else:
            return 10. * torch.log10(1. / torch.mean((Y - Y_pred) ** 2)).cpu().numpy()

