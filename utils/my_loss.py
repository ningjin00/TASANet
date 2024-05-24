import torch
from torch.nn import Module
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, \
    ErrorRelativeGlobalDimensionlessSynthesis, SpectralAngleMapper, RelativeAverageSpectralError, \
    RootMeanSquaredErrorUsingSlidingWindow
class TASALoss(Module):
    def __init__(self, device,r1=2,r2=2):
        super().__init__()
        self.device=device
        self.r1=r1
        self.r2=r2
    def forward(self, x, y):
        mae_fun = MeanAbsoluteError().to(self.device)
        psnr_fun = PeakSignalNoiseRatio().to(self.device)
        ssim_fun = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        l_1 = mae_fun(x, y)
        l_psnr = psnr_fun(x, y)
        if torch.isnan(l_psnr):
            return l_psnr
        l_psnr = 1.0 / l_psnr
        l_ssim = 1 - ssim_fun(x, y)
        return self.r1*l_1+self.r2*l_psnr + l_ssim
