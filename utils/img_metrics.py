import time
import csv
import torch
_ = torch.manual_seed(42)
from torchmetrics.image import RelativeAverageSpectralError
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import UniversalImageQualityIndex
from torchmetrics.image import VisualInformationFidelity
from torchmetrics.image import ErrorRelativeGlobalDimensionlessSynthesis
from torchmetrics.image import RootMeanSquaredErrorUsingSlidingWindow
from torchmetrics.image import SpectralAngleMapper
class ImgAllMetrics:
    def __init__(self, device):
        self.ssim_list = []
        self.psnr_list = []
        self.uqi_list = []
        self.vif_list = []
        self.mae_list = []
        self.rmse_list = []
        self.sam_list = []
        self.rase_list = []
        self.ergas_list = []
        self.ssim = torch.tensor(0.0)
        self.psnr = torch.tensor(0.0)
        self.uqi = torch.tensor(0.0)
        self.vif = torch.tensor(0.0)
        self.mae = torch.tensor(0.0)
        self.rmse = torch.tensor(0.0)
        self.sam = torch.tensor(0.0)
        self.rase = torch.tensor(0.0)
        self.ergas = torch.tensor(0.0)
        self.device = device
        self.best_ssim = 0.0
        self.best_psnr = 0.0

    def get_all_metrics(self, x, y):
        self.ssim = self.__get_ssim(x, y)
        self.psnr = self.__get_psnr(x, y)
        self.uqi = self.__get_uqi(x, y)
        self.vif = self.__get_vif(x, y)
        self.mae = self.__get_mae(x, y)
        self.rmse = self.__get_rmse(x, y)
        self.sam = self.__get_sam(x, y)
        self.rase = self.__get_rase(x, y)
        self.ergas = self.__get_ergas(x, y)
    def save_sigle_metrics(self,x,y,model_name,start_time,row_name=""):
        self.get_all_metrics(x, y)
        row_name="".join(row_name[2][0])+"".join(row_name[2][1])
        save_csv_path = model_name + "_" + start_time + ".csv"
        content=[row_name,self.ssim.cpu().numpy(),self.psnr.cpu().numpy(),self.uqi.cpu().numpy(),
                 self.vif.cpu().numpy(),self.mae.cpu().numpy(),
                 self.rmse.cpu().numpy(),self.sam.cpu().numpy(),
                 self.rase.cpu().numpy(),self.ergas.cpu().numpy()]
        with open(save_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(content)
    def print_model_metrics(self):
        print(
            "SSIM:{:.3f}  PSNR:{:.1f}  UQI:{:.3f}  VIF:{:.3f}  MAE:{:.3f}  RMSE:{:.3f}  SAM:{:.4f}  RASE:{:.0f}  ERGAS:{:.0f}".format(
                self.ssim, self.psnr, self.uqi, self.vif,
                self.mae, self.rmse, self.sam, self.rase, self.ergas))
        if self.best_psnr < self.psnr:
            self.best_psnr = self.psnr
        if self.best_ssim < self.ssim:
            self.best_ssim = self.ssim
        print("             best-SSIM:{:.3f}  best-PSNR:{:.1f}  ".format(self.best_ssim, self.best_psnr))
        return self.best_ssim, self.best_psnr

    def add_mean_list(self, x, y):
        self.get_all_metrics(x, y)
        self.ssim_list.append(self.ssim)
        self.psnr_list.append(self.psnr)
        self.uqi_list.append(self.uqi)
        self.vif_list.append(self.vif)
        self.mae_list.append(self.mae)
        self.rmse_list.append(self.rmse)
        self.sam_list.append(self.sam)
        self.rase_list.append(self.rase)
        self.ergas_list.append(self.ergas)
    def add_mean_list_no(self):
        self.ssim_list.append(self.ssim)
        self.psnr_list.append(self.psnr)
        self.uqi_list.append(self.uqi)
        self.vif_list.append(self.vif)
        self.mae_list.append(self.mae)
        self.rmse_list.append(self.rmse)
        self.sam_list.append(self.sam)
        self.rase_list.append(self.rase)
        self.ergas_list.append(self.ergas)
    def save_mean_metrics(self, model_name ):
        end_time = time.strftime("%Y{y}%m{m}%d{d}%H{i}%M{j}%S{k}").format(y="年", m="月", d="日", i="时", j="分",
                                                                          k="秒")
        save_csv_path = "./model_state_save/log/" + model_name + "_" + end_time + ".csv"
        total_csv_path = "./model_state_save/log/" + "1_total_log.csv"
        catalogue = ["method", "end_time", "SSIM_mean", "PSNR_mean", "SCC_mean", "UQI_mean",
                     "VIF_mean", "MAE_mean", "RMSE_mean", "SAM_mean", "RASE_mean", "ERGAS_mean"]
        content = [model_name, end_time,
                   torch.mean(torch.Tensor(self.ssim_list)).numpy(), torch.mean(torch.Tensor(self.psnr_list)).numpy(),
                    torch.mean(torch.Tensor(self.uqi_list)).numpy(),
                   torch.mean(torch.Tensor(self.vif_list)).numpy(), torch.mean(torch.Tensor(self.mae_list)).numpy(),
                   torch.mean(torch.Tensor(self.rmse_list)).numpy(), torch.mean(torch.Tensor(self.sam_list)).numpy(),
                   torch.mean(torch.Tensor(self.rase_list)).numpy(), torch.mean(torch.Tensor(self.ergas_list)).numpy()]
        for i in range(len(catalogue)):
            print("   {}:{}".format(catalogue[i], content[i]))
        try:
            with open(save_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(catalogue)
                writer.writerow(content)
        except Exception:
            print("报错！！！保存路径不存在")
            print("报错！！！保存路径不存在")
            save_csv_path = model_name + "_" + end_time + ".csv"
            with open(save_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(catalogue)
                writer.writerow(content)
            return

        with open(total_csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(content)
    def __get_mae(self, x, y):
        mae_fun = MeanAbsoluteError().to(self.device)
        return mae_fun(x, y)
    def __get_ssim(self, x, y):
        ssim_fun = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        return ssim_fun(x, y)
    def __get_psnr(self, x, y):
        psnr_fun = PeakSignalNoiseRatio().to(self.device)
        return psnr_fun(x, y)
    def __get_uqi(self, x, y):
        uqi_fun = UniversalImageQualityIndex().to(self.device)
        return uqi_fun(x, y)
    def __get_vif(self, x, y):
        vif_fun = VisualInformationFidelity().to(self.device)
        return vif_fun(x * 10000, y * 10000)
    def __get_rase(self, x, y):
        rase_fun = RelativeAverageSpectralError().to(self.device)
        return rase_fun(x, y)
    def __get_ergas(self, x, y):
        ergas_fun = ErrorRelativeGlobalDimensionlessSynthesis().to(self.device)
        return ergas_fun(x, y)
    def __get_rmse(self, x, y):
        rmse_fun = RootMeanSquaredErrorUsingSlidingWindow().to(self.device)
        return rmse_fun(x, y)
    def __get_sam(self, x, y):
        sam_fun = SpectralAngleMapper().to(self.device)
        return sam_fun(x, y)
