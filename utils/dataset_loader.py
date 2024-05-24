import torch
from utils.config_loader import load_config
import os
from utils.path_tool import add_reverse_rake_bar
from utils.path_tool import join_dir_list
from utils.path_tool import join_dir
from utils.path_tool import find_file_list
import rasterio
import numpy as np
from torch.utils import data
class TrainDataset(data.Dataset):
    def __init__(self, dataset_dir,dataset_config):
        super().__init__()
        self.s1_scale=dataset_config["s1_scale"]
        self.s2_scale=dataset_config["s2_scale"]
        self.dataset_dir = add_reverse_rake_bar(dataset_dir)
        del dataset_config
        if not os.path.exists(self.dataset_dir):
            raise Exception("不存在该目录!请检查您的数据集路径")
        self.img = []
        season_list = self.get_season_list()
        for season in season_list:
            roi_list = self.get_roi_list(season)
            for roi in roi_list:
                patch_list = self.get_patch_list([season, "s1_" + str(roi)])
                for patch in patch_list:
                    self.img.append([season, roi, patch])
    def get_season_list(self):
        season_list = find_file_list(self.dataset_dir)
        return season_list
    def get_roi_list(self, season):
        path = join_dir(self.dataset_dir, season)
        roi_dir_list = find_file_list(path)
        roi_list = [int(s.split('_')[-1]) for s in roi_dir_list]
        return set(roi_list)

    def get_patch_list(self, file_list):
        path = join_dir_list(self.dataset_dir, file_list)
        patch_list = []
        patch_dir_list = find_file_list(path)
        for patch_dir in patch_dir_list:
            patch = patch_dir.split("_")
            patch_list.append([patch[0] + "_" + patch[1] + "_", "_" + patch[3] + "_" + patch[4]])
        return patch_list

    def get_img_data(self, img_path):
        img_data=None
        img_path=img_path[:-1]
        with rasterio.open(img_path) as img_inf:
            img_data = img_inf.read()
            img_data_tensor = torch.from_numpy(img_data.astype("float32"))
            return img_data_tensor
        raise("rasterio打开文件失败！")
    def get_normalized_data(self, img_path_list):
        s1_data = self.get_img_data(img_path_list[0])
        s1_normalized_data = torch.clip(s1_data,-100,0)
        s2_data = self.get_img_data(img_path_list[1])
        s2_normalized_data =torch.clip(s2_data,0,10000)
        s2_cloudy_data = self.get_img_data(img_path_list[2])
        s2_cloudy_normalized_data=torch.clip(s2_cloudy_data,0,10000)
        return s1_normalized_data, s2_normalized_data, s2_cloudy_normalized_data

    def get_data_triplet(self, img_dir_list):
        s1_path = join_dir_list(self.dataset_dir, [img_dir_list[0], "s1_" + str(img_dir_list[1]),
                                                   img_dir_list[2][0] + "s1" + img_dir_list[2][1]])
        s2_path = join_dir_list(self.dataset_dir, [img_dir_list[0], "s2_" + str(img_dir_list[1]),
                                                   img_dir_list[2][0] + "s2" + img_dir_list[2][1]])
        s2_cloudy_path = join_dir_list(self.dataset_dir, [img_dir_list[0], "s2_cloudy_" + str(img_dir_list[1]),
                                                          img_dir_list[2][0] + "s2_cloudy" + img_dir_list[2][1]])
        return self.get_normalized_data([s1_path, s2_path, s2_cloudy_path])
    def get_input_output_data(self,index):
        s1_img, output_s2_img_truth, s2_cloudy_img = self.get_data_triplet(self.img[index])
        s1_img_float=s1_img/self.s1_scale
        output_s2_img_truth_float=output_s2_img_truth/self.s2_scale
        s2_cloudy_img_float=s2_cloudy_img/self.s2_scale
        input_s1_s2_cloudy = np.concatenate((s1_img_float, s2_cloudy_img_float), axis=0)
        return input_s1_s2_cloudy,output_s2_img_truth_float
    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        input_s1_s2_cloudy,output_s2_img_truth=self.get_input_output_data(index)
        return input_s1_s2_cloudy, output_s2_img_truth


if __name__ == '__main__':
    dataset_config_path = "../config/dataset_config.yaml"
    dataset_config =load_config(dataset_config_path)
    train_config_path = "../config/train_config.yaml"
    train_config = load_config(train_config_path)
    print("正在创建数据集...")
    dataset = TrainDataset(train_config['train_dataset_dir'], dataset_config)
    d=dataset.__len__()
    input_data, s2_img = dataset.__getitem__(2)
    print(input_data, s2_img)
