from utils.config_loader import load_config
from utils.dataset_loader import TrainDataset
class PredictDataset(TrainDataset):
    def __init__(self,dataset_dir,dataset_config):
        super().__init__(dataset_dir,dataset_config)
    def __getitem__(self, index):
        input_s1_s2_cloudy,output_s2_img_truth=self.get_input_output_data(index)
        return input_s1_s2_cloudy, output_s2_img_truth,self.img[index]
if __name__ == '__main__':
    dataset_config_path = "../config/dataset_config.yaml"
    dataset_config = load_config(dataset_config_path)
    train_config_path = "../config/train_config.yaml"
    train_config = load_config(train_config_path)
    print("正在创建数据集...")
    dataset = PredictDataset(train_config['train_dataset_dir'], dataset_config)
    input_data, s2_img,img_path_list = dataset.__getitem__(2)
    print(img_path_list)