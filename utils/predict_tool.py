import time
from modules.TASA import TASA
from utils.config_loader import load_config
import torch
from utils.dataset_predict_loader import PredictDataset
from utils.img_metrics import ImgAllMetrics
from utils.whether_use_gpu import whether_use_gpu
from utils.img_tool import save_sar_img, save_sigle_img, save_three_img, save_four_img
def predict(model_name):
    main_config_path = "config/main_config.yaml"
    main_config = load_config(main_config_path)
    del main_config_path
    main_config["model_name"] = model_name
    predict_config_path = "config/predict_config.yaml"
    predict_config = load_config(predict_config_path)
    del predict_config_path
    print("预测模型：{}".format(main_config["model_name"]))
    is_gpu = whether_use_gpu(main_config["use_gpu"])
    del main_config["use_gpu"]
    device = torch.device("cuda" if is_gpu else "cpu")
    del is_gpu
    dataset_config_path = "config/dataset_config.yaml"
    dataset_config = load_config(dataset_config_path)
    del dataset_config_path
    dataset = PredictDataset(dataset_config['predict_dataset_dir'], dataset_config)
    print("数据集预加载，数据集的总对数为：{}".format(len(dataset)))
    predict_dataloader = torch.utils.data.DataLoader(dataset, batch_size=predict_config["batch_size"],
                                                     shuffle=False,num_workers=2)
    predict_dataloader_size = len(predict_dataloader)
    print("数据集初始化完毕，数据集的批次数为：{}".format(predict_dataloader_size))
    model = TASA()
    if predict_config["model_load_path"]:
        checkpoint = torch.load(predict_config["model_load_path"])
        try:
            model.load_state_dict(checkpoint['model'])
            print("已经加载checkpoint！！")
        except Exception:
            model.load_state_dict(checkpoint)
            print("没有加载checkpoint！！")
        print("载入\"{}\"作为网络模型".format(predict_config["model_load_path"]))
    else:
        raise Exception(
            "您没有输入网络模型路径，请在\"config/predict_config.yaml\"中的model_load_path行后面加上网络路径")

    model = model.to(device)
    x = torch.FloatTensor(predict_config["batch_size"], 15, 256, 256)
    x = x.to(device)
    y = torch.FloatTensor(predict_config["batch_size"], 13, 256, 256)
    y = y.to(device)
    img_metrics = ImgAllMetrics(device)
    model = model.eval()
    start_time = time.strftime("%Y{y}%m{m}%d{d}%H{i}%M{j}%S{k}").format(y="年", m="月", d="日", i="时", j="分",
                                                                        k="秒")
    with torch.no_grad():
        for iteration, (x, y, truth_img_path_list) in enumerate(
                predict_dataloader):
            x = x.to(device)
            y = y.to(device)

            time.sleep(0.2)
            out = model(x)
            if iteration % 100 == 0:
                print("{}/{}".format(iteration, predict_dataloader_size), end="   ")
                temp = int((iteration * 1.0 / predict_dataloader_size) * 100)
                print("  [ {}% ]  [".format(temp), end="")
                for i in range(temp):
                    print(">", end="")
                for i in range(100 - temp):
                    print("-", end="")
                print("]")
            img_metrics.add_mean_list(out, y)
            # img_metrics.save_sigle_metrics(out, y, main_config["model_name"],
            #                                start_time, row_name=truth_img_path_list)
            # img_metrics.add_mean_list_no()
            # save_sigle_img(x[:, 2:, :, :],predict_config["model_output_path"]+"s2_cloudy/", truth_img_path_list)
            # save_sigle_img(out,predict_config["model_output_path"]+"s2_"+main_config["model_name"]+"/", truth_img_path_list)
            # save_sigle_img(y,predict_config["model_output_path"]+cloud_coverage+"/s2/", truth_img_path_list)
            # save_sar_img(x[:, :2, :, :],predict_config["model_output_path"]+"sar/", truth_img_path_list)
            save_three_img(x[:, 2:, :, :],out,y,predict_config["model_output_path"], str(iteration))
        img_metrics.save_mean_metrics(main_config["model_name"])
    print("预测成功！")
