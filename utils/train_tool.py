import csv
import time
import torch
from torchinfo import summary
from modules.TASA import TASA
from utils.config_loader import load_config
from utils.dataset_loader import TrainDataset
from utils.img_tool import  train_save_three_img
from utils.model_tool import save_model, get_epoch_and_model_load_path, get_model_save_dir, model_show_time
from utils.img_metrics import ImgAllMetrics
from utils.my_loss import TASALoss
from utils.whether_use_gpu import whether_use_gpu
def save_train_state(model_name, epoch, iteration, best_loss, best_SSIM, best_PSNR, first_creat=0):
    state_csv_path = "model_state.csv"
    end_time = time.strftime("%Y{y}%m{m}%d{d}%H{i}%M{j}%S{k}").format(y="年", m="月", d="日", i="时", j="分", k="秒")
    content = [end_time, model_name, epoch, iteration, best_loss, best_SSIM, best_PSNR]
    with open(state_csv_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(content)
        print("-----写入模型训练记录！")


def train(model_name):
    print("###############################")
    print("model_name=", model_name)
    print("###############################")
    main_config_path = "config/main_config.yaml"
    main_config = load_config(main_config_path)
    del main_config_path
    main_config["model_name"] = model_name
    train_config_path = "config/train_config.yaml"
    train_config = load_config(train_config_path)
    del train_config_path
    is_gpu = whether_use_gpu(main_config["use_gpu"])
    del main_config["use_gpu"]
    device = torch.device("cuda" if is_gpu else "cpu")
    del is_gpu
    is_save = 0
    with open("is_save.txt", "w") as f:
        f.write(str(is_save))
    dataset_config_path = "config/dataset_config.yaml"
    dataset_config = load_config(dataset_config_path)
    del dataset_config_path
    print("正在创建数据集...")
    train_dataset = TrainDataset(dataset_config["train_dataset_dir"], dataset_config)
    del dataset_config["train_dataset_dir"]
    print("预加载_训练_数据总共对数:  {}  ".format(len(train_dataset)))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=train_config["batch_size"],
                                                   shuffle=True ,num_workers=2)
    del train_dataset
    train_dataloader_size = len(train_dataloader)
    print("已经加载_训练_数据_批次：  {}    \n 训练数据集初始化完毕!".format(train_dataloader_size))
    if train_config["use_eval"]:
        eval_dataset = TrainDataset(dataset_config["eval_dataset_dir"], dataset_config)
        print("验证数据集预加载，数据集的总对数为：{}".format(len(eval_dataset)))
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=train_config["batch_size"],
                                                      shuffle=True,num_workers=2)
        del eval_dataset
        eval_dataloader_size = int(len(eval_dataloader))
        print("已经加载_验证_数据大小：  {}    \n 验证数据集初始化完毕!".format(eval_dataloader_size))
        eval_dataiter = iter(eval_dataloader)
    del dataset_config["eval_dataset_dir"]
    model = TASA()
    loss_function = TASALoss(device)
    print("-----使用损失函数 2 2 -----")
    print(loss_function)
    if train_config["is_print_model"]:
        print(model)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        summary(model, input_size=(train_config["batch_size"], 15, 256, 256),
                col_names=("input_size", "output_size", "num_params", "kernel_size"))
    del train_config["is_print_model"]
    train_config["model_save_dir"] = get_model_save_dir(train_config["model_save_dir"], main_config["model_name"])
    epoch_start = 1
    if not train_config["model_load_path"]:
        temp = get_epoch_and_model_load_path(train_config["model_save_dir"])
        if temp:
            (epoch_start, train_config["model_load_path"]) = temp
    epoch_end = epoch_start + train_config["epoch"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
    if train_config["model_load_path"]:
        checkpoint = torch.load(train_config["model_load_path"])
        try:
            model.load_state_dict(checkpoint['model'])
            print("已经加载checkpoint！！")
        except Exception:
            model.load_state_dict(checkpoint)
            optimizer = torch.optim.Adam(model.parameters(), lr=train_config["learning_rate"])
            print("没有加载checkpoint！！")
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("已经加载优化器！！")
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
        except Exception:
            print("没有加载优化器！！")
        print("载入{}作为网络模型".format(train_config["model_load_path"]))
    else:
        print("没有加载模型，路径为空")
    model = model.to(device)
    img_metrics = ImgAllMetrics(device)
    print("---------------------")
    temp = 0
    print("开始训练...")
    model.train()
    best_loss = 100.0
    best_SSIM, best_PSNR = 0.0, 0.0
    x = torch.FloatTensor(train_config["batch_size"], 15, 256, 256)
    x = x.to(device)
    y = torch.FloatTensor(train_config["batch_size"], 13, 256, 256)
    y = y.to(device)
    out = torch.FloatTensor(train_config["batch_size"], 13, 256, 256)
    out = out.to(device)
    out_show = torch.FloatTensor( 13, 256, 256)
    out_show = out_show.to(device)
    for epoch in range(epoch_start, epoch_end + 200):
        if train_config["use_eval"]:
            eval_dataiter = iter(eval_dataloader)
        epoch_start_time = time.time()
        start_time = time.strftime("%Y{y}%m{m}%d{d}%H{i}%M{j}%S{k}").format(y="年", m="月", d="日", i="时", j="分",
                                                                            k="秒")
        print(" 第{}轮开始时间： {}".format(epoch, start_time))
        iter_eval = 0
        for iteration, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            time.sleep(0.2)
            out = model(x)
            time.sleep(0.2)
            out_show=out[0, :, :, :]
            optimizer.zero_grad()
            loss = loss_function(out, y)
            if torch.isnan(loss):
                print(loss)
                print(loss)
                print("  loss为空，请重新加载模型！！！！！！")
                print("  loss为空，请重新加载模型！！！！！！")
                save_train_state(main_config["model_name"], epoch=epoch, iteration=iteration, best_loss=best_loss,
                                 best_SSIM=best_SSIM, best_PSNR=best_PSNR)
                return True
            if iteration % 20 == 0:
                print("epoch:{}  ( {}/{} )  loss:{:.4f}   ".format(epoch, iteration, train_dataloader_size,
                                                                   loss.item()), end="")
                temp = int((iteration * 1.0 / train_dataloader_size) * 100)
                loss_cur = loss.item()
                if loss_cur < best_loss:
                    best_loss = loss_cur
                print("beat-loss:{:.4f}".format(best_loss), end=" ")
                print("  [ {}% ]  [".format(temp), end="")
                for i in range(temp):
                    print(">", end="")
                for i in range(100 - temp):
                    print("-", end="")
                print("]")
            loss.backward()
            optimizer.step()
            if (iteration % train_config["model_train_show_frequency"] == 0
                    and train_config["use_eval"] and iteration > 1):
                with torch.no_grad():
                    print("\nmodel:{}".format(main_config["model_name"]), end="  ")
                    print("eval: ", end="")
                    iter_eval = iter_eval + 1
                    try:
                        x_val, y_val = next(eval_dataiter)
                    except StopIteration:
                        eval_dataiter = iter(eval_dataloader)
                        x_val, y_val = next(eval_dataiter)
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    model.eval()
                    out_val = model(x_val)  #
                    loss = loss_function(out_val, y_val)
                    print("epoch:{}  ({}/{})   loss:{:.4f}".format(epoch, iter_eval, eval_dataloader_size, loss.item()),
                          end="   ")
                    img_metrics.get_all_metrics(out_val, y_val)
                    best_SSIM, best_PSNR = img_metrics.print_model_metrics()
                    print("\n")
                    model.train()
            if iteration % train_config["model_save_frequency_iteration"] == 0 and iteration > 10:
                save_model(model, optimizer, train_config["model_save_dir"], main_config["model_name"], epoch,
                           iteration)
                train_save_three_img(x[0, 2:, :, :], out_show, y[0, :, :, :],
                                     train_config["show_img_path"] + main_config["model_name"] + "/",
                                     epoch)
            with open("is_save.txt", "r") as f:
                is_save = int(f.read())
            if is_save:
                print("进行保存!!  保存后停止")
                save_model(model, optimizer, train_config["model_save_dir"], main_config["model_name"], epoch,
                           iteration)
                train_save_three_img(x[0, 2:, :, :], out_show, y[0, :, :, :],
                                     train_config["show_img_path"] + main_config["model_name"] + "/",
                                     epoch)
                save_train_state(main_config["model_name"], epoch=epoch, iteration=iteration, best_loss=best_loss,
                                 best_SSIM=best_SSIM, best_PSNR=best_PSNR)
                model_show_time(start_time, epoch, epoch_start_time)
                return False
        if epoch % train_config["model_save_frequency_epoch"] == 0:
            save_train_state(main_config["model_name"], epoch=epoch, iteration=iteration, best_loss=best_loss,
                             best_SSIM=best_SSIM, best_PSNR=best_PSNR)
            save_model(model, optimizer, train_config["model_save_dir"], main_config["model_name"], epoch + 1, 0)
            train_save_three_img(x[0, 2:, :, :], out_show, y[0, :, :, :],
                                     train_config["show_img_path"] + main_config["model_name"] + "/",
                                     epoch)
        model_show_time(start_time, epoch, epoch_start_time)
    return False
