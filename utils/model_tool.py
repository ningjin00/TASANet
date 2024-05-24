import os
import time
import torch
from utils.path_tool import find_file_list, join_dir, jodge_dir
def get_epoch_and_model_load_path(model_save_dir):
    model_path_list = find_file_list(model_save_dir)
    if not len(model_path_list):
        return None
    model_path = model_path_list[-1].split("_")
    epoch = 0
    epoch = int(model_path[-3])
    model_load_path = model_save_dir + model_path_list[-1]
    return (epoch + 1, model_load_path)
def get_model_save_dir(model_save_dir, model_name):
    dir_path = join_dir(model_save_dir, model_name)
    jodge_dir(dir_path)
    return dir_path
def save_model(model, optimizer, model_save_dir, model_name, epoch, iteration):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    epoch_save = ""
    epoch = int(epoch)
    if epoch / 100 >= 1:
        epoch_save = str(epoch)
    elif epoch / 10 >= 1:
        epoch_save = "0" + str(epoch)
    else:
        epoch_save = "00" + str(epoch)
    if iteration / 100000 >= 1:
        iteration_save = str(iteration)
    elif iteration / 10000 >= 1:
        iteration_save = "0" + str(iteration)
    else:
        iteration_save = "00" + str(iteration)
    print("epoch:{},epoch_save:{}".format(epoch, epoch_save))
    model_save_path = os.path.join(model_save_dir,
                                   "{}_epoch_{}_iteration_{}.pth".format(model_name, epoch_save, iteration_save))
    torch.save(state, model_save_path)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("第{}轮训练结果已经保存".format(epoch))
    print("保存模型：{}".format(model_name))
    print("保存路径为：{}".format(model_save_path))
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
def model_show_time(start_time, epoch, epoch_start_time):
    end_time = time.strftime("%Y{y}%m{m}%d{d}%H{i}%M{j}%S{k}").format(y="年", m="月", d="日", i="时", j="分",
                                                                      k="秒")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(" 第{}轮开始时间： {}".format(epoch, start_time))
    print(" 第{}轮结束时间： {}".format(epoch, end_time))
    print(" 第{}轮总共用时 {} S".format(epoch, time.time() - epoch_start_time))
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
