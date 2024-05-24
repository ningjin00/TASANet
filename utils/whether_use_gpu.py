import torch
def whether_use_gpu(config_use_gpu):
    is_gpu = False
    cuda_is_available = torch.cuda.is_available()
    if config_use_gpu == False and cuda_is_available == True:
        print("您有一块NVIDIA GPU:{} 可以使用,请在config/train_config.yaml里面修改“train_use_gpu”为True来使用它".format(
            torch.cuda.get_device_name()))
    elif config_use_gpu == True and cuda_is_available == True:
        print("您正在使用NVIDIA {} 进行训练".format(torch.cuda.get_device_name()))
        is_gpu = True
    elif config_use_gpu == False and cuda_is_available == False:
        print("未选择使用GPU进行训练！同时GPU不可用！")
    else:
        raise Exception("选择使用GPU，但GPU不可用！请在配置文件yaml中更改use_gpu是否可用的参数")
    return is_gpu
