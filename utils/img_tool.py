import time
import numpy as np
import cv2 as cv
import torch
from utils.path_tool import get_save_img_path, jodge_dir
def uint16to8(bands, lower_percent=0.001, higher_percent=99.999):
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[0]
    for i in range(n):
        a = 0  #
        b = 255  #
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)
        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out
def getRGBImg(r, g, b, img_size=256):
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img
def get_out_single_img(img_data, img_size=256, scale=10000.0):
    img_single = uint16to8((torch.squeeze(img_data).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)
    img_out_RGB = getRGBImg(img_single[:, :, 3], img_single[:, :, 2], img_single[:, :, 1], img_size)
    return img_out_RGB
def save_sigle_img(img_data, dir_path, img_name_list):
    img_data=torch.squeeze(img_data)
    jodge_dir(dir_path)
    save_path = dir_path+img_name_list[2][0][0]+img_name_list[2][1][0][:-4]+".jpg"
    img = get_out_single_img(img_data, img_size=256, scale=10000.0)
    cv.imwrite(save_path, img)
def get_out_sar_img(img_data, img_size=256, scale=100.0):
    img_single = uint16to8((torch.squeeze(img_data).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)
    img_out_RGB = getRGBImg(img_single[:, :, 1], np.zeros_like(img_single[:, :, 0]), img_single[:, :, 1], img_size)    # img_out_RGB = getRGBImg(img_single[:, :, 0],img_single[:, :, 1],  img_single[:, :, 0], img_size)
    img_out_RGB = np.dot(img_out_RGB[:,:,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    return img_out_RGB
def save_sar_img(sar_data, dir_path, img_name_list):
    sar_data = torch.squeeze(sar_data)
    jodge_dir(dir_path)
    save_path = dir_path + img_name_list[2][0][0] + img_name_list[2][1][0][:-4] + ".jpg"
    img = get_out_sar_img(sar_data, img_size=256, scale=100.0)
    cv.imwrite(save_path, img)
def save_three_img(s2_cloudy,out,s2,dir_path, img_name):
    img_size = 256
    jodge_dir(dir_path)
    save_path = dir_path + img_name + ".jpg"
    output_img = np.zeros((img_size, 3 * img_size, 3), dtype=np.uint8)
    s2_cloudy = get_out_single_img(s2_cloudy, img_size, 10000.0)
    out = get_out_single_img(out, img_size, 10000.0)
    s2 = get_out_single_img(s2, img_size, 10000.0)
    output_img[:, 0 * img_size:1 * img_size, :] = s2_cloudy
    output_img[:, 1 * img_size:2 * img_size, :] = out
    output_img[:, 2 * img_size:3 * img_size, :] = s2
    cv.imwrite(save_path, output_img)
def save_four_img(s2_cloudy,out1,out2,s2,dir_path, img_name_list):
    img_size = 256
    jodge_dir(dir_path)
    save_path = dir_path + img_name_list[2][0][0] + img_name_list[2][1][0][:-4] + ".jpg"
    output_img = np.zeros((img_size, 4 * img_size, 3), dtype=np.uint8)
    s2_cloudy = get_out_single_img(s2_cloudy, img_size, 10000.0)
    out1 = get_out_single_img(out1, img_size, 10000.0)
    out2 = get_out_single_img(out2, img_size, 10000.0)
    s2 = get_out_single_img(s2, img_size, 10000.0)
    output_img[:, 0 * img_size:1 * img_size, :] = s2_cloudy
    output_img[:, 1 * img_size:2 * img_size, :] = out1
    output_img[:, 2 * img_size:3 * img_size, :] = out2
    output_img[:, 3 * img_size:4 * img_size, :] = s2
    cv.imwrite(save_path, output_img)
def train_save_three_img(s2_cloudy,out,s2,dir_path,epoch):
    img_size = 256
    jodge_dir(dir_path)
    save_time = time.strftime("%Y{y}%m{m}%d{d}%H{i}%M{j}%S{k}").format(y="_", m="_", d="_", i="_", j="_", k="_")
    save_path = dir_path + str(epoch)+"_"+save_time + ".jpg"
    output_img = np.zeros((img_size, 3 * img_size, 3), dtype=np.uint8)
    s2_cloudy = get_out_single_img(s2_cloudy, img_size, 10000.0)
    out = get_out_single_img(out.detach(), img_size, 10000.0)
    s2 = get_out_single_img(s2, img_size, 10000.0)
    output_img[:, 0 * img_size:1 * img_size, :] = s2_cloudy
    output_img[:, 1 * img_size:2 * img_size, :] = out
    output_img[:, 2 * img_size:3 * img_size, :] = s2
    print("写入:"+save_path)
    cv.imwrite(save_path, output_img)