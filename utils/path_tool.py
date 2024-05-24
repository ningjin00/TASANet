import os
from glob import glob
def add_reverse_rake_bar(file_dir):
    if file_dir[-1] != '/': file_dir = file_dir + "/"
    return file_dir
def join_dir_list(dir_path, file_list):
    path = add_reverse_rake_bar(dir_path)
    for file in file_list:
        path = os.path.join(path, file)
        path = add_reverse_rake_bar(path)
    return path
def join_dir(dir_path, file_dir):
    path = add_reverse_rake_bar(dir_path)
    path = os.path.join(path, file_dir)
    path = add_reverse_rake_bar(path)
    return path
def find_file_list(dir_path):
    file_list = [os.path.basename(s) for s in glob(os.path.join(dir_path, "*"))]
    return file_list
def get_save_img_path(img_dir, path_list):
    img_dir = add_reverse_rake_bar(img_dir)
    path_list=[list(path_list[0])[0],str(int(path_list[1])),list(path_list[2])[0],list(path_list[3])[0]]
    img_name = path_list[2][:-1]+path_list[3][:-4]+".jpg"
    img_dir = join_dir_list(img_dir,[path_list[0], path_list[1]])
    jodge_dir(img_dir)
    img_dir = add_reverse_rake_bar(img_dir)
    return img_dir+img_name
def jodge_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("创建目录：{}".format(dir_path))
    else:
        pass
