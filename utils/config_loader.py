import yaml
import argparse
def load_config(config_path):
    parser = argparse.ArgumentParser()
    config = {}
    try:
        args = parser.parse_args()
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError as error:
        print(error)
        print("配置文件路径出错，建议使用绝对路径!")
    if config["config_file_name"]:
        print("配置文件\"{}\"加载成功！".format(config["config_file_name"]))
        del config["config_file_name"]
    else:
        raise Exception("配置文件加载失败！")
    return config


if __name__ == '__main__':
    config_path = "../config/dataset_config.yaml"
    config = load_config(config_path)
    print(config)
