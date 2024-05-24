from utils.train_tool import train
def run_train(model_name=""):
    is_ret = train(model_name=model_name)
    if is_ret:
        print("训练错误！请重新训练！")
    else:
        print("训练成功！")
if __name__ == '__main__':
    run_train(model_name="TASA")

