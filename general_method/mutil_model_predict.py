from helper.preprocess import test_pro
from tqdm import tqdm
import numpy as np
import torch
import directory
import os
import pandas as pd
import pickle
# import load_net
from load_net import gen_net

def load_model(net, weight_path, run_config):
    model = net.to(run_config.device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    return model

@torch.no_grad()
def predict(test_dataloader, model, run_config):
    """
    单一模型预测
    """
    model.eval()
    pred_list = []
    with torch.no_grad():
        for i, (texts) in enumerate(tqdm(test_dataloader)):
            output = model(texts.to(run_config.device))
            pred_list += output.sigmoid().detach().cpu().numpy().tolist()
        return np.array(pred_list)

def submit(test_preds_merge, test_set_path):
    test_pre_tensor = torch.tensor(test_preds_merge)
    test_pre = torch.max(test_pre_tensor, 1)[1]

    test_data = pd.read_csv(test_set_path)
    with open ('./model/id2label.pkl', 'rb') as f:
        id2label = pickle.load(f)

    pred_labels = [id2label[i] for i in test_pre]
    submit_file = '/home/zyf/Summer game2021/Datafountain/submits/merge_submit.csv'
    # submit_file = "./submit/submit_{}.csv".format(args.NAME)

    pd.DataFrame({"id": test_data['id'], "label": pred_labels}).to_csv(
        submit_file, index=False)
        
def main():
    #提交文件地址
    if not os.path.exists(directory.SUBMISSION_DIR):
        os.makedirs(directory.SUBMISSION_DIR)

    test_df = test_pro(directory.TEST_SET_B_PATH)  # 对B榜测试集进行预测

    all_weights = os.listdir(directory.MODEL_DIR) #获得训练好的所有的模型
    pred_list = 0
    
    for weight_rel_path in tqdm(all_weights):
        net_name = weight_rel_path.split('_')[0]#取单模型名
        net, run_config, model_config = gen_net(net_name)#加载原有模型
        weight_full_path = "%s/%s" % (directory.MODEL_DIR, weight_rel_path)
        model = load_model(net, weight_full_path, run_config)#加载训练好的模型框架，模型地址， 训练参数 

        single_pred = predict(test_df, model, run_config) #使用该单一模型进行预测
        pred_list += single_pred  #(6004, 35)
        
    res = (pred_list / len(all_weights)).tolist() #[3000, 17]
    submit(res, directory.TEST_SET_B_PATH)#生成提交文件


if __name__ == '__main__':
    main()