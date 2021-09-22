from model.han import HanConfig, HAN
from model.dpcnn import DPCNNConfig, DPCNN
from model.rcnn import TextRCNN, TextRCNNConfig
from model.rcnn_atten import TextRCNNAttn, TextRCNNAttnConfig
from run_config import RunConfig
# from model.textrcnn_attn import TextRCNNAttnConfig, TextRCNNAttn
# from model.textrcnn import TextRCNNConfig, TextRCNN

def gen_net(net_name):
    run_config = RunConfig()#加载训练参数，如epoch, batch_size

    if net_name == 'dpcnn':
        model_config = DPCNNConfig()#加载模型参数， 如dropout, lr
        return DPCNN(run_config, model_config), run_config, model_config #返回模型的同时，也返回参数
    elif net_name == 'rcnnatten':
        model_config = TextRCNNAttnConfig()
        return TextRCNNAttn(run_config, model_config), run_config, model_config
    elif net_name == 'han':
        model_config = HanConfig()
        return HAN(run_config, model_config), run_config, model_config

    elif net_name == 'rcnn':
        model_config = TextRCNNConfig()
        return TextRCNN(run_config, model_config), run_config, model_config
