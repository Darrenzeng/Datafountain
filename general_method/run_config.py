import torch

#等价于args
class RunConfig(object):
    def __init__(self):
        # self.device = torch.device('cuda')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.num_epochs = 1
        self.train_batch_size = 32
        self.test_batch_size = 32
        self.val_batch_size = 64
        self.seq_len = 100
        self.n_splits = 2#k的数量，默认20
