import os

class Config(object):
    def __init__(self, args):
        self.args = args

        # train hyper parameter
        self.device = args.device
        self.multi_gpu = args.multi_gpu
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.clause_max_len = args.clause_max_len
        self.attention_head = args.attention_head
        self.hidden_dropout_prob = 0.1
        self.weight_decay = 0.01
        self.adam_epsilon = 1e-6
        self.lr = args.lr
        self.bert_lr = args.bert_lr
        self.warmup_proportion = args.warmup_proportion
        self.seed = args.seed
        self.max_norm = args.max_norm
        self.init_mode = args.init_mode
        # dataset

        # path and name
        self.root = "./"
        self.data_path = args.data_path
        self.model_dir = args.model_path
        self.checkpoint_dir = os.path.join(self.root, "checkpoint", args.model_name, "BERTLR_{:e}_LR_{:e}_BS_{}".format(self.bert_lr, self.lr, self.batch_size))
        self.result_dir = args.result_path
        self.log_dir = os.path.join(self.root, "log")
        self.train_prefix = args.train_prefix
        self.dev_prefix = args.dev_prefix
        self.test_prefix = args.test_prefix
        self.model_name = args.model_name
        self.log_save_name = "LOG_{}_BERTLR_{:e}_LR_{:e}_BS_{}".format(args.model_name, self.bert_lr, self.lr, self.batch_size)

        # log-diff setting
        self.period = args.period
        self.test_epoch = args.test_epoch

        # debug
        self.debug = args.debug
        if self.debug:
            self.dev_prefix = self.train_prefix
            self.test_prefix = self.train_prefix

