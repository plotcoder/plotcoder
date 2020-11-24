import os

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    """
    The class for recording the training process.
    """

    def __init__(self, args):
        self.log_interval = args.log_interval
        self.log_dir = os.path.join(args.out_dir, 'logs')
        self.log_name = os.path.join(self.log_dir, 'log.csv')
        self.tb_dir = os.path.join(args.out_dir, 'tb')
        self.writer = SummaryWriter(self.tb_dir)
        self.best_eval_acc = 0
        self.records = []

    def write_summary(self, summary):
        print("global-step: %(global_step)d, train-acc: %(train_acc).3f, train-loss: %(train_loss).3f, eval-label-acc: %(eval_label_acc).3f, eval-data-acc: %(eval_data_acc).3f, eval-acc: %(eval_acc).3f, eval-loss: %(eval_loss).3f" % summary)
        self.records.append(summary)
        df = pd.DataFrame(self.records)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        df.to_csv(self.log_name, index=False)
        for k, v in summary.items():
            if k == 'global_step':
                continue
            else:
                k = '/'.join(k.split('_'))
            self.writer.add_scalar(k, v, summary['global_step'])
        self.best_eval_acc = max(self.best_eval_acc, summary['eval_acc'])
