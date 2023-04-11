import torch
import numpy as np
import torch
import numpy as np
from torch import nn

from .utils.model_utils import sigmoid_hm, _fill_fc_weights

class Bev_predictor(nn.Module):
    def __init__(self, num_class, in_channels):
        super(Bev_predictor, self).__init__()

        classes = num_class - 1
        self.regression_head_cfg = [['offset'], ['loc_z'], ['dim'], ['ori_cls', 'ori_offset']]
        self.regression_channel_cfg = [[2,], [1,], [3,], [8, 8]]
        
        self.head_conv = 64 # 256

        use_norm = "BN"
        if use_norm == 'BN': norm_func = nn.BatchNorm2d
        else: norm_func = nn.Identity

        self.bn_momentum = 0.1
        self.abn_activision = 'leaky_relu'

        # ###########################################
        # ###############  Cls Heads ################
        # ########################################### 

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
            norm_func(self.head_conv, momentum=self.bn_momentum), 
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_conv, classes, kernel_size=1, padding=1 // 2, bias=True),
        )
        
        self.class_head[-1].bias.data.fill_(- np.log(1 / 0.01 - 1))

        ###########################################
        ############  Regression Heads ############
        ########################################### 
        
        self.reg_features = nn.ModuleList()
        self.reg_heads = nn.ModuleList()

        # init regression heads
        for idx, regress_head_key in enumerate(self.regression_head_cfg):

            feat_layer = nn.Sequential(nn.Conv2d(in_channels, self.head_conv, kernel_size=3, padding=1, bias=False),
                                        norm_func(self.head_conv, momentum=self.bn_momentum), 
                                        nn.ReLU(inplace=True))
            self.reg_features.append(feat_layer)

            # init output head
            head_channels = self.regression_channel_cfg[idx]
            head_list = nn.ModuleList()
            for key_index, key in enumerate(regress_head_key):
                key_channel = head_channels[key_index]
                output_head = nn.Conv2d(self.head_conv, key_channel, kernel_size=1, padding=1 // 2, bias=True)

                _fill_fc_weights(output_head, 0)
                head_list.append(output_head)

            self.reg_heads.append(head_list)

    def forward(self, features):

        # output classification
        output_cls = self.class_head(features)

        output_regs = []
        # output regression
        for i, reg_feature_head in enumerate(self.reg_features):
            reg_feature = reg_feature_head(features)

            for j, reg_output_head in enumerate(self.reg_heads[i]):
                output_reg = reg_output_head(reg_feature)
                
                output_regs.append(output_reg)

        output_cls = sigmoid_hm(output_cls) # sigmoid & clamp
        output_regs = torch.cat(output_regs, dim=1)

        return output_cls, output_regs