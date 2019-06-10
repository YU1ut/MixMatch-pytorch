import math
import torch.nn as nn
from models.TEBlock import TEBlock,TEBlock1,GlobalAveragePooling
from models.Attention import Self_Attn

class TE_Module(nn.Module):
    """
    reference: TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED
            LEARNING
    https://arxiv.org/pdf/1610.02242.pdf
    """
    def __init__(self, _num_inchannels=3, _num_stages=3, _use_avg_on_conv3=True,run_type=0):
        super(TE_Module, self).__init__()
        self.num_inchannels = _num_inchannels
        self.num_stages = _num_stages
        self.use_avg_on_conv3 = _use_avg_on_conv3

        assert (self.num_stages >= 3)
        nChannels = 128
        nChannels1 = 256
        nChannels2 = 512
        count_stage=0

        additional_stage=0
        if run_type != 0:
            additional_stage+=1
        blocks = [nn.Sequential() for i in range(self.num_stages+additional_stage)]
        # 1st block,kernel size 3,3,3
        blocks[count_stage].add_module('Block1_ConvB1', TEBlock(self.num_inchannels, nChannels, 3))
        blocks[count_stage].add_module('Block1_ConvB2', TEBlock(nChannels, nChannels, 3))
        blocks[count_stage].add_module('Block1_ConvB3', TEBlock(nChannels, nChannels, 3))
        blocks[count_stage].add_module('Block1_MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        blocks[count_stage].add_module('Block1_Dropout',nn.Dropout(p=0.5, inplace=True))
        count_stage+=1
        blocks[count_stage].add_module('Block2_ConvB1', TEBlock(nChannels, nChannels1, 3))
        blocks[count_stage].add_module('Block2_ConvB2', TEBlock(nChannels1, nChannels1, 3))
        blocks[count_stage].add_module('Block2_ConvB3', TEBlock(nChannels1, nChannels1, 3))
        blocks[count_stage].add_module('Block2_MaxPool', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        blocks[count_stage].add_module('Block2_Dropout', nn.Dropout(p=0.5, inplace=True))
        count_stage+=1
        if run_type==1 or run_type==2 or run_type==3 or run_type==4 or run_type==8:
            blocks[count_stage].add_module('Attention', Self_Attn(nChannels1, 'relu'))
            count_stage+=1
        blocks[count_stage].add_module('Block3_ConvB1', TEBlock1(nChannels1, nChannels2, 3))
        blocks[count_stage].add_module('Block3_ConvB2', TEBlock1(nChannels2, nChannels1, 1))
        blocks[count_stage].add_module('Block3_ConvB3', TEBlock1(nChannels1, nChannels, 1))
        #add final average pooling
        blocks.append(nn.Sequential())
        blocks[-1].add_module('GlobalAveragePooling', GlobalAveragePooling())

        self._feature_blocks = nn.ModuleList(blocks)
        if run_type==0:
            self.all_feat_names =['conv' + str(s + 1) for s in range(self.num_stages)] + ['classifier', ]
        elif run_type==1 or run_type==2 or run_type==3 or run_type==4 or run_type==8:
            self.all_feat_names = ['conv1'] + ['conv2'] + ['Attention'] + ['conv' + str(s + 1) for s in
                                                                           range(2, self.num_stages)] + ['classifier', ]
        # example ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10', 'classifier']
        assert (len(self.all_feat_names) == len(self._feature_blocks))
        self.weight_initialization()
    def _parse_out_keys_arg(self, out_feat_keys):
        """
        :param out_feat_keys:
        :return:
        the lasy layer index from out_feat_keys
        """

        # By default return the features of the last layer / module.
        out_feat_keys = [self.all_feat_names[-1], ] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
            raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
            if key not in self.all_feat_names:
                raise ValueError(
                    'Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
            elif key in out_feat_keys[:f]:
                raise ValueError('Duplicate output feature key: {0}.'.format(key))

        # Find the highest output feature in `out_feat_keys
        max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

        return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.
        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.
        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)
        go_attention_flag = False
        feat = x
        for f in range(max_out_feat + 1):
            key = self.all_feat_names[f]
            if key == 'Attention':
                go_attention_flag = True
                feat, attention = self._feature_blocks[f](feat)
            else:
                feat = self._feature_blocks[f](feat)
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats) == 1 else out_feats
        if go_attention_flag:
            return out_feats, attention
        else:
            return out_feats, None


    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.requires_grad:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):  # init with not transform on batchnorm
                if m.weight.requires_grad:
                    m.weight.data.fill_(1)
                if m.bias.requires_grad:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias.requires_grad:
                    m.bias.data.zero_()