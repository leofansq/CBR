import torch.nn as nn

class DecoupleViewProjection(nn.Module):
    def __init__(self, in_dim):
        super(DecoupleViewProjection, self).__init__()
        self.bev_transform_module = TransformModule(dim=in_dim)
        self.fv_transform_module = TransformModule(dim=in_dim)
        # self.bn = nn.BatchNorm2d(512)

    def forward(self, x):
        B, C, H, W = x.view([-1, int(x.size()[1])] + list(x.size()[2:])).size()
        # x = self.bn(x)
        bev_features = self.bev_transform_module(x)
        fv_features = self.fv_transform_module(x)
        return bev_features, fv_features


class TransformModule(nn.Module):
    def __init__(self, dim=25):
        super(TransformModule, self).__init__()
        self.dim = dim
        self.mat_list = nn.ModuleList()
        # self.bn = nn.BatchNorm2d(512)
        self.fc_transform = nn.Sequential(
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU(),
            nn.Linear(dim * dim, dim * dim),
            nn.ReLU()
        )

    def forward(self, x):
        # shape x: B, C, H, W
        # x = self.bn(x)
        x = x.view(list(x.size()[:2]) + [self.dim * self.dim, ])
        view_comb = self.fc_transform(x)
        view_comb = view_comb.view(list(view_comb.size()[:2]) + [self.dim, self.dim])
        return view_comb