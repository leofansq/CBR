import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossViewEnhancement(nn.Module):
    def __init__(self, in_dim):
        super(CrossViewEnhancement, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)

        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.f_conv = nn.Conv2d(in_channels=in_dim*2-16, out_channels=in_dim-16, kernel_size=3, stride=1, padding=1, bias=True)
        
        self.pool_ratio = 2
        self.pool = nn.AvgPool2d(self.pool_ratio)

    def forward(self, front_x, bev_x):
        m_batchsize, C, height, width = front_x.size()
        proj_query = self.pool(self.query_conv(bev_x))  # B,C,H_b,W
        proj_key = self.pool(self.key_conv(front_x))  # B,C,H_f,W
        proj_key = torch.mean(proj_key, dim=-2, keepdim=True) # B,C,1,W
        proj_value = self.pool(self.value_conv(front_x))  # B,C,H_f,W

        for i in range(width//self.pool_ratio):
            energy = torch.bmm(proj_key[:,:,:,i].permute(0, 2, 1), proj_query[:,:,:,i]) # B,C(1),H_b
            # B,C(1),H_f(1),H_b * B,C,H_f(1),H_b(1) ---> B,C,H_f(1),H_b ---> B,C,H_b,W(1) ---> B,C,H_b,W
            if i == 0:
                T = ((energy/torch.sqrt(torch.sum(energy**2,dim=-1,keepdim=True))).unsqueeze(2) * torch.sum(proj_value[:,:,:,i], dim=-1, keepdim=True).unsqueeze(-1)).squeeze(2).unsqueeze(-1)
            else:
                T = torch.cat((T, ((energy/torch.sqrt(torch.sum(energy**2,dim=-1,keepdim=True))).unsqueeze(2) * torch.sum(proj_value[:,:,:,i], dim=-1, keepdim=True).unsqueeze(-1)).squeeze(2).unsqueeze(-1)), dim=-1)

        T = F.interpolate(T, scale_factor=self.pool_ratio, mode="nearest")

        bev_enhance = torch.cat((bev_x[:,16:,:,:], T), dim=1)
        bev_enhance = self.f_conv(bev_enhance)
        output = torch.cat((bev_x[:,:16,:,:], bev_enhance), dim=1)

        return output
