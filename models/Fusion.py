import torch.nn as nn
import torch
class BilinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=768, dim2=768, scale_dim1=2, scale_dim2=2, mmhid=768,
                 dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1 // scale_dim1, dim2 // scale_dim2
        skip_dim = dim1 + dim2 + 2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(
            nn.Linear(dim1_og + dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(
            nn.Linear(dim1_og + dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1 + 1) * (dim2 + 1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            # z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            # o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2((1-nn.Sigmoid()(z1)) * h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)  # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out

class tf(nn.Module):
    def __init__(self):
        super(tf, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.ln1 = nn.LayerNorm(768)
        self.ln2 = nn.LayerNorm(768)
        self.tf=BilinearFusion()
    def forward(self, x, y):
        x = self.global_avg_pool(x)
        y=self.global_avg_pool(y)
        # 展平处理
        x = x.view(x.size(0), -1)  # 变为 (B, 768*2)
        y=y.view(y.size(0), -1)
        x=self.ln1(x)
        y=self.ln2(y)
        x=self.tf(x,y)
        return x


class IDH_MTTU_Predict(nn.Module):
    def __init__(self):
        super(IDH_MTTU_Predict, self).__init__()
        self.kl1 = nn.Linear(768, 512)
        self.kl2 = nn.Linear(512,64)
        self.kl3 = nn.Linear(64, 2)
        self.drop = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU()
        self.norm=nn.LayerNorm(768)

    def forward(self, x):
        x=self.norm(x)
        x = self.kl1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.kl2(x)
        x = self.relu(x)
        x = self.drop(x)
        x=self.kl3(x)
        return x