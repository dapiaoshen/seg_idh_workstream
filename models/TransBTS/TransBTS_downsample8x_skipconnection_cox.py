import torch,math
import torch.nn as nn
from models.TransBTS.Transformer import TransformerModel
from models.TransBTS.PositionalEncoding import FixedPositionalEncoding,LearnedPositionalEncoding
from models.TransBTS.Unet_skipconnection import Unet

class Grade_netwoek(nn.Module):
    def __init__(self):
        super(Grade_netwoek, self).__init__()
        self.avg_pool_3d = nn.AvgPool3d(16, 1)
        self.max_pool_3d = nn.MaxPool3d(16, 1)
        self.Hidder_layer_1 = nn.Linear(1280, 512)
        self.Hidder_layer_2 = nn.Linear(512, 32)
        self.drop_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(32, 2)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x4_1, encoder_output):
        x = self.feature_fusion_layer(x4_1, encoder_output)
        x = self.drop_layer(x)
        x = self.Hidder_layer_1(x)
        x = self.Hidder_layer_2(x)
        y = self.classifier(x)
        return y

    def feature_fusion_layer(self, x4_1, encoder_output):
        x4_1_avg = self.avg_pool_3d(x4_1)
        x4_1_max = self.max_pool_3d(x4_1)

        encoder_avg = self.avg_pool_3d(encoder_output)
        encoder_avg_max = self.max_pool_3d(encoder_output)

        x4_1_avg = x4_1_avg.view(x4_1_avg.size(0), -1)
        x4_1_max = x4_1_max.view(x4_1_max.size(0), -1)

        encoder_avg = encoder_avg.view(encoder_avg.size(0), -1)
        encoder_avg_max = encoder_avg_max.view(encoder_avg_max.size(0), -1)

        return torch.cat([x4_1_avg, x4_1_max, encoder_avg, encoder_avg_max], dim=1)

class IDH_Feature_Attention(nn.Module):
    def __init__(self):
        super(IDH_Feature_Attention, self).__init__()
        self.attention_refinement_module1 = AttentionRefinementModule(128,128)
        self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        self.feature_fusion_module = FeatureFusionModule(640,640)
        self.avg_pool_3d = nn.AdaptiveAvgPool3d(1)
        self.max_pool_3d = nn.AdaptiveMaxPool3d(1)
        self.hidder_layer_1 = nn.Linear(1280, 512)
        self.hidder_layer_2 = nn.Linear(512, 32)
        self.drop_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(32, 2)


    def forward(self,x4_1,encoder_output):  # x4_1: torch.Size([1, 128, 16, 16, 16]) encoder_output: torch.Size([1, 512, 16, 16, 16])

        x4_1 = self.attention_refinement_module1(x4_1)
        encoder_output = self.attention_refinement_module2(encoder_output)
        feature_fusion = self.feature_fusion_module(x4_1,encoder_output)
        avg_feature = self.avg_pool_3d(feature_fusion)
        max_feature = self.max_pool_3d(feature_fusion)
        avg_pool_feature = avg_feature.view(avg_feature.size(0), -1)
        max_pool_feature = max_feature.view(max_feature.size(0), -1)
        pool_feature = torch.cat([avg_pool_feature,max_pool_feature],dim=1)
        x = self.drop_layer(pool_feature)
        x = self.hidder_layer_1(x)
        x = self.hidder_layer_2(x)
        return self.classifier(x)


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        # self.bn = nn.BatchNorm3d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))


    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv3d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))


    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        x = self.avgpool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x

class IDH_1p19q_network(nn.Module):
    def __init__(self):
        super(IDH_1p19q_network, self).__init__()
        self.avg_pool_3d_out = nn.AvgPool3d(128, 1)
        self.max_pool_3d_out = nn.MaxPool3d(128, 1)

        self.avg_pool_3d = nn.AvgPool3d(16, 1)
        self.max_pool_3d = nn.MaxPool3d(16, 1)
        self.Hidder_layer_1 = nn.Linear(1312, 512)  #1280  #1088
        self.Hidder_layer_2 = nn.Linear(512, 32) #512
        self.drop_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(32, 1)
        self.act = nn.Sigmoid()

    def forward(self,x4_1,encoder_output,out_feature):  #x4_1,encoder_output
        x = self.feature_fusion_layer(x4_1,encoder_output,out_feature)
        x = self.drop_layer(x)
        x = self.Hidder_layer_1(x)
        x = self.Hidder_layer_2(x)
        hazard = self.classifier(x)

        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return hazard

    def feature_fusion_layer(self, x4_1, encoder_output,out_feature):
        x4_1_avg = self.avg_pool_3d(x4_1)
        x4_1_max = self.max_pool_3d(x4_1)

        encoder_avg = self.avg_pool_3d(encoder_output)
        encoder_avg_max = self.max_pool_3d(encoder_output)

        out_avg = self.avg_pool_3d_out(out_feature)
        out_max = self.max_pool_3d_out(out_feature)

        x4_1_avg = x4_1_avg.view(x4_1_avg.size(0), -1)
        x4_1_max = x4_1_max.view(x4_1_max.size(0), -1)

        encoder_avg = encoder_avg.view(encoder_avg.size(0), -1)
        encoder_avg_max = encoder_avg_max.view(encoder_avg_max.size(0), -1)

        out_avg = out_avg.view(out_avg.size(0), -1)
        out_max = out_max.view(out_max.size(0), -1)

        return torch.cat([x4_1_avg, x4_1_max, encoder_avg, encoder_avg_max,out_avg,out_max], dim=1)


class Genomic_Clinical(nn.Module):
    def __init__(self):
        super(Genomic_Clinical, self).__init__()
        encoder1 = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Dropout(p=0.2))  # 640,768
        encoder2 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Dropout(p=0.2))  # 640,768
        encoder3 = nn.Sequential(nn.Linear(16, 9), nn.ReLU(), nn.Dropout(p=0.2))  # 640,768
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3)

    def forward(self,input):

        return self.encoder(input)


class Cox_network(nn.Module):
    def __init__(self):
        super(Cox_network, self).__init__()
        self.avg_pool_3d_3 = nn.AvgPool3d(32, 1)
        self.max_pool_3d_3 = nn.MaxPool3d(32, 1)

        self.avg_pool_3d = nn.AvgPool3d(16, 1)
        self.max_pool_3d = nn.MaxPool3d(16, 1)

        # encoder2 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.2)) #640,768
        # encoder3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.2))
        # encoder4 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(p=0.2))
        # self.encoder_c = nn.Sequential(nn.Linear(4, 4), nn.Dropout(p=0.2))
        # self.encoder = nn.Sequential(encoder2, encoder3, encoder4)

        encoder1 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(p=0.2)) #640,768
        encoder2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.2))
        encoder3 = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Dropout(p=0.2))

        encoder4 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.2))
        encoder5 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.2))
        encoder6 = nn.Sequential(nn.Linear(128, 16), nn.ReLU(), nn.Dropout(p=0.2))

        self.encoder1 = nn.Sequential(encoder1, encoder2, encoder3)
        self.encoder2 = nn.Sequential(encoder4, encoder5, encoder6)

        self.classifier = nn.Sequential(nn.Linear(32, 1)) #41

        self.genomic_module = Genomic_Clinical()
        self.fusion =BilinearFusion(skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=16, dim2=16, scale_dim1=1, scale_dim2=1, mmhid=32, dropout_rate=0.2)
    # def forward(self,x4_1,encoder_output,clinical):
    #     deep = self.feature_fusion_layer(x4_1, encoder_output)
    #     code = self.encoder(deep)
    #     clinical = self.encoder_c(clinical)
    #     out = self.classifier(torch.cat([code,clinical],dim=-1))
    #     return out

    def forward(self,x4_1,encoder_output):
        deep_1,deep_2 = self.feature_fusion_layer(x4_1, encoder_output)
        code_1 = self.encoder1(deep_1)
        code_2 = self.encoder2(deep_2)
        # genomic = self.genomic_module(genomic)
        fusion_feature = self.fusion(code_1,code_2)
        # genomic = self.encoder_c(genomic)
        # out = self.classifier(torch.cat([code,genomic],dim=-1))
        out = self.classifier(fusion_feature)
        return out


    def feature_fusion_layer(self,x4_1,encoder_output):

        x4_1_avg = self.avg_pool_3d(x4_1)
        # x4_1_max = self.max_pool_3d(x4_1)
        # y4_1_avg = self.avg_pool_3d(y4_1)

        # y3_1_avg = self.avg_pool_3d_3(y3_1)

        encoder_avg = self.avg_pool_3d(encoder_output)
        # encoder_avg_max = self.max_pool_3d(encoder_output)

        x4_1_avg = x4_1_avg.view(x4_1_avg.size(0), -1)
        # x4_1_max = x4_1_max.view(x4_1_max.size(0), -1)
        # y4_1_avg = y4_1_avg.view(y4_1_avg.size(0), -1)
        # y3_1_avg = y3_1_avg.view(y3_1_avg.size(0), -1)

        encoder_avg = encoder_avg.view(encoder_avg.size(0), -1)
        # encoder_avg_max = encoder_avg_max.view(encoder_avg_max.size(0), -1)

        # return torch.cat([x4_1_avg,x4_1_max,encoder_avg,encoder_avg_max], dim=1)
        return x4_1_avg,encoder_avg
        # return torch.cat([x4_1_avg, encoder_avg], dim=1)


class IDH_network(nn.Module):
    def __init__(self):
        super(IDH_network, self).__init__()
        self.avg_pool_3d_3 = nn.AvgPool3d(32, 1)
        self.max_pool_3d_3 = nn.MaxPool3d(32, 1)

        self.avg_pool_3d = nn.AvgPool3d(16, 1)
        self.max_pool_3d = nn.MaxPool3d(16, 1)
        self.Hidder_layer_1 = nn.Linear(640, 256)  #1280  #1088
        self.Hidder_layer_2 = nn.Linear(256, 32) #512
        self.drop_layer = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(32, 1)
        self.relu1 = nn.ReLU(True)
        self.relu2 = nn.ReLU(True)
        self.tanh = nn.Sigmoid()


    def forward(self,x4_1,encoder_output):  #x4_1,encoder_output
        x = self.feature_fusion_layer(x4_1,encoder_output)
        x = self.drop_layer(x)
        x = self.Hidder_layer_1(x)
        x = self.relu1(x)
        x = self.Hidder_layer_2(x)
        x = self.relu2(x)
        y = self.classifier(x)
        return y#self.tanh(y)


    def feature_fusion_layer(self,x4_1,encoder_output):

        x4_1_avg = self.avg_pool_3d(x4_1)
        #x4_1_max = self.max_pool_3d(x4_1)

        encoder_avg = self.avg_pool_3d(encoder_output)
        #encoder_avg_max = self.max_pool_3d(encoder_output)

        x4_1_avg = x4_1_avg.view(x4_1_avg.size(0), -1)
        #x4_1_max = x4_1_max.view(x4_1_max.size(0), -1)

        encoder_avg = encoder_avg.view(encoder_avg.size(0), -1)
        #encoder_avg_max = encoder_avg_max.view(encoder_avg_max.size(0), -1)

        #return torch.cat([x4_1_avg,x4_1_max,encoder_avg,encoder_avg_max], dim=1)

        return torch.cat([x4_1_avg, encoder_avg], dim=1)


class Decoder_modual(nn.Module):
    def __init__(self):
        super(Decoder_modual, self).__init__()

        self.embedding_dim = 512

        self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)

        self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)

        self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim // 4, out_channels=self.embedding_dim // 8)
        self.DeBlock4 = DeBlock(in_channels=self.embedding_dim // 8)

        self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim // 8, out_channels=self.embedding_dim // 16)
        self.DeBlock3 = DeBlock(in_channels=self.embedding_dim // 16) #32

        self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim // 16, out_channels=self.embedding_dim // 32)
        self.DeBlock2 = DeBlock(in_channels=self.embedding_dim // 32)  #16

        self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)

        self.Softmax = nn.Softmax(dim=1)

    def forward(self,x1_1, x2_1, x3_1, x8):
        return self.decode(x1_1, x2_1, x3_1, x8)

    def decode(self, x1_1, x2_1, x3_1, x8):

        x8 = self.Enblock8_1(x8)
        y4_1 = self.Enblock8_2(x8)

        y4 = self.DeUp4(y4_1, x3_1)  # (1, 64, 32, 32, 32)
        y3_1 = self.DeBlock4(y4)

        y3 = self.DeUp3(y3_1, x2_1)  # (1, 32, 64, 64, 64)
        y2_1 = self.DeBlock3(y3)

        y2 = self.DeUp2(y2_1, x1_1)  # (1, 16, 128, 128, 128)
        y1_1 = self.DeBlock2(y2)

        y = self.endconv(y1_1)      # (1, 4, 128, 128, 128)

        y = self.Softmax(y)

        return y4_1,y3_1,y2_1,y1_1,y


class TransformerBTS(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(TransformerBTS, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 3)
        self.seq_length = self.num_patches
        self.flatten_dim = 128 * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,

            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:

            self.conv_x = nn.Conv3d(
                128,
                self.embedding_dim,
                kernel_size=3,
                stride=1,
                padding=1
            )

        self.Unet = Unet(in_channels=4, base_channels=16, num_classes=4)
        self.bn = nn.BatchNorm3d(128)
        self.relu = nn.LeakyReLU(inplace=True)

    def encode(self, x):
        if self.conv_patch_representation:
            # combine embedding with conv patch distribution
            x1_1, x2_1, x3_1, x4_1 = self.Unet(x)
            x = self.bn(x4_1)
            x = self.relu(x)
            x = self.conv_x(x)

            x = x.permute(0, 2, 3, 4, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        else:
            x = self.Unet(x)
            x = self.bn(x)
            x = self.relu(x)
            x = (
                x.unfold(2, 2, 2)
                .unfold(3, 2, 2)
                .unfold(4, 2, 2)
                .contiguous()
            )
            x = x.view(x.size(0), x.size(1), -1, 8)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)

        x = self.pre_head_ln(x)

        intmd_layers = [1, 2, 3, 4]
        # assert intmd_layers is not None, "pass the intermediate layers for MLA"
        # encoder_outputs = {}
        #all_keys = []
        # for i in intmd_layers:
        #     val = str(2 * i - 1)
        #     _key = 'Z' + str(i)
        #     all_keys.append(_key)
        #     encoder_outputs[_key] = intmd_x[val]
        # all_keys.reverse()

        #x8 = encoder_outputs[all_keys[0]]
        # x8 = intmd_x['7']
        x8 = self._reshape_output(x)
        # print("x8 shape:", x8.shape)

        return x1_1, x2_1, x3_1,x4_1, x8

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x):

        x1_1, x2_1, x3_1,x4_1, encoder_output = self.encode(x)
        return x1_1, x2_1, x3_1,x4_1, encoder_output
        # return x1_1, x2_1, x3_1,x4_1, encoder_output
        # x1_1 shape: (N,16,128,128,128)
        # x2_1 shape: (N,32,64,64,64)
        # x3_1 shape: (N,64,32,32,32)
        # x4_1 shape: (N, 128, 16, 16, 16)
        # encoder_output  shape: (N,4096,512)
        # intmd_encoder_outputs (N,4096,512)*7
        # y4_1 shape (N, 128, 16, 16, 16)

        # print("intmd_encoder_outputs:",intmd_encoder_outputs)
        # print("intmd_encoder_outputs['0']:", intmd_encoder_outputs['0'])
        # y4_1,decoder_output = self.decode(
        #    x1_1, x2_1, x3_1, encoder_output, intmd_encoder_outputs, auxillary_output_layers)
        # IDH_out = self.IDH_classifier(x4_1, encoder_output,y4_1)
        # print("encoder_output:",encoder_output.shape)
        # if auxillary_output_layers is not None:
        #    auxillary_outputs = {}
        #    for i in auxillary_output_layers:
        #        val = str(2 * i - 1)
        #        _key = 'Z' + str(i)
        #        auxillary_outputs[_key] = intmd_encoder_outputs[val]
        #
        #    return IDH_out,decoder_output
        #
        # return IDH_out,decoder_output
    def get_last_shared_layer(self):
        return self.pre_head_ln

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 4, 1, 2, 3).contiguous()

        return x

    # def IDH_classifier(self,x4_1, encoder_output,y4_1 ):
    #     """
    #
    #     :param x4_1: (N, 128, 16, 16, 16)
    #     :param encoder_output: (N,4096,512)
    #     :param y4_1:(N, 128, 16, 16, 16)
    #     :return:
    #     """
    #
    #     x4_1 = self.avg_pool_3d_1(x4_1)
    #     x4_1 = x4_1.view(x4_1.size(0),-1)
    #     y4_1 = self.avg_pool_3d_2(y4_1)
    #     y4_1 = y4_1.view(y4_1.size(0), -1)
    #     encoder_output = encoder_output.mean(axis=1)
    #     feature_fusion = torch.cat([x4_1,encoder_output,y4_1],dim=1)
    #     x = self.Hidder_layer(feature_fusion)
    #     x = self.bn_layer(x)
    #     x= self.classifier(x)
    #     y = self.sigmoid(x)
    #     #print ("y shape:", y.shape)
    #     return y

class BTS(TransformerBTS):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=True,
        positional_encoding_type="learned",
    ):
        super(BTS, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes

        # self.Softmax = nn.Softmax(dim=1)
        #
        # self.Enblock8_1 = EnBlock1(in_channels=self.embedding_dim)
        # self.Enblock8_2 = EnBlock2(in_channels=self.embedding_dim // 4)
        #
        # self.DeUp4 = DeUp_Cat(in_channels=self.embedding_dim//4, out_channels=self.embedding_dim//8)
        # self.DeBlock4 = DeBlock(in_channels=self.embedding_dim//8)
        #
        # self.DeUp3 = DeUp_Cat(in_channels=self.embedding_dim//8, out_channels=self.embedding_dim//16)
        # self.DeBlock3 = DeBlock(in_channels=self.embedding_dim//16)
        #
        # self.DeUp2 = DeUp_Cat(in_channels=self.embedding_dim//16, out_channels=self.embedding_dim//32)
        # self.DeBlock2 = DeBlock(in_channels=self.embedding_dim//32)
        # self.endconv = nn.Conv3d(self.embedding_dim // 32, 4, kernel_size=1)

class EnBlock1(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock1, self).__init__()
        self.bn1 = nn.BatchNorm3d(512)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        return x1


class EnBlock2(nn.Module):
    def __init__(self, in_channels):
        super(EnBlock2, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(512 // 4)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm3d(512 // 4)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x
        return x1

class DeUp_Cat(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeUp_Cat, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)

    def forward(self, x, prev):
        x1 = self.conv1(x)
        y = self.conv2(x1)
        # y = y + prev
        y = torch.cat((prev, y), dim=1)
        y = self.conv3(y)
        return y

class DeBlock(nn.Module):
    def __init__(self, in_channels):
        super(DeBlock, self).__init__()

        self.bn1 = nn.BatchNorm3d(in_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(in_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x1 = self.bn1(x)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x1 = self.conv2(x1)
        x1 = x1 + x
        return x1




def TransBTS_1(dataset='brats', _conv_repr=True, _pe_type="learned"):

    if dataset.lower() == 'brats':
        img_dim = 128
        num_classes = 4

    num_channels = 4
    patch_dim = 8
    aux_layers = [1, 2, 3, 4]
    model = BTS(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=512,
        num_heads=8,
        num_layers=4, #4
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )
    return model

class BilinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1+dim2+2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out


############
# Omic Model
############
class MaxNet(nn.Module):
    def __init__(self, input_dim=80, omic_dim=32, dropout_rate=0.2, act=None, label_dim=1, init_max=True):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act = act

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False))

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max: init_max_weights(self)

        self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.encoder(x)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out
def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 4, 128, 128, 128), device=cuda0)
        model = TransBTS_1(dataset='brats', _conv_repr=True, _pe_type="learned")
        model.cuda()
        y = model(x)
        print(y.shape)
