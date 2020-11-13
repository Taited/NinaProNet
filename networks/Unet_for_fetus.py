import torch
import torch.nn as nn

from dataloaders.GR_modules import conv_block, UpCat, UpCatconv
from dataloaders.GR_modules import UnetGridGatingSignal3, GridAttentionBlock2D, UnetDsv3
from dataloaders.GR_modules import SE_Conv_Block_fetus, SE_Conv_Block_nomax_fetus, CBAM_conv_block


# U-net for medical image segmentation
class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, feature_scale=4):
        super(Unet, self).__init__()
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]    # 64, 128, 256, 512, 1024 //66, 132, 264, 528, 1056(3.7M)
        filters = [int(x / self.feature_scale) for x in filters]

        self.conv1 = conv_block(in_ch, filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = conv_block(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = conv_block(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
        # self.upconv4 = conv_block(filters[4], filters[3])
        self.up3 = UpCatconv(filters[3], filters[2])
        # self.upconv3 = conv_block(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        # self.upconv2 = conv_block(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])
        # self.upconv1 = conv_block(filters[1], filters[0])

        self.final = nn.Sequential(nn.Conv2d(filters[0], out_ch, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)
        center = self.center(pool4)

        up_4 = self.up4(conv4, center)
        # up_4 = self.upconv4(up_4)
        up_3 = self.up3(conv3, up_4)
        # up_3 = self.upconv3(up_3)
        up_2 = self.up2(conv2, up_3)
        # up_2 = self.upconv2(up_2)
        up_1 = self.up1(conv1, up_2)
        # up_1 = self.upconv1(up_1)

        out = self.final(up_1)

        return out


#   Spatial Attention U-net for binary image segmentation
class Spatial_Atten_Unet(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Spatial_Atten_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2],
                                                    inter_channels=filters[1])
        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2])
        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[4],
                                                    inter_channels=filters[3])

        # upsampling
        self.up_concat4 = UpCatconv(filters[4], filters[3], self.is_deconv, drop_out=True)
        self.up_concat3 = UpCatconv(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UpCatconv(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UpCatconv(filters[1], filters[0], self.is_deconv)

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=(224, 300))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=(224, 300))
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=(224, 300))
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        # self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Sigmoid())
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        # dsv4 = self.dsv4(up4)
        # dsv3 = self.dsv3(up3)
        # dsv2 = self.dsv2(up2)
        # dsv1 = self.dsv1(up1)
        # out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        out = self.final(up1)

        return out


#   Channel Attention U-net for binary image segmentation
class Channel_Atten_enc_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Channel_Atten_enc_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = SE_Conv_Block_fetus(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = SE_Conv_Block_fetus(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = SE_Conv_Block_fetus(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = SE_Conv_Block_fetus(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4 = conv_block(filters[4], filters[3], drop_out=True)
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3 = conv_block(filters[3], filters[2])
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2 = conv_block(filters[2], filters[1])
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1 = conv_block(filters[1], filters[0])

        # # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
        # # final conv (without any concat)
        # self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Sigmoid())
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction(encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        # gating = self.gating(center)

        # Upscaling Part (Decoder)
        up_4 = self.up4(conv4, center)
        up_4 = self.upconv4(up_4)
        up_3 = self.up3(conv3, up_4)
        up_3 = self.upconv3(up_3)
        up_2 = self.up2(conv2, up_3)
        up_2 = self.upconv2(up_2)
        up_1 = self.up1(conv1, up_2)
        up_1 = self.upconv1(up_1)

        # # Deep Supervision
        # dsv4 = self.dsv4(up_4)
        # dsv3 = self.dsv3(up_3)
        # dsv2 = self.dsv2(up_2)
        # dsv1 = self.dsv1(up_1)
        # out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        out = self.final(up_1)

        return out


#   Channel Attention U-net for binary image segmentation
class Channel_Atten_dec_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Channel_Atten_dec_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4 = SE_Conv_Block_fetus(filters[4], filters[3], drop_out=True)
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3 = SE_Conv_Block_fetus(filters[3], filters[2])
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2 = SE_Conv_Block_fetus(filters[2], filters[1])
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1 = SE_Conv_Block_fetus(filters[1], filters[0])

        # # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
        # # final conv (without any concat)
        # self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Sigmoid())
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction(encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        # gating = self.gating(center)

        # Upscaling Part (Decoder)
        up_4 = self.up4(conv4, center)
        up_4 = self.upconv4(up_4)
        up_3 = self.up3(conv3, up_4)
        up_3 = self.upconv3(up_3)
        up_2 = self.up2(conv2, up_3)
        up_2 = self.upconv2(up_2)
        up_1 = self.up1(conv1, up_2)
        up_1 = self.upconv1(up_1)

        # # Deep Supervision
        # dsv4 = self.dsv4(up_4)
        # dsv3 = self.dsv3(up_3)
        # dsv2 = self.dsv2(up_2)
        # dsv1 = self.dsv1(up_1)
        # out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        out = self.final(up_1)

        return out


#   Channel Attention U-net for binary image segmentation
class Channel_Atten_encdec_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Channel_Atten_encdec_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = SE_Conv_Block_nomax_fetus(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = SE_Conv_Block_nomax_fetus(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = SE_Conv_Block_nomax_fetus(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = SE_Conv_Block_nomax_fetus(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4 = SE_Conv_Block_nomax_fetus(filters[4], filters[3], drop_out=True)
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3 = SE_Conv_Block_nomax_fetus(filters[3], filters[2])
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2 = SE_Conv_Block_nomax_fetus(filters[2], filters[1])
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1 = SE_Conv_Block_nomax_fetus(filters[1], filters[0])

        # # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
        # # final conv (without any concat)
        # self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Sigmoid())
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction(encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        # gating = self.gating(center)

        # Upscaling Part (Decoder)
        up_4 = self.up4(conv4, center)
        up_4 = self.upconv4(up_4)
        up_3 = self.up3(conv3, up_4)
        up_3 = self.upconv3(up_3)
        up_2 = self.up2(conv2, up_3)
        up_2 = self.upconv2(up_2)
        up_1 = self.up1(conv1, up_2)
        up_1 = self.upconv1(up_1)

        # # Deep Supervision
        # dsv4 = self.dsv4(up_4)
        # dsv3 = self.dsv3(up_3)
        # dsv2 = self.dsv2(up_2)
        # dsv1 = self.dsv1(up_1)
        # out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        out = self.final(up_1)

        return out


#   Scale Attention U-net for binary image segmentation
# class Scale_Atten_Unet_fetus(nn.Module):
#     def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
#         super(Scale_Atten_Unet_fetus, self).__init__()
#         self.is_deconv = is_deconv
#         self.in_channels = in_channels
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale
#         self.padding_list = [2, 4, 6, 8]
#         self.dilation_list = [2, 4, 6, 8]
#         self.num_branches = 4
#
#         filters = [64, 128, 256, 512, 1024]
#         filters = [int(x / self.feature_scale) for x in filters]
#
#         # downsampling
#         self.conv1 = conv_block(self.in_channels, filters[0])
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
#
#         self.conv2 = conv_block(filters[0], filters[1])
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
#
#         self.conv3 = conv_block(filters[1], filters[2])
#         self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
#
#         self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))
#
#         self.center = conv_block(filters[3], filters[4], drop_out=True)
#
#         # 逆卷积，也可以使用上采样
#         self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
#         self.up3 = UpCatconv(filters[3], filters[2])
#         self.up2 = UpCatconv(filters[2], filters[1])
#         self.up1 = UpCatconv(filters[1], filters[0])
#
#         # deep supervision
#         self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(256, 256))
#         self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(256, 256))
#         self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
#         self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)
#
#         self.scale_att = CBAM_conv_block(in_size=n_classes * 16, out_size=n_classes * 4)
#         # final convolution
#         self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())
#
#     def forward(self, inputs):
#         # Feature Extraction (encoder)
#         conv1 = self.conv1(inputs)
#         maxpool1 = self.maxpool1(conv1)
#         conv2 = self.conv2(maxpool1)
#         maxpool2 = self.maxpool2(conv2)
#         conv3 = self.conv3(maxpool2)
#         maxpool3 = self.maxpool3(conv3)
#         conv4 = self.conv4(maxpool3)
#         maxpool4 = self.maxpool4(conv4)
#
#         center = self.center(maxpool4)
#
#         # Upscaling Part (Decoder)
#         up_4 = self.up4(conv4, center)
#         up_3 = self.up3(conv3, up_4)
#         up_2 = self.up2(conv2, up_3)
#         up_1 = self.up1(conv1, up_2)
#
#         # Deep Supervision
#         dsv4 = self.dsv4(up_4)
#         dsv3 = self.dsv3(up_3)
#         dsv2 = self.dsv2(up_2)
#         dsv1 = self.dsv1(up_1)
#         dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
#         # print('dsv_cat:', dsv_cat.size())
#         out = self.scale_att(dsv_cat)
#         out = self.final(out)
#
#         return out
class Scale_Atten_2_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Scale_Atten_2_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.padding_list = [2, 4, 6, 8]
        self.dilation_list = [2, 4, 6, 8]
        self.num_branches = 4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
        self.up3 = UpCatconv(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(224, 300))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(224, 300))
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)

        self.scale_att = CBAM_conv_block(in_size=n_classes * 8, out_size=n_classes * 4)
        # final convolution
        self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction (encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # Upscaling Part (Decoder)
        up_4 = self.up4(conv4, center)
        up_3 = self.up3(conv3, up_4)
        up_2 = self.up2(conv2, up_3)
        up_1 = self.up1(conv1, up_2)

        # Deep Supervision
        # dsv4 = self.dsv4(up_4)
        # dsv3 = self.dsv3(up_3)
        dsv2 = self.dsv2(up_2)
        dsv1 = self.dsv1(up_1)
        dsv_cat = torch.cat([dsv1, dsv2], dim=1)
        # print('dsv_cat:', dsv_cat.size())
        out = self.scale_att(dsv_cat)
        out = self.final(out)

        return out


class Scale_Atten_3_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Scale_Atten_3_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.padding_list = [2, 4, 6, 8]
        self.dilation_list = [2, 4, 6, 8]
        self.num_branches = 4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
        self.up3 = UpCatconv(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(224, 300))
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)

        self.scale_att = CBAM_conv_block(in_size=n_classes * 12, out_size=n_classes * 4)
        # final convolution
        self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction (encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # Upscaling Part (Decoder)
        up_4 = self.up4(conv4, center)
        up_3 = self.up3(conv3, up_4)
        up_2 = self.up2(conv2, up_3)
        up_1 = self.up1(conv1, up_2)

        # Deep Supervision
        # dsv4 = self.dsv4(up_4)
        dsv3 = self.dsv3(up_3)
        dsv2 = self.dsv2(up_2)
        dsv1 = self.dsv1(up_1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3], dim=1)
        # print('dsv_cat:', dsv_cat.size())
        out = self.scale_att(dsv_cat)
        out = self.final(out)

        return out


class Scale_Atten_4_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Scale_Atten_4_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.padding_list = [2, 4, 6, 8]
        self.dilation_list = [2, 4, 6, 8]
        self.num_branches = 4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
        self.up3 = UpCatconv(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)

        self.scale_att = CBAM_conv_block(in_size=n_classes * 16, out_size=n_classes * 4)
        # final convolution
        self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction (encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # Upscaling Part (Decoder)
        up_4 = self.up4(conv4, center)
        up_3 = self.up3(conv3, up_4)
        up_2 = self.up2(conv2, up_3)
        up_1 = self.up1(conv1, up_2)

        # Deep Supervision
        dsv4 = self.dsv4(up_4)
        dsv3 = self.dsv3(up_3)
        dsv2 = self.dsv2(up_2)
        dsv1 = self.dsv1(up_1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        # print('dsv_cat:', dsv_cat.size())
        out = self.scale_att(dsv_cat)
        out = self.final(out)

        return out


class Scale_Atten_C_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Scale_Atten_C_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.padding_list = [2, 4, 6, 8]
        self.dilation_list = [2, 4, 6, 8]
        self.num_branches = 4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # 逆卷积，也可以使用上采样
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
        self.up3 = UpCatconv(filters[3], filters[2])
        self.up2 = UpCatconv(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])

        # deep supervision
        self.cetr = UnetDsv3(in_size=filters[4], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)

        self.scale_att = CBAM_conv_block(in_size=n_classes * 20, out_size=n_classes * 4)
        # final convolution
        self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction (encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # Upscaling Part (Decoder)
        up_4 = self.up4(conv4, center)
        up_3 = self.up3(conv3, up_4)
        up_2 = self.up2(conv2, up_3)
        up_1 = self.up1(conv1, up_2)

        # Deep Supervision
        cetr = self.cetr(center)
        dsv4 = self.dsv4(up_4)
        dsv3 = self.dsv3(up_3)
        dsv2 = self.dsv2(up_2)
        dsv1 = self.dsv1(up_1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4, cetr], dim=1)
        # print('dsv_cat:', dsv_cat.size())
        out = self.scale_att(dsv_cat)
        out = self.final(out)

        return out


#   CBAM Attention U-net for binary image segmentation
class CBAM_Atten_Unet(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(CBAM_Atten_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = CBAM_conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = CBAM_conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = CBAM_conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = CBAM_conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        # self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
        #                                     is_batchnorm=self.is_batchnorm)

        # attention blocks
        # self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2],
        #                                             inter_channels=filters[1])
        # self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
        #                                             inter_channels=filters[2])
        # self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[4],
        #                                             inter_channels=filters[3])

        # upsampling
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4 = CBAM_conv_block(filters[4], filters[3], drop_out=True)
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3 = CBAM_conv_block(filters[3], filters[2])
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2 = CBAM_conv_block(filters[2], filters[1])
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1 = CBAM_conv_block(filters[1], filters[0])

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=(224, 300))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=(224, 300))
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=(224, 300))
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        # self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Sigmoid())
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        # gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        # g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up4(conv4, center)
        up4 = self.upconv4(up4)
        # g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up3(conv3, up4)
        up3 = self.upconv3(up3)
        # g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up2(conv2, up3)
        up2 = self.upconv2(up2)
        up1 = self.up1(conv1, up2)
        up1 = self.upconv1(up1)

        # Deep Supervision
        # dsv4 = self.dsv4(up4)
        # dsv3 = self.dsv3(up3)
        # dsv2 = self.dsv2(up2)
        # dsv1 = self.dsv1(up1)
        # out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        out = self.final(up1)

        return out


#   Channel with Spatial Attention U-net for binary image segmentation
class Channel_Spatial_Atten_Unet(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Channel_Spatial_Atten_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = SE_Conv_Block_nomax_fetus(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = SE_Conv_Block_nomax_fetus(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = SE_Conv_Block_nomax_fetus(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = SE_Conv_Block_nomax_fetus(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2],
                                                    inter_channels=filters[1])
        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2])
        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[4],
                                                    inter_channels=filters[3])

        # upsampling
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4 = SE_Conv_Block_nomax_fetus(filters[4], filters[3], drop_out=True)
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3 = SE_Conv_Block_nomax_fetus(filters[3], filters[2])
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2 = SE_Conv_Block_nomax_fetus(filters[2], filters[1])
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1 = SE_Conv_Block_nomax_fetus(filters[1], filters[0])

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        # self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Sigmoid())
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up4(g_conv4, center)
        up4 = self.upconv4(up4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up3(g_conv3, up4)
        up3 = self.upconv3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up2(g_conv2, up3)
        up2 = self.upconv2(up2)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)
        up1 = self.up1(conv1, up2)
        up1 = self.upconv1(up1)

        # Deep Supervision
        # dsv4 = self.dsv4(up4)
        # dsv3 = self.dsv3(up3)
        # dsv2 = self.dsv2(up2)
        # dsv1 = self.dsv1(up1)
        # out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        out = self.final(up1)

        return out


#   Spatial with Scale Attention U-net for binary image segmentation
class Spatial_Scale_Atten_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Spatial_Scale_Atten_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.padding_list = [2, 4, 6, 8]
        self.dilation_list = [2, 4, 6, 8]
        self.num_branches = 4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2], drop_out=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)
        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2],
                                                    inter_channels=filters[1])
        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2])
        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[4],
                                                    inter_channels=filters[3])

        # 逆卷积，也可以使用上采样
        self.up4 = UpCatconv(filters[4], filters[3], drop_out=True)
        self.up3 = UpCatconv(filters[3], filters[2], drop_out=True)
        self.up2 = UpCatconv(filters[2], filters[1])
        self.up1 = UpCatconv(filters[1], filters[0])

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)

        self.scale_att = CBAM_conv_block(in_size=n_classes * 16, out_size=n_classes * 4)
        # final convolution
        self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction (encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up2(g_conv2, up3)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)
        up1 = self.up1(conv1, up2)

        # Deep Supervision
        # out = self.final(up1)
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)
        out = self.final(out)

        return out


#   Channel with Scale Attention U-net for binary image segmentation
class Channel_Scale_Atten_Unet_fetus(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Channel_Scale_Atten_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.padding_list = [2, 4, 6, 8]
        self.dilation_list = [2, 4, 6, 8]
        self.num_branches = 4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = SE_Conv_Block_nomax_fetus(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = SE_Conv_Block_nomax_fetus(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = SE_Conv_Block_nomax_fetus(filters[1], filters[2], drop_out=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = SE_Conv_Block_nomax_fetus(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)

        # upsampling
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4_1 = SE_Conv_Block_nomax_fetus(filters[4], filters[3], drop_out=True)
        # self.upconv4_2 = SC_Conv_Block(filters[3], filters[4], filters[3], self.padding_list,
        #                                self.dilation_list, self.num_branches)
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3_1 = SE_Conv_Block_nomax_fetus(filters[3], filters[2], drop_out=True)
        # self.upconv3_2 = SC_Conv_Block(filters[2], filters[3], filters[2], self.padding_list,
        #                                self.dilation_list, self.num_branches)
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2_1 = SE_Conv_Block_nomax_fetus(filters[2], filters[1])
        # self.upconv2_2 = SC_Conv_Block(filters[1], filters[2], filters[1], self.padding_list,
        #                                self.dilation_list, self.num_branches)
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1_1 = SE_Conv_Block_nomax_fetus(filters[1], filters[0])
        # self.upconv1_2 = SC_Conv_Block(filters[0], filters[1], filters[0], self.padding_list,
        #                                self.dilation_list, self.num_branches)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)

        self.scale_att = CBAM_conv_block(in_size=n_classes * 16, out_size=n_classes * 4)
        # final conv (without any concat)
        self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        # gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        up4 = self.up4(conv4, center)
        up4 = self.upconv4_1(up4)
        up3 = self.up3(conv3, up4)
        up3 = self.upconv3_1(up3)
        up2 = self.up2(conv2, up3)
        up2 = self.upconv2_1(up2)
        up1 = self.up1(conv1, up2)
        up1 = self.upconv1_1(up1)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)
        out = self.final(out)

        return out


#   Spatial with CBAM Attention U-net for binary image segmentation
class Spatial_CBAM_Atten_Unet(nn.Module):
    def __init__(self, in_channels=4, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Spatial_CBAM_Atten_Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = CBAM_conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = CBAM_conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = CBAM_conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = CBAM_conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2],
                                                    inter_channels=filters[1])
        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2])
        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[4],
                                                    inter_channels=filters[3])

        # upsampling
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4 = CBAM_conv_block(filters[4], filters[3])
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3 = CBAM_conv_block(filters[3], filters[2])
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2 = CBAM_conv_block(filters[2], filters[1])
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1 = CBAM_conv_block(filters[1], filters[0])

        # deep supervision
        # self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=(256, 256))
        # self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        # self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())
        self.final = nn.Sequential(nn.Conv2d(filters[0], n_classes, kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up4(g_conv4, center)
        up4 = self.upconv4(up4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up3(g_conv3, up4)
        up3 = self.upconv3(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up2(g_conv2, up3)
        up2 = self.upconv2(up2)
        up1 = self.up1(conv1, up2)
        up1 = self.upconv1(up1)

        # Deep Supervision
        # dsv4 = self.dsv4(up4)
        # dsv3 = self.dsv3(up3)
        # dsv2 = self.dsv2(up2)
        # dsv1 = self.dsv1(up1)
        # out = self.final(torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1))
        out = self.final(up1)

        return out


#   Spatial, Channel with Scale Attention U-net for binary image segmentation
class Spatial_Channel_Scale_atten_Unet_fetus(nn.Module):
    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Spatial_Channel_Scale_atten_Unet_fetus, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.padding_list = [2, 4, 6, 8]
        self.dilation_list = [2, 4, 6, 8]
        self.num_branches = 4

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = conv_block(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = conv_block(filters[0], filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = conv_block(filters[1], filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv4 = conv_block(filters[2], filters[3], drop_out=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = conv_block(filters[3], filters[4], drop_out=True)
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1),
                                            is_batchnorm=self.is_batchnorm)
        # attention blocks
        # self.attentionblock1 = GridAttentionBlock2D(in_channels=filters[0], gating_channels=filters[1],
        #                                             inter_channels=filters[0])
        self.attentionblock2 = GridAttentionBlock2D(in_channels=filters[1], gating_channels=filters[2],
                                                    inter_channels=filters[1])
        self.attentionblock3 = GridAttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                    inter_channels=filters[2])
        self.attentionblock4 = GridAttentionBlock2D(in_channels=filters[3], gating_channels=filters[4],
                                                    inter_channels=filters[3])

        # 逆卷积，也可以使用上采样
        self.up4 = UpCat(filters[4], filters[3])
        self.upconv4_1 = SE_Conv_Block_nomax_fetus(filters[4], filters[3], drop_out=True)
        # self.upconv4_2 = SC_Conv_Block(filters[3], filters[4], filters[3], self.padding_list,
        #                                self.dilation_list, self.num_branches, drop_out=True)
        self.up3 = UpCat(filters[3], filters[2])
        self.upconv3_1 = SE_Conv_Block_nomax_fetus(filters[3], filters[2])
        # self.upconv3_2 = SC_Conv_Block(filters[2], filters[3], filters[2], self.padding_list,
        #                                self.dilation_list, self.num_branches)
        self.up2 = UpCat(filters[2], filters[1])
        self.upconv2_1 = SE_Conv_Block_nomax_fetus(filters[2], filters[1])
        # self.upconv2_2 = SC_Conv_Block(filters[1], filters[2], filters[1], self.padding_list,
        #                                self.dilation_list, self.num_branches)
        self.up1 = UpCat(filters[1], filters[0])
        self.upconv1_1 = SE_Conv_Block_nomax_fetus(filters[1], filters[0])
        # self.upconv1_2 = SC_Conv_Block(filters[0], filters[1], filters[0], self.padding_list,
        #                                self.dilation_list, self.num_branches)

        # deep supervision
        # self.dsvc = UnetDsv3(in_size=filters[4], out_size=n_classes * 4, scale_factor=(224, 300))
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes * 4, scale_factor=(256, 256))
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes * 4, kernel_size=1)

        # Scale attention
        self.scale_att = CBAM_conv_block(in_size=n_classes * 16, out_size=n_classes * 4)
        # final convolution
        self.final = nn.Sequential(nn.Conv2d(n_classes * 4, n_classes, kernel_size=1), nn.Softmax2d())

    def forward(self, inputs):
        # Feature Extraction (encoder)
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up4(g_conv4, center)
        up4 = self.upconv4_1(up4)
        # up4 = self.upconv4_2(up4)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up3(g_conv3, up4)
        up3 = self.upconv3_1(up3)
        # up3 = self.upconv3_2(up3)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up2(g_conv2, up3)
        up2 = self.upconv2_1(up2)
        # up2 = self.upconv2_2(up2)
        # g_conv1, att1 = self.attentionblock1(conv1, up2)
        up1 = self.up1(conv1, up2)
        up1 = self.upconv1_1(up1)
        # up1 = self.upconv1_2(up1)

        # Deep Supervision
        # out = self.final(up1)
        # dsvc = self.dsvc(center)
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        out = self.scale_att(dsv_cat)
        out = self.final(out)

        return out