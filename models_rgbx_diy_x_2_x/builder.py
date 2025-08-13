import torch.nn as nn
import torch.nn.functional as F

from .utils.init_func import init_weight
from .encoders.segformer_iaff import mit_b2, mit_b3,mit_b4, mit_b5, mit_b0, mit_b1
class EncoderDecoder(nn.Module):
    def __init__(self, cfg_in=None,norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        from easydict import EasyDict as edict
        cfg = edict(cfg_in)
        cfg.backbone = 'mit_b2'  # Remember change the path below.
        # cfg.pretrained_model = './models_segformer/trained_models/segformer.b4.512x512.ade.160k.pth'
        # cfg.pretrained_model = './models_segformer/trained_models/segformer.b4.1024x1024.city.160k.pth'
        cfg.decoder = 'MLPDecoder'
        cfg.decoder_embed_dim = 512
        cfg.bn_eps = 1e-3
        cfg.bn_momentum = 0.1
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        # import backbone and decoder
        # if cfg.backbone == 'swin_s':
        #     print('Using backbone: Swin-Transformer-small')
        #     from .encoders.dual_swin import swin_s as backbone
        #     self.channels = [96, 192, 384, 768]
        #     self.backbone = backbone(norm_fuse=norm_layer)
        # elif cfg.backbone == 'swin_b':
        #     print('Using backbone: Swin-Transformer-Base')
        #     from .encoders.dual_swin import swin_b as backbone
        #     self.channels = [128, 256, 512, 1024]
        #     self.backbone = backbone(norm_fuse=norm_layer)
        if cfg.backbone == 'mit_b5':
            print('Using backbone: Segformer-B5')
            cfg.pretrained_model = './models_rgbx_diy_x_2_x/pretrained/segformers/mit_b5.pth'
            self.backbone = mit_b5(norm_fuse=norm_layer, m1=cfg.m1, m2=cfg.m2)
        elif cfg.backbone == 'mit_b4':
            print('Using backbone: Segformer-B4')
            self.backbone = mit_b4(norm_fuse=norm_layer, m1=cfg.m1, m2=cfg.m2)
            cfg.pretrained_model = './models_rgbx_diy_x_2_x/pretrained/segformers/mit_b4.pth'
        elif cfg.backbone == 'mit_b3':
            print('Using backbone: Segformer-B3')
            cfg.pretrained_model = './models_rgbx_diy_x_2_x/pretrained/segformers/mit_b3.pth'
            self.backbone = mit_b3(norm_fuse=norm_layer,m1=cfg.m1,m2=cfg.m2)
        elif cfg.backbone == 'mit_b2':
            print('Using backbone: Segformer-B2')
            cfg.pretrained_model = './models_rgbx_diy_x_2_x/pretrained/segformers/mit_b2.pth'
            self.backbone = mit_b2(norm_fuse=norm_layer, m1=cfg.m1, m2=cfg.m2)
        elif cfg.backbone == 'mit_b1':
            print('Using backbone: Segformer-B1')
            self.backbone = mit_b1(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            print('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            self.backbone = mit_b0(norm_fuse=norm_layer)
        elif cfg.backbone == 'mobileNetv2_2m':
            print('Using backbone: mobileNetv2_2m')
            self.backbone = mobileNetv2_2m(n_classes=10)
            self.channels = [24, 32, 160, 320]

        else:
            print('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        if cfg.decoder == 'MLPDecoder':
            print('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.n_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        elif cfg.decoder == 'HTLF':
            print('Using HTLF Decoder')
            from .decoders.HTLF import HTLFModule
            self.decode_head = HTLFModule(in_channels=self.channels, num_classes=cfg.n_classes,  embed_dim=cfg.decoder_embed_dim)

        elif cfg.decoder == 'UPernet':
            print('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.n_classes, norm_layer=norm_layer, channels=512)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.n_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'deeplabv3+':
            print('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.n_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.n_classes, norm_layer=norm_layer)

        else:
            print('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.n_classes, norm_layer=norm_layer)

        self.criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        if self.criterion:
            if cfg.backbone != 'mobileNetv2_2m':
                self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            print('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        print('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out = self.decode_head.forward(x)
        # print('out shape:', out.shape)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        # if label is not None:
        #     loss = self.criterion(out, label.long())
        #     if self.aux_head:
        #         loss += self.aux_rate * self.criterion(aux_fm, label.long())
        #     return loss
        return out


if __name__ == '__main__':
    import torch
    import json

    with open("E:\ws_python\MiLNet-main\models_rgbx_diy_x_2_x\mfnet_rgbx_diy.json", 'r') as fp:
        cfg = json.load(fp)
    cfg["m1"] = "BFAM"
    cfg["m2"] = "SDM2d"
    from ptflops import get_model_complexity_info

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncoderDecoder(cfg)


    # Create a wrapper Module that handles device consistency
    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, rgb_input):
            # Create dummy thermal input on same device as rgb_input
            thermal_input = torch.zeros(rgb_input.size(0), 3, *rgb_input.shape[2:],
                                        device=rgb_input.device)
            return self.model(rgb_input, thermal_input)


    # Use the wrapper for FLOPs calculation
    wrapper = ModelWrapper(model).to(device)

    # Calculate FLOPs
    macs, params = get_model_complexity_info(
        wrapper,
        (3, 480, 640),  # input shape
        as_strings=True,
        print_per_layer_stat=True,
        verbose=True
    )

    print('Flops:  ' + macs)
    print('Params: ' + params)

    # from toolbox import compute_speed, compute_speed_2
    #
    # compute_speed_2(model, input_size=(1, 3, 480, 640), iteration=500)
