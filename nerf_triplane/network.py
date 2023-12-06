import types
#import immutabledict
from typing import Any, Mapping
import torch
import torch.nn as nn
import torch.nn.functional as F
#import jax
#import jax.numpy as jnp
#import gin

import sys, os

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
#from nerfies import warping
#from nerfies import model_utils
#from nerfies import modules

from .deformation import *
from encoding import get_encoder
from .renderer import NeRFRenderer


# class DeformationField_ori(nn.Module):
#     def __init__(self, dim_embed, dim_signal, hidden_size=64, n_blocks=7,skips=[4]):
#         super().__init__()
#         self.dim_embed = dim_embed
#         self.dim_signal = dim_signal
#         self.skips = skips

#         self.blocks_embed = nn.ModuleList([
#             nn.Linear(dim_embed + dim_signal, hidden_size)
#         ])
#         for i in range(n_blocks - 3):
#             self.blocks_embed.append(nn.Linear(hidden_size, hidden_size))
#         self.out_embed = nn.Linear(hidden_size, dim_embed)

#         self.blocks_signal = nn.ModuleList([
#             nn.Linear(dim_embed + dim_signal, hidden_size)
#         ])
#         for i in range(n_blocks - 3):
#             self.blocks_signal.append(nn.Linear(hidden_size, hidden_size))
#         self.out_signal = nn.Linear(hidden_size, dim_signal)

#         n_skips = sum([i in skips for i in range(n_blocks - 1)])
#         if n_skips > 0:
#             self.fc_embed_skips = nn.ModuleList(
#                 [nn.Linear(dim_embed, hidden_size) for i in range(n_skips)]
#             )
#             self.fc_signal_skips = nn.ModuleList(
#                 [nn.Linear(dim_signal, hidden_size) for i in range(n_skips)]
#             )

#         self.act_fn = F.relu

        

#     def forward(self, input):
#         embed = input[..., :self.dim_embed]
#         signal = input[..., -self.dim_signal:]

#         # embed
#         skip_idx = 0
#         net_embed = input
#         for idx, layer in enumerate(self.blocks_embed):
#             net_embed = self.act_fn(layer(net_embed))
#             if (idx + 1) in self.skips and (idx < len(self.blocks_embed) - 1):
#                 net_embed = net_embed + self.fc_embed_skips[skip_idx](embed)
#                 skip_idx += 1
#         embed_deformed = self.out_embed(net_embed)
        
#         # signal
#         skip_idx = 0
#         net_signal = input
#         for idx, layer in enumerate(self.blocks_signal):
#             net_signal = self.act_fn(layer(net_signal))
#             if (idx + 1) in self.skips and (idx < len(self.blocks_signal) - 1):
#                 net_signal = net_signal + self.fc_signal_skips[skip_idx](signal)
#                 skip_idx += 1
#         signal_deformed = self.out_signal(net_signal)

#         output = torch.cat((embed_deformed, signal_deformed), -1)
#         return output
    
# class ExpressionEnc(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.LeakyReLU(0.02, True),
#             nn.Linear(32, 32),
#         )
    
#     def forward(self, x):
#         output = self.encoder(x)
#         return output

# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y) 
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x


# def create_warp_field(model, num_batch_dims):  
#     return warping.create_warp_field(
#         field_type='se3',
#         num_freqs= 8,
#         num_embeddings= 8001,
#         num_features= 8,
#         num_batch_dims= num_batch_dims,
#         metadata_encoder_type= 'glo')
def create_warp_field(model, num_batch_dims):  
    return warping.create_warp_field(
        field_type=model.warp_field_type,
        num_freqs=model.num_warp_freqs,
        num_embeddings=model.num_warp_embeddings,
        num_features=model.num_warp_features,
        num_batch_dims=num_batch_dims,
        metadata_encoder_type=model.warp_metadata_encoder_type,
        **model.warp_kwargs)

class NeRFNetwork(NeRFRenderer):
    # warp_field_type: str = 'se3'
    # num_warp_freqs: int = 8
    # num_warp_embeddings: int = 8001
    # num_warp_features: int = 8
    # warp_metadata_encoder_type: str = 'glo'
    # warp_kwargs: Mapping[str, Any] = immutabledict.immutabledict()

    def __init__(self,
                 opt,
                 audio_dim = 32,
                 # torso net (hard coded for now)
                 ):
        super().__init__(opt)



        # audio embedding
        self.emb = self.opt.emb

        if 'esperanto' in self.opt.asr_model:
            self.audio_in_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_in_dim = 29
        elif 'hubert' in self.opt.asr_model:
            self.audio_in_dim = 1024
        else:
            self.audio_in_dim = 32
            
        if self.emb:
            self.embedding = nn.Embedding(self.audio_in_dim, self.audio_in_dim)

        # audio network
        # self.expnet = nn.Lineaer(256, 128)
        self.audio_dim = audio_dim
        self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)

        self.att = self.opt.att
        if self.att > 0:
            self.audio_att_net = AudioAttNet(self.audio_dim)


        #deformation part
        #self.warp_field = get_encoder('deform')
        #self.warp_field = warping.create_warp_field(field_type = 'se3',num_freqs = 8, num_embeddings = 8001, num_features = 8,  num_batch_dims = 2, metadata_encoder_type = 'glo')
        #self.warp_field = create_warp_field(self, num_batch_dims = 2)
        #self.point_encoder = model_utils.vmap_module(modules.SinusoidalEncoder, num_batch_dims=2)#(num_freq = 10)
        # DYNAMIC PART
        self.warp_field = SE3DeformationField(deformation_field_config = SE3DeformationFieldConfig(), )


        self.num_levels = 12
        self.level_dim = 1
        #gridencoder에서 hash 가져옴
        #output = encoder, encoder.output_dim (32)
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 64
        self.geo_feat_dim = 64
        self.eye_att_net = MLP(self.in_dim, 1, 16, 2)
        self.eye_dim = 1 if self.exp_eye else 0
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers)
        ## color network
        self.num_layers_color = 2
        self.hidden_dim_color = 64
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim + self.individual_dim, 3, self.hidden_dim_color, self.num_layers_color)

        self.unc_net = MLP(self.in_dim, 1, 32, 2)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 64, 2)

        self.testing = False

        if self.torso:
            # torso deform network
            self.register_parameter('anchor_points', 
                                    nn.Parameter(torch.tensor([[0.01, 0.01, 0.1, 1], [-0.1, -0.1, 0.1, 1], [0.1, -0.1, 0.1, 1]])))
            self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('frequency', input_dim=2, multires=8)
            # self.torso_deform_encoder, self.torso_deform_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=1, base_resolution=16, log2_hashmap_size=16, desired_resolution=512)
            self.anchor_encoder, self.anchor_in_dim = get_encoder('frequency', input_dim=6, multires=3)
            self.torso_deform_net = MLP(self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 2, 32, 3)

            # torso color network
            self.torso_encoder, self.torso_in_dim = get_encoder('tiledgrid', input_dim=2, num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=16, desired_resolution=2048)
            self.torso_net = MLP(self.torso_in_dim + self.torso_deform_in_dim + self.anchor_in_dim + self.individual_dim_torso, 4, 32, 3)


    def forward_torso(self, x, poses, c=None):
        # x: [N, 2] in [-1, 1]
        # head poses: [1, 4, 4]
        # c: [1, ind_dim], individual code

        # test: shrink x
        x = x * self.opt.torso_shrink

        # deformation-based
        wrapped_anchor = self.anchor_points[None, ...] @ poses.permute(0, 2, 1).inverse()
        wrapped_anchor = (wrapped_anchor[:, :, :2] / wrapped_anchor[:, :, 3, None] / wrapped_anchor[:, :, 2, None]).view(1, -1)
        # print(wrapped_anchor)
        # enc_pose = self.pose_encoder(poses)
        enc_anchor = self.anchor_encoder(wrapped_anchor)
        enc_x = self.torso_deform_encoder(x)

        if c is not None:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1), c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_x, enc_anchor.repeat(x.shape[0], 1)], dim=-1)

        dx = self.torso_deform_net(h)
        
        x = (x + dx).clamp(-1, 1)

        x = self.torso_encoder(x, bound=1)

        # h = torch.cat([x, h, enc_a.repeat(x.shape[0], 1)], dim=-1)
        h = torch.cat([x, h], dim=-1)

        h = self.torso_net(h)

        alpha = torch.sigmoid(h[..., :1])*(1 + 2*0.001) - 0.001
        color = torch.sigmoid(h[..., 1:])*(1 + 2*0.001) - 0.001

        return alpha, color, dx


    @staticmethod
    @torch.jit.script

#    def warp_point(self, points):

#        batch_size = 1
#        warp_extra = {
#            'alpha': 0.0,
#            'time_alpha': 0.0,
#    }
#        warp_metadata = torch.ones((batch_size, 1), dtype=torch.uint32)
#        warp_metadata = warp_metadata.unsqueeze(1).expand(*points.shape[:2], 1)
#        warp_out = self.warp_field(points, warp_metadata, warp_extra, use_warp_jacobian = False, metadata_encoded = False)
#        points = warp_out['warped_points']
#        points_embed = self.warp_field(points)

#        return points_embed
    

    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        if self.emb:
            a = self.embedding(a).transpose(-1, -2).contiguous() # [1/8, 29, 16]

        enc_a = self.audio_net(a) # [1/8, 64]

        if self.att > 0:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a

    
    def predict_uncertainty(self, unc_inp):
        if self.testing or not self.opt.unc_loss:
            unc = torch.zeros_like(unc_inp)
        else:
            unc = self.unc_net(unc_inp.detach())

        return unc


        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # enc_a: [1, aud_dim]
        # c: [1, ind_dim], individual code
        # e: [1, 1], eye feature
#        warp = self.warp_point(x)
        # warp_extra = {
        #      'alpha': 0.0,
        #      'time_alpha': 0.0,
        # }
        # batch_size = 1
        # use_warp_jacobian = False
        # metadata_encoded = False
        # warp_metadata = jnp.ones((batch_size, 1), jnp.uint32)
        # warp_metadata = jnp.broadcast_to(warp_metadata[:, jnp.newaxis, :], shape=(*x.shape[:2], 1))
        # warp_out = self.warp_field(x, warp_metadata, warp_extra, use_warp_jacobian, metadata_encoded)
        # print(warp_out)
        # x = warp_out['warped_points']
#        points = self.point_encoder(warp)
        #warp = self.warp_field['warped_points']
#        print(warp_out)
    def forward(self, x, d, enc_a, c, e=None):

        warp_out = self.warp_field(x)

        enc_x = self.encode_x(warp_out, bound=self.bound)

        # signal = input[42, -64:]

        # skip_idx = 0
        # net_signal = input
        # for idx, layer in enumerate(self.blocks_signal):
        #     net_signal = self.act_fn(layer(net_signal))
        #     if (idx + 1) in self.skips and (idx < len(self.blocks_signal) - 1):
        #         net_signal = net_signal + self.fc_signal_skips[skip_idx](signal)
        #         skip_idx += 1
        # signal_deformed = self.out_signal(net_signal)
        # expression = self.expnet(signal_deformed)

        # ExpNet = ExpressionEnc().to(device)


        sigma_result = self.density(x, enc_a, e, enc_x)
        sigma = sigma_result['sigma']
        geo_feat = sigma_result['geo_feat']
        aud_ch_att = sigma_result['ambient_aud']
        eye_att = sigma_result['ambient_eye']

        # color
        enc_d = self.encoder_dir(d)

        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)
                
        h_color = self.color_net(h)
        color = torch.sigmoid(h_color)*(1 + 2*0.001) - 0.001
        
        uncertainty = self.predict_uncertainty(enc_x)
        uncertainty = torch.log(1 + torch.exp(uncertainty))

        return sigma, color, aud_ch_att, eye_att, uncertainty[..., None]


    def density(self, x, enc_a, e=None, enc_x=None):
        # x: [N, 3], in [-bound, bound]
        if enc_x is None:
            enc_x = self.encode_x(x, bound=self.bound)

        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att

        if e is not None:
            # e = self.encoder_eye(e)
            eye_att = torch.sigmoid(self.eye_att_net(enc_x))
            e = e * eye_att
            # e = e.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, e], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w], dim=-1)

        h = self.sigma_net(h)

        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att,
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        # ONLY train torso
        if self.torso:
            params = [
                {'params': self.torso_encoder.parameters(), 'lr': lr},
                {'params': self.torso_deform_encoder.parameters(), 'lr': lr, 'weight_decay': wd},
                {'params': self.torso_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.torso_deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
                {'params': self.anchor_points, 'lr': lr_net, 'weight_decay': wd}
            ]

            if self.individual_dim_torso > 0:
                params.append({'params': self.individual_codes_torso, 'lr': lr_net, 'weight_decay': wd})

            return params

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},

            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            # {'params': self.encoder_xyz.parameters(), 'lr': lr},

            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
        ]
        if self.att > 0:
            params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.emb:
            params.append({'params': self.embedding.parameters(), 'lr': lr})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        if self.train_camera:
            params.append({'params': self.camera_dT, 'lr': 1e-5, 'weight_decay': 0})
            params.append({'params': self.camera_dR, 'lr': 1e-5, 'weight_decay': 0})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params