import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import tinycudann as tcnn
import torch.nn.functional as torch_F

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(
        self,
        d_feature,
        mode,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        head_channels=128,
        in_channels_dir_a=48,
        static_head_layers=2,
        weight_norm=True,
        multires_view=4,
        squeeze_out=True,
        encode_apperence=True,
    ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.encode_apperence = encode_apperence
        if self.encode_apperence:
            # -3 is to remove dir
            dims = (
                [d_in + head_channels - 3]
                + [d_hidden for _ in range(n_layers)]
                + [d_out]
            )
        else:
            dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += 0 if self.encode_apperence else (input_ch - 3)
            in_channels_dir_a += input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        if self.encode_apperence:
            # direction and appearance encoding layers
            static_encoding_od = OrderedDict(
                [
                    (
                        "static_linear_0",
                        nn.Linear(d_feature + in_channels_dir_a, head_channels),
                    ),
                    ("static_relu_0", nn.ReLU(True)),
                ]
            )
            for s_layer_i in range(1, static_head_layers):
                static_encoding_od[f"static_linear_{s_layer_i}"] = nn.Linear(
                    head_channels, head_channels
                )
                static_encoding_od[f"static_relu_{s_layer_i}"] = nn.ReLU(True)
            self.static_encoding = nn.Sequential(static_encoding_od)
            self.xyz_encoding_final = nn.Linear(d_feature, d_feature)

    def forward(self, points, normals, view_dirs, feature_vectors, input_dir_a=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.encode_apperence:
            # color prediction
            xyz_encoding_final = self.xyz_encoding_final(feature_vectors)  # (B, W)
            dir_encoding_input = torch.cat(
                [xyz_encoding_final, view_dirs, input_dir_a], 1
            )
            dir_encoding = self.static_encoding(dir_encoding_input)
        else:
            xyz_encoding_final = torch.zeros_like(feature_vectors)

        rendering_input = None

        if self.mode == "idr":
            if self.encode_apperence:
                rendering_input = torch.cat([points, normals, dir_encoding], dim=-1)
            else:
                rendering_input = torch.cat(
                    [points, view_dirs, normals, feature_vectors], dim=-1
                )
        elif self.mode == "no_view_dir":
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == "no_normal":
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x, xyz_encoding_final, view_dirs


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).to(x.device) * torch.exp(self.variance * 10.0)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=6,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,

    ):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
     
        self.input_dim = 131
        dims[0] = self.input_dim

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.multires = multires

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]   # 473 , 381

            elif l == 0:
                out_dim = dims[l + 1]

            elif l == 4:

                out_dim = dims[l]
            else:
               out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
          
            
            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                        
                        
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )

                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
            
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )

            setattr(self, "lin" + str(l), lin)
           
        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs, embed_fn_fine = True):
        inputs = inputs * self.scale
        x = inputs
    
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            
            if l == 0:
                x = lin(x)
            
            elif l == 3:
                x = lin(x) 
                        
            elif l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)
                x = lin(x)
                
            else:
                x = lin(x)

            if l < self.num_layers - 2:
                    x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.reshape(-1, 3)


class NeuconW(nn.Module):
    def __init__(
        self,
        sdfNet_config,
        colorNet_config,
        SNet_config,
        in_channels_a,
        encode_a,
        NELO_para
    ):

        super(NeuconW, self).__init__()
        self.sdfNet_config = sdfNet_config
        self.colorNet_config = colorNet_config
        self.SNet_config = SNet_config
        self.in_channels_a = in_channels_a
        self.encode_a = encode_a

        # xyz encoding layers + sdf layer
        self.sdf_net = SDFNetwork(**self.sdfNet_config)

        self.xyz_encoding_final = nn.Linear(512, 512)

        # Static deviation
        self.deviation_network = SingleVarianceNetwork(**self.SNet_config)

        # Static color
        self.color_net = RenderingNetwork(
            **self.colorNet_config,
            in_channels_dir_a=self.in_channels_a,
            encode_apperence=self.encode_a,
        )
        
        ###nelo
        self.NELO_para = NELO_para.model.object.sdf
        encoding_dim = self.build_encoding(self.NELO_para.encoding)
        input_dim = 3 + encoding_dim
        self.warm_up_end = self.NELO_para.warm_up_end #defauld 5000
        #self.build_mlp(cfg_sdf.mlp, input_dim=input_dim)

    def build_encoding(self, cfg_encoding):
        if self.NELO_para.encoding.type == "fourier":
            encoding_dim = 6 * cfg_encoding.levels
        elif self.NELO_para.encoding.type == "hashgrid":
            # Build the multi-resolution hash grid.
            l_min, l_max = self.NELO_para.encoding.hashgrid.min_logres, self.NELO_para.encoding.hashgrid.max_logres
            r_min, r_max = 2 ** l_min, 2 ** l_max
            num_levels = self.NELO_para.encoding.levels
            self.growth_rate = np.exp((np.log(r_max) - np.log(r_min)) / (num_levels - 1))
            config = dict(
                otype="HashGrid",
                n_levels=self.NELO_para.encoding.levels,
                n_features_per_level=self.NELO_para.encoding.hashgrid.dim,
                log2_hashmap_size=self.NELO_para.encoding.hashgrid.dict_size,
                base_resolution=2 ** self.NELO_para.encoding.hashgrid.min_logres,
                per_level_scale=self.growth_rate,
            )
            self.tcnn_encoding = tcnn.Encoding(3, config)
            self.resolutions = []
            for lv in range(0, num_levels):
                size = np.floor(r_min * self.growth_rate ** lv).astype(int) + 1
                self.resolutions.append(size)
            encoding_dim = self.NELO_para.encoding.hashgrid.dim * self.NELO_para.encoding.levels
        else:
            raise NotImplementedError("Unknown encoding type")
        return encoding_dim
    
    def encode(self, points_3D):
        if self.NELO_para.encoding.type == "fourier":
            # points_enc = nerf_util.positional_encoding(points_3D, num_freq_bases=self.cfg_sdf.encoding.levels)
            feat_dim = 6
        elif self.NELO_para.encoding.type == "hashgrid":
            # Tri-linear interpolate the corresponding embeddings from the dictionary.
            vol_min, vol_max = self.NELO_para.encoding.hashgrid.range
            points_3D_normalized = (points_3D - vol_min) / (vol_max - vol_min)  # Normalize to [0,1].
            tcnn_input = points_3D_normalized.view(-1, 3)
            tcnn_output = self.tcnn_encoding(tcnn_input)
            points_enc = tcnn_output.view(*points_3D_normalized.shape[:-1], tcnn_output.shape[-1])
            feat_dim = self.NELO_para.encoding.hashgrid.dim
        else:
            raise NotImplementedError("Unknown encoding type")
        # Coarse-to-fine.
        if self.NELO_para.encoding.coarse2fine.enabled:
            mask = self._get_coarse2fine_mask(points_enc, feat_dim=feat_dim)
            points_enc = points_enc * mask
        points_enc = torch.cat([points_3D, points_enc], dim=-1)  # [B,R,N,3+LD]
        return points_enc


    def sdf(self, input_xyz):
        # geometry prediction
        return self.sdf_net.sdf(input_xyz)  # (B, w+1)
        # return static_sdf[:, 1], static_sdf[:, 1:]

    def gradient(self, x):
        return self.sdf_net.gradient(x)
    #### nelo
    def set_active_levels(self, current_iter=None):
        anneal_levels = max((current_iter - self.warm_up_end) // self.NELO_para.encoding.coarse2fine.step, 1)
        if anneal_levels >= 12:
            self.anneal_levels = 12
            self.active_levels = 12
        else:
            self.anneal_levels = min(self.NELO_para.encoding.levels, anneal_levels)
            self.active_levels = max(self.NELO_para.encoding.coarse2fine.init_active_level, self.anneal_levels)
    def set_normal_epsilon(self, current_iter):
        if self.NELO_para.encoding.coarse2fine.enabled:
            epsilon_res = self.resolutions[self.anneal_levels - 1]
        else:
            epsilon_res = self.resolutions[-1]
        # epsilon_res = self.resolutions[-1]
        self.normal_eps = 1. / epsilon_res
    @torch.no_grad()
    def _get_coarse2fine_mask(self, points_enc, feat_dim):
        mask = torch.zeros_like(points_enc)
        mask[..., :(self.active_levels * feat_dim)] = 1
        # mask[..., :(16 * feat_dim)] = 1
        return mask
    
    def get_curvature_weight(self, current_iteration, init_weight):
        if current_iteration <= self.warm_up_end:
            self.curvature_weights = current_iteration / self.warm_up_end * init_weight
            decay_factor = self.growth_rate ** (self.anneal_levels - 1)
            self.curvature_weights  = init_weight / decay_factor
        else:
            decay_factor = self.growth_rate ** (self.anneal_levels - 1)
            # decay_factor = self.growth_rate ** (16 - 1)
            self.curvature_weights  = init_weight / decay_factor
    
    def compute_gradients(self, x, training=False, sdf=None):
        # Note: hessian is not fully hessian but diagonal elements
        if self.NELO_para.gradient.mode == "analytical":
            requires_grad = x.requires_grad
            with torch.enable_grad():
                # 1st-order gradient
                x.requires_grad_(True)
                sdf = self.sdf(x)
                gradient = torch.autograd.grad(sdf.sum(), x, create_graph=True)[0]
                # 2nd-order gradient (hessian)
                if training:
                    hessian = torch.autograd.grad(gradient.sum(), x, create_graph=True)[0]
                else:
                    hessian = None
                    gradient = gradient.detach()
            x.requires_grad_(requires_grad)
        elif self.NELO_para.gradient.mode == "numerical":
            if self.NELO_para.gradient.taps == 6:
                eps = self.normal_eps
                # 1st-order gradient
                eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
                eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]
                
                points_enc1 = self.encode(x + eps_x)
                sdf_x_pos = self.sdf(points_enc1)
                
                points_enc2 = self.encode(x - eps_x)
                sdf_x_neg = self.sdf(points_enc2)
                
                points_enc3 = self.encode(x + eps_y)
                sdf_y_pos = self.sdf(points_enc3)
                
                points_enc4 = self.encode(x - eps_y)
                sdf_y_neg = self.sdf(points_enc4)
                
                points_enc5 = self.encode(x + eps_z)
                sdf_z_pos = self.sdf(points_enc5)
                
                points_enc6 = self.encode(x - eps_z)
                sdf_z_neg = self.sdf(points_enc6)
                        
                gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
                gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
                gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
                gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)  # [...,3]
                # 2nd-order gradient (hessian)
                if training:
                    assert sdf is not None  # computed when feed-forwarding through the network
                    hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [...,1]
                    hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [...,3]
                else:
                    hessian = None
            elif self.NELO_para.gradient.taps == 4:
                eps = self.normal_eps / np.sqrt(3)
                k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
                k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
                k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
                k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]
                
                points_enc1 = self.encode(x + k1 * eps)
                sdf1 = self.sdf(points_enc1)
                
                points_enc2 = self.encode(x + k2 * eps)
                sdf2 = self.sdf(points_enc2)
                
                points_enc3 = self.encode(x + k3 * eps)
                sdf3 = self.sdf(points_enc3)
                
                points_enc4 = self.encode(x + k4 * eps)
                sdf4 = self.sdf(points_enc4)
               
                gradient = (k1*sdf1 + k2*sdf2 + k3*sdf3 + k4*sdf4) / (4.0 * eps)
                if training:
                    assert sdf is not None  # computed when feed-forwarding through the network
                    # the result of 4 taps is directly trace, but we assume they are individual components
                    # so we use the same signature as 6 taps
                    hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2   # [N,1]
                    hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0
                else:
                    hessian = None
            else:
                raise ValueError("Only support 4 or 6 taps.")
        return gradient, hessian
    
 
    def forward(self, x):
        device = x.device
        input_xyz, view_dirs, input_dir_a = torch.split(
            x, [3, 3, self.in_channels_a], dim=-1
        )

        n_rays, n_samples, _ = input_xyz.size()
        input_dir_a = input_dir_a.view(n_rays * n_samples, -1)

        # geometry prediction
        points_3D = input_xyz
        points_enc = self.encode(points_3D).reshape(-1, 131)  # [...,3+LD]
        ## nelo coarse2fine activate
      

        sdf_nn_output = self.sdf_net(points_enc, embed_fn_fine = False)  # [B, R, 131] 128 +3 = 131
        static_sdf = sdf_nn_output[:, :1]
        xyz_ = sdf_nn_output[:, 1:]

        # color prediction
        ####nelo
      
        points_3D = points_3D.reshape(-1, 3)
        gradients, hessians = self.compute_gradients(points_3D, training=True, sdf=static_sdf)
        normals = torch_F.normalize(gradients, dim=-1)  # [B,R,N,3]
    
        static_rgb, xyz_encoding_final, view_encoded = self.color_net(
            input_xyz.view(-1, 3),
            normals,
            view_dirs.view(-1, 3),
            xyz_,
            input_dir_a,
        )  # (B, 3)
        # sdf gradient
        static_deviation = self.deviation_network(torch.zeros([1, 3], device=device))[
            :, :1
        ].clamp(
            1e-6, 1e6
        )  # (B, 1)

        static_out = (
            static_rgb.view(n_rays, n_samples, 3),
            static_deviation,
            static_sdf.view(n_rays, n_samples),
            gradients.view(n_rays, n_samples, 3),  # static_gradient
            hessians.view(n_rays, n_samples, 3)
        )
        
        

        return static_out
