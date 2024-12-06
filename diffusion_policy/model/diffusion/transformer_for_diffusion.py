from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from torchvision.ops import MLP
from model.ntrack_decoder import NTrackTransformerDecoder
logger = logging.getLogger(__name__)

# we depart from the diffusion policy
# the context has timesteps from 1...horizon
# however, the future timesteps do NOT have hand poses and do not have RGB
# the goal is to predict where we want the hand to go

def convert_boolean_mask_to_additive_mask(mask):
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
                 input_dim: int,
                 output_dim: int, # this is the action dim of a single hand
                 horizon: int,
                 n_obs_steps: int = None,
                 cond_dim: int = 0,
                 future_cond_dim: int = 0,
                 n_layer: int = 12,
                 n_head: int = 12,
                 n_emb: int = 768,
                 p_drop_emb: float = 0.1,
                 p_drop_attn: float = 0.1,
                 causal_attn: bool=False,
                 time_as_cond: bool=True,
                 n_cond_layers: int = 0,
                 use_hand_collapse_input_emb: bool=False,
                 use_flatten_hands_2x: bool=True,
                 use_2x_horizon: bool=False,
                 unconditional=True,
                 conditioning_to_use=["rgb", "proprioceptive", "future_camera_pose"],
                 use_mse_loss=False,
                 mask_history=True
                 ) -> None:
        super().__init__()

        self.output_dim = output_dim
        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        self.n_obs_steps = n_obs_steps
        self.unconditional = unconditional
        self.conditioning_to_use = conditioning_to_use
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            # assert time_as_cond
            # T_cond += n_obs_steps
            T_cond += horizon

        # input embedding stem
        if use_hand_collapse_input_emb:
            # process two hands, with a mask 0 1 dictating whether a hand is included
            # as well as a positional embedding for L/R
            # into a fixed sized vector representing 0, 1, 2 hands
            # make it a small MLP instead of just one linear layer
            assert (input_dim / 2) % 1 == 0
            # self.input_emb = nn.Linear(int(input_dim / 2), n_emb)
            self.input_emb = MLP(
                in_channels=int(input_dim / 2) + n_emb,
                hidden_channels=[512, n_emb],
                activation_layer=torch.nn.Mish
            )

            self.hand_chilarity_pos_emb = nn.Parameter(torch.zeros(2, n_emb))
        elif use_flatten_hands_2x:
            # flattens both hands into one track
            # embeds each hand with same thing + position
            self.input_emb = nn.Linear(int(input_dim / 2), n_emb)
            self.hand_chilarity_pos_emb = nn.Parameter(torch.zeros(2, n_emb))
        else:
            # need to take in the hand pos embedding AND the hand joints themselves
            raise NotImplementedError
            self.input_emb = nn.Linear(input_dim, n_emb)

        self.use_hand_collapse_input_emb = use_hand_collapse_input_emb

        # flatten_hands_2x
        # flattens the left and right hands into one track
        # TODO: use a L/R embedding + a shared time embedding. for now, just doubles the horizon
        # architecturally:
        # encoder only architecture. uses cross attention to produce two fixed size embedding (for past RGB and future camera poses) with cross attention,
        # then concatenates that global vector to each action and does self attention over the actions (encoder only)
        self.use_flatten_hands_2x = use_flatten_hands_2x
        self.use_2x_horizon = use_2x_horizon

        if self.use_flatten_hands_2x and self.use_2x_horizon:
            self.pos_emb = nn.Parameter(torch.zeros(1, 2*T, n_emb))
        else:
            self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.n_head = n_head

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)

        # used to scale the time emb so we can sum it
        self.time_emb_proj = nn.Linear(n_emb, n_emb)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            # rgb embedding
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

            if not time_as_cond:
                self.cond_obs_combiner = nn.Linear(n_emb*3, n_emb)

        if self.use_flatten_hands_2x and "proprioceptive" in self.conditioning_to_use:
            self.proprioceptive_emb = nn.Linear(int(output_dim//2), n_emb)

            if not time_as_cond:
                self.proprioceptive_combiner = nn.Linear(n_emb * 3, n_emb)

        if "future_camera_pose" in conditioning_to_use:
            self.future_cond_obs_emb = nn.Linear(future_cond_dim, n_emb)

            if not time_as_cond:
                self.future_cond_obs_combiner = nn.Linear(n_emb * 3, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        if self.unconditional:
            self.decoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )
        elif not self.unconditional and T_cond > 0:
            # only have timesteps here
            # the chilarity is determined by the hand embedding
            # self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, horizon, n_emb))

            if n_cond_layers > 0:
                if "rgb" in self.conditioning_to_use:
                    self.encoder = nn.TransformerEncoder(
                        encoder_layer=encoder_layer,
                        num_layers=n_cond_layers
                    )

                if "future_camera_pose" in self.conditioning_to_use:
                    self.future_encoder = nn.TransformerEncoder(
                        encoder_layer=encoder_layer,
                        num_layers=n_cond_layers
                    )

                if self.use_flatten_hands_2x:
                    if "proprioceptive" in self.conditioning_to_use:
                        self.proprioceptive_encoder = nn.TransformerEncoder(
                            encoder_layer=encoder_layer,
                            num_layers=n_cond_layers
                        )
            else:
                print("ln119 if you do this, you need to rewrite the code to concat the timestep, otherwise it will get erased")
                raise NotImplementedError
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )

                self.future_encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )

            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            if self.use_flatten_hands_2x:
                self.decoder = NTrackTransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=n_layer,
                    num_tracks=len(conditioning_to_use)
                    # device=self.device
                ).to(self.device)
            else:
                self.decoder = nn.TransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=n_layer
                )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        if self.unconditional:
            self.decoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            # additive mask applies the mask to the score before going into the softmax

            # 1s are present
            # 0s are not

            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = convert_boolean_mask_to_additive_mask(mask)

            # register buffer places the tensor on the same device as the model
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                # T_cond is n_obs_steps + 1
                # T is horizon

                # the size of t and s is horizon x n_obs_steps, a 2D grid
                # if we index into this 2D grid, t gives the x value and s gives the y value
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )

                # this mask is a causal mask for cross attention
                # recall that the target / decoder length is the horizon
                # and the source / context length is the n_obs_steps
                # this mask is horizon X n_obs_steps
                # it says when decoding, each action timestep can only look at the history that's already happened
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)

        if self.use_flatten_hands_2x:
            # produce one hand per element
            self.head = nn.Linear(n_emb, int(output_dim//2))
        else:
            self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon

        # condition on diffusion timestep
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

        self.use_mse_loss = use_mse_loss
        if use_mse_loss:
            self.query_param = torch.nn.Parameter(torch.zeros(1, self.horizon*2, n_emb))
        self.mask_history = mask_history

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.ReLU,
            NTrackTransformerDecoder)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

            # to_instantiate = ["cond_obs_emb",
            #                   "proprioceptive_emb",
            #                   "future_cond_obs_emb",
            #                   ]

            # for str_ in to_instantiate:
            #     if getattr(module, str_) is not None:
            #         torch.nn.init.normal_(getattr(module, str_), mean=0.0, std=0.02)

            if hasattr(module, "query_param") and module.query_param is not None:
                torch.nn.init.normal_(module.query_param, mean=0.0, std=0.02)

            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)

            if hasattr(module, "hand_chilarity_pos_emb") and module.hand_chilarity_pos_emb is not None:
                torch.nn.init.normal_(module.hand_chilarity_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        if self.use_mse_loss:
            no_decay.add('query_param')
        if self.use_hand_collapse_input_emb or self.use_flatten_hands_2x:
            no_decay.add("hand_chilarity_pos_emb")


        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        future_cond: Optional[torch.Tensor]=None,
        hand_present_boolean: Optional[torch.Tensor]=None,
        action=None,
        **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        sample: noisy action
        action: clean action (proprio)
        """
        # 1. time

        # for now, do this
        assert hand_present_boolean is not None

        effective_batch_size = hand_present_boolean.shape[0]

        # these are diffusion timesteps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        diffusion_timestep_embedding = self.time_emb_proj(self.time_emb(timesteps).unsqueeze(1))
        # (B,1,n_emb)

        # process input
        # these are the actions

        if self.use_hand_collapse_input_emb:
            assert not self.use_flatten_hands_2x
            assert not self.use_mse_loss
            # B, horizon, 84 -> B, horizon, 42 and B, horizon, 42
            left_sample, right_sample = sample.chunk(2, dim=-1)

            left_input_emb = self.input_emb(torch.cat([self.hand_chilarity_pos_emb[0].unsqueeze(0).unsqueeze(0).expand(left_sample.shape[0], left_sample.shape[1], -1), left_sample], axis=-1))

            right_input_emb = self.input_emb(torch.cat([self.hand_chilarity_pos_emb[1].unsqueeze(0).unsqueeze(0).expand(right_sample.shape[0], right_sample.shape[1], -1), right_sample], axis=-1))
            # apply hand masking
            left_input_emb = left_input_emb * hand_present_boolean[:, :, 0].unsqueeze(-1)
            right_input_emb = right_input_emb * hand_present_boolean[:, :, 1].unsqueeze(-1)
            input_emb = left_input_emb + right_input_emb
        elif self.use_flatten_hands_2x:
            if self.use_mse_loss:
                input_emb = self.query_param.expand(effective_batch_size, -1, -1)
            else:
                left_sample, right_sample = sample.chunk(2, dim=-1)
                flattened_sample = torch.cat([left_sample, right_sample], dim=1)

                # effective_batch, 2*horizon, n_emb
                input_emb = self.input_emb(flattened_sample)
            # unravel
        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([diffusion_timestep_embedding, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            # -> batch, 1, n_embed
            # cond_embeddings = diffusion_timestep_embedding
            if self.obs_as_cond:
                assert not self.unconditional
                # cond: effective_batch, n_obs_steps, cond_dim
                # (B,To,n_emb)
                cond_obs_emb = self.cond_obs_emb(cond)

                if self.time_as_cond:
                    # the time embedding gets added to the front of the embedding sequence
                    # B, To+1, n_emb
                    cond_embeddings = torch.cat([diffusion_timestep_embedding, cond_obs_emb], dim=1)

                    # tc: this is the n_obs_steps
                    tc = cond_embeddings.shape[1]


                    # we need a position/time embedding for each horizon index
                    cond_position_embeddings = self.cond_pos_emb[
                        :, :tc, :
                    ]  # each position maps to a (learnable) vector

                    x = self.drop(cond_embeddings + cond_position_embeddings)
                else:
                    cond_position_embeddings = self.cond_pos_emb[
                        :, :self.n_obs_steps, :
                    ]  # each position maps to a (learnable) vector

                    # make sure that each conditioning RGB frame has time
                    # this is the rgb embedding before self attention
                    # x = self.drop(cond_embeddings + position_embeddings)

                    x = self.drop(self.cond_obs_combiner(torch.cat([cond_obs_emb,
                                                                    diffusion_timestep_embedding.expand(effective_batch_size, self.n_obs_steps, -1),
                                                                    cond_position_embeddings.expand(effective_batch_size, -1, -1)], axis=-1)))
                    # x = self.drop(cond_obs_emb + diffusion_timestep_embedding + position_embeddings)

                assert not torch.any(torch.isnan(x))

            # B, nobssteps, n_embed
            # TODO: make this only attend to the hands that exist
            # even without that, should still work now because we have the categorical variable

            # x here is the RGB conditioning
            if not self.unconditional:
                if self.use_flatten_hands_2x:
                    # build mask, you can only look (within the nobssteps history) at elements with valid hand booleans
                    # rgb
                    # theres no mask for rgb because we always have rgb
                    if "rgb" in self.conditioning_to_use:
                        x_rgb = self.encoder(x)
                        assert not torch.any(torch.isnan(x_rgb))

                    if "proprioceptive" in self.conditioning_to_use:
                        # takes in normalized actions and outputs an embedding
                        # embed actions
                        # concatenate diffusion timestep embedding
                        # sum position embedding
                        # drop
                        # mask based on hand boolean
                        # finally action_encode

                        # effective_batch, horizon, 84
                        # -> effective_batch, horizon*2, 42
                        # -> effective_batch, n_obs_steps, 42
                        historical_actions = action[:, :self.n_obs_steps]

                        # we expect the left and right actions to be concatenated here, and they get chunked
                        # if there are only actions for one arm, this doesn't make sense
                        flat_act_l, flat_act_r = historical_actions.chunk(2, dim=-1)
                        assert len(action.shape) == 3
                        embedded_proprioceptive = self.proprioceptive_emb(torch.cat([flat_act_l, flat_act_r], dim=1))

                        # append the diffusion timestep
                        # -> effective_batch, 1+n_obs_steps*2, d_embed
                        # print("ln507")
                        # print(diffusion_timestep_embedding.shape)
                        # print(embedded_actions.shape)
                        if self.time_as_cond:
                            embedded_proprioceptive = torch.cat([diffusion_timestep_embedding, embedded_proprioceptive], dim=1)

                            # add position embeddings plus a hand chilarity embedding
                            # EB, 1+n_obs_steps*2, d_embed
                            # diffusion timestep embedding, left cond pos embed, right cond pos embed
                            new_position_embedding = torch.cat([cond_position_embeddings, cond_position_embeddings[:, 1:, :]], axis=1)

                            # add chilarity embeddings
                            new_position_embedding[:, 1:self.n_obs_steps + 1, :] = new_position_embedding[:, 1:self.n_obs_steps + 1, :] + self.hand_chilarity_pos_emb[0].unsqueeze(0).unsqueeze(0)
                            new_position_embedding[:, self.n_obs_steps + 1:, :] = new_position_embedding[:, self.n_obs_steps + 1:, :] + self.hand_chilarity_pos_emb[1].unsqueeze(0).unsqueeze(0)
                            embedded_proprioceptive = self.drop(embedded_proprioceptive + new_position_embedding)
                        else:
                            new_position_embedding = torch.cat([cond_position_embeddings, cond_position_embeddings], axis=1)
                            new_position_embedding[:, :self.n_obs_steps, :] = new_position_embedding[:, :self.n_obs_steps, :] + self.hand_chilarity_pos_emb[0].unsqueeze(0).unsqueeze(0)
                            new_position_embedding[:, self.n_obs_steps:, :] = new_position_embedding[:, self.n_obs_steps:, :] + self.hand_chilarity_pos_emb[1].unsqueeze(0).unsqueeze(0)
                            # embedded_actions = self.drop(embedded_actions + diffusion_timestep_embedding + new_position_embedding)

                            embedded_proprioceptive = self.drop(self.proprioceptive_combiner(torch.cat([embedded_proprioceptive,
                                                                                                        diffusion_timestep_embedding.expand(effective_batch_size, self.n_obs_steps*2, -1),
                                                                                                        new_position_embedding.expand(effective_batch_size, -1, -1)], axis=-1)))
                            # embedded_proprioceptive = self.drop(embedded_proprioceptive + diffusion_timestep_embedding + new_position_embedding)

                        # -> effective_batch, 1+n_obs_steps*2, d_embed

                        # mask: EB, 1+n_obs_steps*2, 1+n_obs_steps*2
                        # each row index queries each column index
                        # -> EB, 1+n_obs_steps*2
                        # -> EB, 1, 1+n_obs_steps*2
                        effective_batch_size = hand_present_boolean.shape[0]

                        if self.time_as_cond:
                            src_mask = torch.cat([torch.ones(effective_batch_size, 1).to(self.device),
                                                  hand_present_boolean[:, :self.n_obs_steps, 0],
                                                  hand_present_boolean[:, :self.n_obs_steps, 1]], axis=1).unsqueeze(1).expand(-1, 1+self.n_obs_steps*2, -1)
                        else:
                            src_mask = torch.cat([hand_present_boolean[:, :self.n_obs_steps, 0], hand_present_boolean[:, :self.n_obs_steps, 1]], axis=1).unsqueeze(1).expand(-1, self.n_obs_steps*2, -1)

                        # need to scale mask for heads
                        src_mask = src_mask.repeat(self.n_head, 1, 1).to(self.device)
                        x_proprioceptive = self.proprioceptive_encoder(embedded_proprioceptive,
                                                       mask=convert_boolean_mask_to_additive_mask(src_mask))
                        assert not torch.any(torch.isnan(x_proprioceptive))

                else:
                    x = self.encoder(x)

            """
            Start to construct future embeddings
            """
            # future frames
            # future_cond_embeddings = diffusion_timestep_embedding

            if "future_camera_pose" in self.conditioning_to_use:
                # print("ln570 future cond size")
                # print(future_cond.shape)
                # the time embedding gets added to the front of the embedding sequence
                future_cond_obs_emb = self.future_cond_obs_emb(future_cond)

                if self.time_as_cond:
                    future_cond_obs_emb = torch.cat([diffusion_timestep_embedding, future_cond_obs_emb], dim=1)

                    # cond_pos_emb: 1, horizon, n_emb gets indexed
                    future_position_embeddings = torch.cat([self.cond_pos_emb[:, 0:1, :], self.cond_pos_emb[:, -(self.horizon - self.n_obs_steps):, :]], axis=1)

                    future_x = self.drop(future_cond_obs_emb + future_position_embeddings)
                else:
                    future_position_embeddings = self.cond_pos_emb[:, self.n_obs_steps:, :]

                    future_x = self.drop(self.future_cond_obs_combiner(torch.cat([future_cond_obs_emb,
                                                                                  diffusion_timestep_embedding.expand(effective_batch_size, self.horizon - self.n_obs_steps, -1),
                                                                                  future_position_embeddings.expand(effective_batch_size, -1, -1)],  axis=-1)))

                future_x = self.future_encoder(future_x)

                assert not torch.any(torch.isnan(future_x))

            """
            End construction of future embeddings
            """
            if not self.unconditional:
                if self.use_flatten_hands_2x:
                    pass  # setup memory later
                else:
                    # (B,T_cond,n_emb)
                    # T_cond is the number of obs steps
                    memory = torch.cat([x, future_x], axis=1)


            # decoder
            # the below line contains the noisy actions
            # B, horizon, action_embed_dim
            # input_emb is the action embedding
            token_embeddings = input_emb

            # t is the action horizon
            # but pos_emb is also created with t
            # position embedding embeds the timesteps
            # token_seq_len = token_embeddings.shape[1]

            # -> EB, horizon, n_emb
            position_embeddings = self.pos_emb[
                :, :, :
            ]  # each position maps to a (learnable) vector

            # combine the actions with time embeddings
            if self.unconditional or self.use_flatten_hands_2x:
                # remember: in the flatten 2x case, we have 2*horizon actions
                if self.use_2x_horizon:
                    x = self.drop(token_embeddings +
                                  position_embeddings)
                else:
                    x = self.drop(token_embeddings +
                                  torch.cat([position_embeddings, position_embeddings], axis=1) +
                                  # diffusion_timestep_embedding +
                                  torch.cat([self.hand_chilarity_pos_emb[0].tile(self.horizon, 1), self.hand_chilarity_pos_emb[1].tile(self.horizon, 1)], axis=0).unsqueeze(0))
                assert not torch.any(torch.isnan(x))
            else:
                raise NotImplementedError
                x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)

            """
            start building masks
            """
            if self.use_hand_collapse_input_emb:
                # the tgt mask should allow quadratic attention amongst all action elements that are valid
                # zero otherwise...
                # -> B, horizon
                hand_present_boolean_solo = torch.logical_or(hand_present_boolean[..., 0], hand_present_boolean[..., 1])

                batch_size = hand_present_boolean_solo.shape[0]
                horizon = hand_present_boolean_solo.shape[1]

                # unsqueeze the horizon and head dimension
                # stack the hpbs along the rows
                # -> B, horizon, horizon
                mask = hand_present_boolean_solo.unsqueeze(1).expand(-1, horizon, -1)

                # -> B, num_heads, horizon, horizon -> B*num_heads, horizon, horizon
                mask = mask.unsqueeze(1).expand(-1, self.n_head, -1,  -1).reshape(batch_size*self.n_head, horizon, horizon)

                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

                # the memory mask should allow querying context vectors that are valid, plus the initial diffusion time embedding
                # -> B, n_obs_steps + 1 for time dimension
                # nobs_steps = tc - 1
                # BUT: because we have a lot of future timestep conditioning here, we need to just stick the WHOLE horizon / hand_present_boolean_solo in here
                base_mem_mask = torch.cat([torch.ones(batch_size, 1).to(self.device), hand_present_boolean_solo], axis=1)

                # -> B, horizon, horizon + 1
                memory_mask = base_mem_mask.unsqueeze(1).expand(-1, horizon, -1)

                # -> B, num_heads, horizon, horizon + 1 -> B*num_heads, horizon, horizon + 1
                memory_mask = memory_mask.unsqueeze(1).expand(-1, self.n_head, -1, -1).reshape(batch_size*self.n_head, horizon, horizon+1)

                memory_mask = memory_mask.float().masked_fill(memory_mask == 0, float('-inf')).masked_fill(memory_mask == 1, float(0.0))
                """
                stop building masks
                """
                x = self.decoder(
                    tgt=x,
                    memory=memory,
                    tgt_mask=mask.clone().detach(),
                    memory_mask=memory_mask.clone().detach()
                )
            elif self.use_flatten_hands_2x:
                # -> effective_batch, 2*horizon

                # just mask out all the inactive hands
                effective_batch_size = hand_present_boolean.shape[0]
                horizon = hand_present_boolean.shape[1]

                # notice: this is of size HORIZON
                # so we try to reconstruct existing hands with this objective too
                if self.mask_history:
                    tgt_mask = torch.cat([torch.zeros(effective_batch_size, self.n_obs_steps).to(self.device),
                                          hand_present_boolean[..., 0][:, self.n_obs_steps:],
                                          torch.zeros(effective_batch_size, self.n_obs_steps).to(self.device),
                                          hand_present_boolean[..., 1][:, self.n_obs_steps:]], axis=1).unsqueeze(1).expand(-1, 2*horizon, -1).repeat(self.n_head, 1, 1).to(self.device)
                else:
                    tgt_mask = torch.cat([hand_present_boolean[..., 0], hand_present_boolean[..., 1]], axis=1).unsqueeze(1).expand(-1, 2*horizon, -1).repeat(self.n_head, 1, 1).to(self.device)

                if self.unconditional:
                    # only do self attention on actions
                    x = self.decoder(
                        x,
                        mask=convert_boolean_mask_to_additive_mask(tgt_mask),
                )
                else:
                    memory_list = []
                    memory_mask_list = []

                    for key in self.conditioning_to_use:
                        assert key in ["proprioceptive", "rgb", "future_camera_pose"]

                        if key == "rgb":
                            memory_rgb = x_rgb

                            if self.time_as_cond:
                                memory_rgb_mask = torch.ones(effective_batch_size, 2 * horizon,
                                                             self.n_obs_steps + 1).repeat(self.n_head, 1, 1).to(self.device)
                            else:
                                memory_rgb_mask = torch.ones(effective_batch_size, 2 * horizon,
                                                             self.n_obs_steps).repeat(self.n_head, 1, 1).to(self.device)
                            memory_list.append(memory_rgb)
                            memory_mask_list.append(memory_rgb_mask)
                        elif key == "proprioceptive":
                            memory_action = x_proprioceptive
                            if self.time_as_cond:
                                memory_action_mask = torch.cat([torch.ones(effective_batch_size, 1).to(self.device),
                                                                hand_present_boolean[..., :self.n_obs_steps, 0],
                                                                hand_present_boolean[..., :self.n_obs_steps, 1]],
                                                               axis=1).unsqueeze(1).expand(-1, 2 * horizon, -1).repeat(
                                    self.n_head, 1, 1).to(self.device)
                            else:
                                memory_action_mask = torch.cat([
                                                                hand_present_boolean[..., :self.n_obs_steps, 0],
                                                                hand_present_boolean[..., :self.n_obs_steps, 1]],
                                                               axis=1).unsqueeze(1).expand(-1, 2 * horizon, -1).repeat(
                                    self.n_head, 1, 1).to(self.device)
                            memory_list.append(memory_action)
                            memory_mask_list.append(memory_action_mask)
                        elif key == "future_camera_pose":
                            memory_future_camera_pose = future_x
                            if self.time_as_cond:
                                memory_future_camera_pose_mask = torch.ones(effective_batch_size, horizon - self.n_obs_steps + 1).unsqueeze(1).expand(-1, 2*horizon, -1).repeat(self.n_head, 1, 1).to(self.device)
                            else:
                                memory_future_camera_pose_mask = torch.ones(effective_batch_size, horizon - self.n_obs_steps).unsqueeze(1).expand(-1, 2*horizon, -1).repeat(self.n_head, 1, 1).to(self.device)
                            memory_list.append(memory_future_camera_pose)
                            memory_mask_list.append(memory_future_camera_pose_mask)

                    # none of the valid queries are allowed to interact with invalid queries during decoding
                    # sa: in this stage, the tgt mask guarantees non-interaction
                    # msa / cross attention: in this stage, invalid queries are allowed to interact with the valid context/memory, but the embeddings they produce keep propagating through success layers without affecting the valid indices
                    # ff

                    # any query can access the memory
                    # for each memory mask, each action can only attend to
                    import pdb
                    pdb.set_trace()
                    x = self.decoder(
                        tgt=x,
                        tgt_mask=tgt_mask,
                        memory_list=memory_list,
                         memory_mask_list=[convert_boolean_mask_to_additive_mask(_) for _ in memory_mask_list]
                    )
                    assert not torch.any(torch.isnan(x))
            else:
                raise NotImplementedError
            # x = self.decoder(
            #     tgt=x,
            #     memory=memory,
            #     tgt_mask=self.mask,
            #     memory_mask=self.memory_mask
            # )
            # (B,T,n_emb)

        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)

        if self.use_flatten_hands_2x:
            # -> effective_batch, horizon, 84
            x = x.reshape(effective_batch_size, horizon, 2, int(self.output_dim//2)).flatten(start_dim=-2, end_dim=-1)
        return x


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)



def unit_test_my_mask():
    # test if my mask is correctly working
    # TODO:
    pass