import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
import wandb

from sub_models.functions_losses import SymLogTwoHotLoss, SeparationLoss
from sub_models.attention_blocks import get_subsequent_mask_with_batch_length, get_subsequent_mask
from sub_models.transformer_model import StochasticTransformerKVCache
import agents


class EncoderBN(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        feature_width = 64//2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))

        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B L C H W -> (B L) C H W")
        x = self.backbone(x)
        x = rearrange(x, "(B L) C H W -> B L (C H W)", B=batch_size)
        return x


class DecoderBN(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels*final_feature_width*final_feature_width, bias=False))
        backbone.append(Rearrange('B L (C H W) -> (B L) C H W', C=last_channels, H=final_feature_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "(B L) C H W -> B L C H W", B=batch_size)
        return obs_hat


class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, transformer_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(transformer_hidden_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B L (K C) -> B L K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits


class RewardDecoder(nn.Module):
    def __init__(self, num_classes, embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(transformer_hidden_dim, num_classes)

    def forward(self, feat):
        feat = self.backbone(feat)
        reward = self.head(feat)
        return reward


class TerminationDecoder(nn.Module):
    def __init__(self,  embedding_size, transformer_hidden_dim) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(transformer_hidden_dim, transformer_hidden_dim, bias=False),
            nn.LayerNorm(transformer_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(transformer_hidden_dim, 1),
            # nn.Sigmoid()
        )

    def forward(self, feat):
        feat = self.backbone(feat)
        termination = self.head(feat)
        termination = termination.squeeze(-1)  # remove last 1 dim
        return termination


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, obs_hat, obs):
        loss = (obs_hat - obs)**2
        loss = reduce(loss, "B L C H W -> B L", "sum")
        return loss.mean()


class CategoricalKLDivLossWithFreeBits(nn.Module):
    def __init__(self, free_bits) -> None:
        super().__init__()
        self.free_bits = free_bits

    def forward(self, p_logits, q_logits):
        p_dist = OneHotCategorical(logits=p_logits)
        q_dist = OneHotCategorical(logits=q_logits)
        kl_div = torch.distributions.kl.kl_divergence(p_dist, q_dist) # (B, L-1, stoch_dim)
        kl_div = reduce(kl_div, "B L D -> B L", "sum")
        kl_div = kl_div.mean()
        real_kl_div = kl_div
        kl_div = torch.max(torch.ones_like(kl_div)*self.free_bits, kl_div)
        return kl_div, real_kl_div


class WorldModel(nn.Module):
    def __init__(self, action_dim, record_run, conf):
        super().__init__()
        self.transformer_hidden_dim = conf.Models.WorldModel.TransformerHiddenDim
        self.final_feature_width = 4
        self.stoch_dim = 32
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim
        self.use_amp = True
        self.tensor_dtype = torch.float16 if self.use_amp else torch.float32
        self.imagine_batch_size = -1
        self.imagine_batch_length = -1
        self.record_run= record_run

        # HarmonyDream
        self.sigma_obs = nn.Parameter(torch.tensor(0.0))
        self.sigma_reward = nn.Parameter(torch.tensor(0.0))
        self.sigma_dyn = nn.Parameter(torch.tensor(0.0))
        self.sigma_att = nn.Parameter(torch.tensor(0.0))
        self.sigma_rep = nn.Parameter(torch.tensor(0.0))
        self.sigma_cap = nn.Parameter(torch.tensor(0.0))
        self.sigma_conIso = nn.Parameter(torch.tensor(0.0))

        # Separation loss
        self.lambda_rep = conf.Models.WorldModel.SeparationLoss.RepulsionCoefficient
        self.lambda_cap = conf.Models.WorldModel.CapacityLoss.Coefficient
        self.lambda_conIso = conf.Models.WorldModel.SeparationLoss.ConformalIsometryCoefficient
        self.sep_loss_balance = conf.Models.WorldModel.SeparationLoss.SeparationLossBalance
        self.i = 0

        # Capacity loss
        self.cap_loss_gate = conf.Models.WorldModel.CapacityLoss.Gate

        self.encoder = EncoderBN(
            in_channels=conf.Models.WorldModel.InChannels,
            stem_channels=32,
            final_feature_width=self.final_feature_width
        )
        self.storm_transformer = StochasticTransformerKVCache(
            stoch_dim=self.stoch_flattened_dim,
            action_dim=action_dim,
            feat_dim=self.transformer_hidden_dim,
            num_layers=conf.Models.WorldModel.TransformerNumLayers,
            num_heads=conf.Models.WorldModel.TransformerNumHeads,
            max_length=conf.Models.WorldModel.TransformerMaxLength,
            dropout=0.1,
            conf=conf
        )
        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels*self.final_feature_width*self.final_feature_width,
            transformer_hidden_dim=self.transformer_hidden_dim,
            stoch_dim=self.stoch_dim
        )
        self.image_decoder = DecoderBN(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=conf.Models.WorldModel.InChannels,
            stem_channels=32,
            final_feature_width=self.final_feature_width
        )
        self.reward_decoder = RewardDecoder(
            num_classes=255,
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=self.transformer_hidden_dim
        )
        self.termination_decoder = TerminationDecoder(
            embedding_size=self.stoch_flattened_dim,
            transformer_hidden_dim=self.transformer_hidden_dim
        )

        self.mse_loss_func = MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_with_logits_loss_func = nn.BCEWithLogitsLoss()
        self.symlog_twohot_loss_func = SymLogTwoHotLoss(num_classes=255, lower_bound=-20, upper_bound=20)
        self.categorical_kl_div_loss = CategoricalKLDivLossWithFreeBits(free_bits=1)
        self.separation_loss_func = SeparationLoss(conf=conf)
        self.optimizer = torch.optim.AdamW(self.parameters(), 
                                           lr=conf.Models.WorldModel.LearningRate, 
                                           weight_decay=conf.Models.WorldModel.WeightDecay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def encode_obs(self, obs):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            embedding = self.encoder(obs)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.straight_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        
        return flattened_sample

    def calc_last_dist_feat(self, latent, action):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            temporal_mask = get_subsequent_mask(latent)
            dist_feat = self.storm_transformer(latent, action, temporal_mask)
            last_dist_feat = dist_feat[:, -1:]
            prior_logits = self.dist_head.forward_prior(last_dist_feat)
            prior_sample = self.straight_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
        return prior_flattened_sample, last_dist_feat

    def predict_next(self, last_flattened_sample, action, log_video=True):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            dist_feat = self.storm_transformer.forward_with_kv_cache(last_flattened_sample, action)
            prior_logits = self.dist_head.forward_prior(dist_feat)

            # decoding
            prior_sample = self.straight_throught_gradient(prior_logits, sample_mode="random_sample")
            prior_flattened_sample = self.flatten_sample(prior_sample)
            if log_video:
                obs_hat = self.image_decoder(prior_flattened_sample)
            else:
                obs_hat = None
            reward_hat = self.reward_decoder(dist_feat)
            reward_hat = self.symlog_twohot_loss_func.decode(reward_hat)
            termination_hat = self.termination_decoder(dist_feat)
            termination_hat = termination_hat > 0

        return obs_hat, reward_hat, termination_hat, prior_flattened_sample, dist_feat

    def straight_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample

    def flatten_sample(self, sample):
        return rearrange(sample, "B L K C -> B L (K C)")

    def init_imagine_buffer(self, imagine_batch_size, imagine_batch_length, dtype):
        '''
        This can slightly improve the efficiency of imagine_data
        But may vary across different machines
        '''
        if self.imagine_batch_size != imagine_batch_size or self.imagine_batch_length != imagine_batch_length:
            print(f"init_imagine_buffer: {imagine_batch_size}x{imagine_batch_length}@{dtype}")
            self.imagine_batch_size = imagine_batch_size
            self.imagine_batch_length = imagine_batch_length
            latent_size = (imagine_batch_size, imagine_batch_length+1, self.stoch_flattened_dim)
            hidden_size = (imagine_batch_size, imagine_batch_length+1, self.transformer_hidden_dim)
            scalar_size = (imagine_batch_size, imagine_batch_length)
            self.latent_buffer = torch.zeros(latent_size, dtype=dtype, device="cuda")
            self.hidden_buffer = torch.zeros(hidden_size, dtype=dtype, device="cuda")
            self.action_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.reward_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")
            self.termination_hat_buffer = torch.zeros(scalar_size, dtype=dtype, device="cuda")

    def imagine_data(self, agent: agents.ActorCriticAgent, sample_obs, sample_action,
                     imagine_batch_size, imagine_batch_length, log_video, logger):
        self.init_imagine_buffer(imagine_batch_size, imagine_batch_length, dtype=self.tensor_dtype)
        obs_hat_list = []

        self.storm_transformer.reset_kv_cache_list(imagine_batch_size, dtype=self.tensor_dtype)
        # context
        context_latent = self.encode_obs(sample_obs)
        for i in range(sample_obs.shape[1]):  # context_length is sample_obs.shape[1]
            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                context_latent[:, i:i+1],
                sample_action[:, i:i+1],
                log_video=log_video
            )
        self.latent_buffer[:, 0:1] = last_latent
        self.hidden_buffer[:, 0:1] = last_dist_feat

        # imagine
        for i in range(imagine_batch_length):
            action = agent.sample(torch.cat([self.latent_buffer[:, i:i+1], self.hidden_buffer[:, i:i+1]], dim=-1))
            self.action_buffer[:, i:i+1] = action

            last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = self.predict_next(
                self.latent_buffer[:, i:i+1], self.action_buffer[:, i:i+1], log_video=log_video)

            self.latent_buffer[:, i+1:i+2] = last_latent
            self.hidden_buffer[:, i+1:i+2] = last_dist_feat
            self.reward_hat_buffer[:, i:i+1] = last_reward_hat
            self.termination_hat_buffer[:, i:i+1] = last_termination_hat
            if log_video:
                obs_hat_list.append(last_obs_hat[::imagine_batch_size//16])  # uniform sample vec_env

        if log_video:
            logger.log("Imagine/predict_video", torch.clamp(torch.cat(obs_hat_list, dim=1), 0, 1).cpu().float().detach().numpy())

        return torch.cat([self.latent_buffer, self.hidden_buffer], dim=-1), self.action_buffer, self.reward_hat_buffer, self.termination_hat_buffer

    def update(self, obs, action, reward, termination, total_steps, logger=None):
        self.train()
        batch_size, batch_length = obs.shape[:2]

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.use_amp):
            # encoding
            embedding = self.encoder(obs) # (B, L, 4096)
            post_logits = self.dist_head.forward_post(embedding) # (B, L, stoch_dim, stoch_dim)
            sample = self.straight_throught_gradient(post_logits, sample_mode="random_sample") # (B, L, stoch_dim, stoch_dim)
            flattened_sample = self.flatten_sample(sample) # (B, L, stoch_dim*stoch_dim)

            # decoding image
            obs_hat = self.image_decoder(flattened_sample) # (B, L, C, H, W)

            # transformer
            temporal_mask = get_subsequent_mask_with_batch_length(batch_length, flattened_sample.device) # (1, L, L)
            dist_feat = self.storm_transformer(flattened_sample, action, temporal_mask) # (B, L, h)
            prior_logits = self.dist_head.forward_prior(dist_feat)
            
            # decoding reward and termination with dist_feat
            reward_hat = self.reward_decoder(dist_feat)
            termination_hat = self.termination_decoder(dist_feat)

            # env loss
            reconstruction_loss = self.mse_loss_func(obs_hat, obs)
            reward_loss = self.symlog_twohot_loss_func(reward_hat, reward)
            termination_loss = self.bce_with_logits_loss_func(termination_hat, termination)

            # dyn-rep loss
            dynamics_loss, dynamics_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:].detach(), prior_logits[:, :-1])
            representation_loss, representation_real_kl_div = self.categorical_kl_div_loss(post_logits[:, 1:], prior_logits[:, :-1].detach())
            
            # Separation loss
            att_loss, rep_loss, conIso_loss, stats = self.separation_loss_func(prior_logits, dist_feat)

            # capacity loss
            cap_loss = reduce(dist_feat, "B L D -> D", "mean")
            cap_loss = -torch.sum(cap_loss ** 2)
            cap_loss_gated = cap_loss < self.cap_loss_gate

            # HarmonyDream >>>
            # group losses
            obs_loss = reconstruction_loss
            reward_loss_group = reward_loss + termination_loss
            dynamics_loss_group = 0.5 * dynamics_loss + 0.1 * representation_loss

            # get sigmas
            sigma_obs = torch.exp(self.sigma_obs)
            sigma_reward = torch.exp(self.sigma_reward)
            sigma_dyn = torch.exp(self.sigma_dyn)
            sigma_att = torch.exp(self.sigma_att)
            sigma_rep = torch.exp(self.sigma_rep)
            sigma_cap = torch.exp(self.sigma_cap)
            sigma_conIso = torch.exp(self.sigma_conIso)

            # Calculate rectified Harmonious Loss
            harmonized_obs_loss = obs_loss / sigma_obs + torch.log(1 + sigma_obs)
            harmonized_reward_loss = reward_loss_group / sigma_reward + torch.log(1 + sigma_reward)
            harmonized_dynamics_loss = dynamics_loss_group / sigma_dyn + torch.log(1 + sigma_dyn)
            harmonized_att_loss = torch.where(stats['att_loss_gated'], 0.0, att_loss / sigma_att + torch.log(1 + sigma_att))
            harmonized_rep_loss = torch.where(stats['rep_loss_gated'], 0.0, rep_loss / sigma_rep + torch.log(1 + sigma_rep))
            harmonized_cap_loss = torch.where(cap_loss_gated, 0.0, 
                                              cap_loss / sigma_cap.detach() - cap_loss.detach() / sigma_cap + torch.log(1 + sigma_cap))
            harmonized_conIso_loss = conIso_loss / sigma_conIso + torch.log(1 + sigma_conIso)
            # <<< HarmonyDream
            
            total_loss = harmonized_obs_loss + harmonized_reward_loss + harmonized_dynamics_loss
            if self.i > 20000: 
                total_loss += self.sep_loss_balance * (harmonized_att_loss + 
                                                       self.lambda_rep * harmonized_rep_loss +
                                                       self.lambda_cap * harmonized_cap_loss +
                                                       self.lambda_conIso * harmonized_conIso_loss)
            self.i += 1
    
            # total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss
            # total_loss = reconstruction_loss + reward_loss + termination_loss + 0.5*dynamics_loss + 0.1*representation_loss + \
                        #  att_loss + rep_loss + cap_loss + conIso_loss

            

        # gradient descent
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1000.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        if self.record_run:
            wandb.log({
                "WorldModel/1.0.reconstruction_loss": reconstruction_loss.item(),
                "WorldModel/1.1.reward_loss": reward_loss.item(),
                "WorldModel/1.2.termination_loss": termination_loss.item(),
                "WorldModel/1.3.dynamics_loss": dynamics_loss.item(),
                "WorldModel/1.4.representation_loss": representation_loss.item(), 
                "WorldModel/1.5.att_loss": att_loss.item(),
                "WorldModel/1.6.rep_loss": rep_loss.item(),
                "WorldModel/1.7.cap_loss": cap_loss.item(), 
                "WorldMode;/1.8.conIso_loss": conIso_loss.item(),
                "WorldModel/1.8.harmonized_obs_loss": harmonized_obs_loss.item(),
                "WorldModel/2.0.harmonized_reward_loss": harmonized_reward_loss.item(),
                "WorldModel/2.1.harmonized_dynamics_loss": harmonized_dynamics_loss.item(),
                "WorldModel/2.2.harmonized_att_loss": harmonized_att_loss.item(),
                "WorldModel/2.3.harmonized_rep_loss": harmonized_rep_loss.item(),
                "WorldModel/2.4.harmonized_cap_loss": harmonized_cap_loss.item(),
                "WorldModel/2.5.harmonized_conIso_loss": harmonized_conIso_loss.item(),
                "WorldModel/2.5.total_loss": total_loss.item(),
                "sep_loss/1.0.mean_jsd": stats["mean_jsd"],
                "sep_loss/1.1.std_jsd": stats["std_jsd"],
                "sep_loss/2.0.pairwise_mse_mean": stats["pairwise_mse_mean"],
                "sep_loss/2.1.pairwise_mse_std": stats["pairwise_mse_std"],
                "sep_loss/3.0.att_loss": stats["att_loss"],
                "sep_loss/3.1.rep_loss": stats["rep_loss"],
                "sep_loss/3.2.conIso_loss": stats["conIso_loss"],
                "sep_loss/4.0.att_pairs_ratio": stats["att_pairs_ratio"],
                "sep_loss/5.0.rep_pairs_ratio": stats["rep_pairs_ratio"],
            }, step=total_steps)
        
        # if logger is not None:
            # logger.log("WorldModel/reconstruction_loss", reconstruction_loss.item())
            # logger.log("WorldModel/reward_loss", reward_loss.item())
            # logger.log("WorldModel/termination_loss", termination_loss.item())
            # logger.log("WorldModel/dynamics_loss", dynamics_loss.item())
            # logger.log("WorldModel/dynamics_real_kl_div", dynamics_real_kl_div.item())
            # logger.log("WorldModel/representation_loss", representation_loss.item())
            # logger.log("WorldModel/representation_real_kl_div", representation_real_kl_div.item())
            # logger.log("WorldModel/total_loss", total_loss.item())
            # logger.log("sep_loss/1.mean_jsd", stats["mean_jsd"])
            # logger.log("sep_loss/1.std_jsd", stats["std_jsd"])
            # logger.log("sep_loss/2.pairwise_mse_mean", stats["pairwise_mse_mean"])
            # logger.log("sep_loss/2.pairwise_mse_std", stats["pairwise_mse_std"])
            # logger.log("sep_loss/3.sotf_att", stats["soft_att"])
            # logger.log("sep_loss/3.soft_rep", stats["sotf_rep"])
            # logger.log("sep_loss/4.att_loss", stats["att_loss"])
            # logger.log("sep_loss/4.rep_loss", stats["rep_loss"])
            # logger.log("sep_loss/5.att_pairs_ratio", stats["att_pairs_ratio"])
            # logger.log("sep_loss/5.rep_pairs_ratio", stats["rep_pairs_ratio"])
            # logger.log("sep_loss/6.threshold", self.sep_threshold.item())
            # logger.log("sep_loss/6.sep_threshold_grad", stats["sep_threshold_grad"])
            

