"""
networks.py

Neural network components used across the project (RL agents, distributional RL, PPO, and
sequence/transformer-based policies).

This module includes:
- Initialization helpers (`weight_init`, `weight_init_xavier`)
- Utility layers (`SafeBatchNorm1d`, `NoisyLinear`)
- Value networks for DQN variants (`Dueling_QNetwork`)
- FQF building blocks (`QVN`, `FPN`)
- Attention/Transformer encoders for the tabular firefighter-state representation
- Decision Transformer architecture (`DT_Network` and helpers)
- PPO actor-critic networks (`PPO_ActorCriticAM`, `PPOActorCritic`)

Import safety
-------------
This module contains no executable script block. It does set global matmul precision via
`torch.set_float32_matmul_precision('high')` at import time, which is an intentional global
setting in the original file and is preserved here.

Notes
-----
- Several networks assume a specific "state tensor" layout (e.g., 82 lines = 2 header lines
  + 80 firefighter lines; 40 features per line).
- Some parts include asserts intended to help debugging/compilation (e.g., in DT_Network).
"""

import torch
import torch.nn as nn
# --- Optionnel : neutraliser torch.compile pour ce module si activé ailleurs

import torch._dynamo as dynamo
torch.set_float32_matmul_precision('high')

import numpy as np
import math
import torch.nn.functional as F
from copy import copy

class SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that safely handles batch of size 1 during training.

    If the incoming batch has a single sample while the module is in training
    mode, the normalization step is skipped to avoid runtime errors. This
    mirrors a no-op identity layer for that particular batch size.
    """

    def forward(self, s):  # type: ignore[override]
        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        if self.training and s.size(0) == 1:
            return s
        return super().forward(s)




def weight_init(modules):
    """
    Initialize Linear layers with Kaiming normal weights and zero biases.
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """
    for module in modules:
        if isinstance(module, nn.Sequential):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
def weight_init_xavier(layers):
    """
    Initialize provided Linear layers with Xavier uniform weights (small gain).
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """
    for layer in layers:
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)


class NoisyLinear(nn.Linear):
    """
    Noisy linear layer (factorized Gaussian noise) for exploration in value-based RL.
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """
    # Noisy Linear Layer for factorised Gaussian noise 
    def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_init = sigma_init
        # make the sigmas trainable:
        self.sigma_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_weight = nn.Parameter(torch.FloatTensor(out_features, in_features))

        # extra parameter for the bias 
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features,))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        # not trainable tensor for the nn.Module
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))
        # reset parameter as initialization of the layer
        self.reset_parameter()
    
    def reset_parameter(self):  

        """
        reset_parameter.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        bound = 1 / math.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def f(self, x):

        """
        F.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        return x.normal_().sign().mul(x.abs().sqrt())

    def forward(self, state):
        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        # sample random noise in sigma weight buffer and bias buffer
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

        weight = self.mu_weight + self.sigma_weight * self.eps_q.ger(self.eps_p)
        bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()

        return F.linear(state, weight, bias) 
        
    
class Dueling_QNetwork(nn.Module):

    """
    Dueling Q-network with optional attention-based state encoding and noisy layers.
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """

    def __init__(self, state_size, action_size, layer_size, n_step, seed, num_layers = 8, layer_type="ff", use_batchnorm=True):

        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        super(Dueling_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.use_batchnorm = use_batchnorm

        # AM
        self.n_heads = 4
        self.d_model = 64
        self.d_input = 40 # roles + disp
        self.attention = Attention(self.d_input, self.d_model, self.n_heads)

        # infos NN
        self.role_encoder = nn.Linear(self.d_input, self.d_model)
        self.infos_encoder = nn.Linear(self.d_input, self.d_model)

        # Standard NN

        self.emb_state_size = self.d_model * (action_size + 2)

        layers = []

        layers.append(nn.Linear(self.emb_state_size, layer_size))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        self.head = nn.Linear(self.state_size, self.layer_size) # no AM

        # Advantage / Value
        
        if layer_type == "noisy": # pas de batchnorm pour les noisy nets

            self.ff_1_A = NoisyLinear(layer_size, layer_size)
            self.ff_1_V = NoisyLinear(layer_size, layer_size)
            self.advantage = NoisyLinear(layer_size,action_size)
            self.value = NoisyLinear(layer_size,1)
            # weight_init([self.model,self.ff_1_A, self.ff_1_V]) # no init for noisy

        else:

            self.ff_1_A = nn.Linear(layer_size, layer_size)
            self.ff_1_V = nn.Linear(layer_size, layer_size)
            self.bn_A = nn.BatchNorm1d(layer_size)
            self.bn_V = nn.BatchNorm1d(layer_size)
            self.advantage = nn.Linear(layer_size,action_size)
            self.value = nn.Linear(layer_size,1)
            weight_init([self.model,self.ff_1_A, self.ff_1_V])       
        
    def forward(self, state):
        
        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        
        # if state.dim() == 1:
        #     state = state.unsqueeze(0)

        if state.dim() == 2:
            state = state.unsqueeze(0)

        B, L, F = state.shape

        # print("1", state.shape)

        infos_line = state[:, 0, :]    # [B, F]
        role_line = state[:, 1, :]   # [B, F]
        ff_state = state[:, 2:, :]   # [B, N, F]
        # print("ff_state.shape =", ff_state.shape)
        attn_output = self.attention(ff_state) 
        x_flat = attn_output.flatten(start_dim=1)

        infos_vec = self.infos_encoder(infos_line)
        role_vec = self.role_encoder(role_line)
        
        
        x = torch.cat([infos_vec, role_vec, x_flat], dim=1)

        # print("2", x.shape)

        
        # no AM
        # x = self.head(state)

        x = self.model(x)
        # print("post unsq state:", state.shape)

        if self.layer_type == "noisy": # pas de batchnorm pour les noisy nets
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))
        else:
            x_A = torch.relu(self.bn_A(self.ff_1_A(x)))
            x_V = torch.relu(self.bn_V(self.ff_1_V(x)))

        value = self.value(x_V)
        value = value.expand(state.size(0), self.action_size)
        advantage = self.advantage(x_A)
        Q = value + advantage - advantage.mean()
        if self.use_batchnorm:
            Q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            Q = value + advantage - advantage.mean()
        return Q



class QVN(nn.Module):
    """Quantile Value Network"""
    def __init__(self, state_size, action_size,layer_size, AM, n_steps, device, seed, N, num_layers, layer_type, use_batchnorm):
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        super(QVN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.use_batchnorm = use_batchnorm
        self.AM = AM
        self.N = N
        self.n_cos = 64
        self.pis = torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1,1,self.n_cos).to(device)
        self.device = device

        # Network Architecture
 
        self.cos_embedding = nn.Linear(self.n_cos, self.layer_size)
        self.cos_layer_out = self.layer_size

        layer = []

        if self.AM:

            print("with AM")

            # AM
            self.n_heads = 4
            self.d_model = 64
            self.d_input = 40 # roles + disp
            self.attention = Attention(self.d_input, self.d_model, self.n_heads)
            self.num_entities = self.action_size + 2
    
            # infos NN
            self.role_encoder = nn.Linear(self.d_input, self.d_model)
            self.infos_encoder = nn.Linear(self.d_input, self.d_model)
    
            # Standard NN
    
            self.emb_state_size = self.d_model * (self.action_size + 2)            
            layer.append(nn.Linear(self.emb_state_size, layer_size))


        else:

            print("without AM")
            
            layer.append(nn.Linear(self.state_size, layer_size))

            
        if use_batchnorm:
            layer.append(nn.BatchNorm1d(layer_size))
        layer.append(nn.ReLU())

        self.head = nn.Sequential(*layer)

        layers = []

        layers.append(nn.Linear(layer_size, layer_size))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(layer_size, layer_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)

        if layer_type == "noisy": # pas de batchnorm pour les noisy nets

            self.ff_1_A = NoisyLinear(layer_size, layer_size)
            self.ff_1_V = NoisyLinear(layer_size, layer_size)
            self.advantage = NoisyLinear(layer_size,action_size)
            self.value = NoisyLinear(layer_size,1)
            # weight_init([self.model,self.ff_1_A, self.ff_1_V]) # no init for noisy

        else:

            self.ff_1_A = nn.Linear(layer_size, layer_size)
            self.ff_1_V = nn.Linear(layer_size, layer_size)
            self.bn_A = nn.BatchNorm1d(layer_size)
            self.bn_V = nn.BatchNorm1d(layer_size)
            self.advantage = nn.Linear(layer_size,action_size)
            self.value = nn.Linear(layer_size,1)
            weight_init([self.model,self.ff_1_A, self.ff_1_V])  
 


    def calc_input_layer(self):
        """
        calc_input_layer.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        dummy_state = torch.zeros(1, self.num_entities, self.d_input, device=self.device)
        x = self._encode_state(dummy_state)
        x = self.model(x)
        return x.flatten().shape[0]

    def calc_cos(self,taus):

        """
        calc_cos.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        batch_size = taus.shape[0]
        n_tau = taus.shape[1]
        cos = torch.cos(taus.unsqueeze(-1)*self.pis)

        assert cos.shape == (batch_size,n_tau,self.n_cos), "cos shape is incorrect"
        return cos

    def _reshape_state(self, state: torch.Tensor) -> torch.Tensor:
        """Ensure the state tensor has shape [B, L, F]."""

        if state.dim() == 1:
            if state.numel() % self.d_input != 0:
                raise AssertionError(
                    f"Flattened state of length {state.numel()} is not divisible by feature size {self.d_input}"
                )
            state = state.reshape(1, -1, self.d_input)
        elif state.dim() == 2:
            if state.shape[-1] == self.d_input and state.shape[0] == self.num_entities:
                state = state.unsqueeze(0)
            elif state.shape[-1] == self.d_input and state.shape[0] != self.num_entities:
                state = state.unsqueeze(0)
            elif state.shape[-1] % self.d_input == 0:
                L = state.shape[-1] // self.d_input
                state = state.reshape(state.shape[0], L, self.d_input)
            else:
                raise AssertionError(
                    f"State with shape {state.shape} cannot be reshaped to [B, L, {self.d_input}]"
                )
        elif state.dim() != 3:
            raise AssertionError(
                f"Unsupported state dimension {state.dim()}; expected 1D, 2D, or 3D tensor"
            )

        if state.shape[-1] != self.d_input:
            raise AssertionError(
                f"Expected feature dimension {self.d_input}, got {state.shape[-1]}"
            )
        return state

    def _encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        _encode_state.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        state = self._reshape_state(state)

        B, L, F = state.shape

        infos_line = state[:, 0, :]
        role_line = state[:, 1, :]
        ff_state = state[:, 2:, :]

        attn_output = self.attention(ff_state)
        x_flat = attn_output.flatten(start_dim=1)

        infos_vec = self.infos_encoder(infos_line)
        role_vec = self.role_encoder(role_line)

        x = torch.cat([infos_vec, role_vec, x_flat], dim=1)
        x = self.head(x)
        return x

    def forward(self, state):

        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        x = self._encode_state(state)

        return self.model(x)
        
    def get_quantiles(self, x, taus, embedding=None):

        """
        Get quantiles.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        if embedding==None:
            x = self.head(x)
        else:
            x = embedding
            
        batch_size = x.shape[0]
        num_tau = taus.shape[1]
        cos = self.calc_cos(taus) # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size*num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, num_tau, self.cos_layer_out) # (batch, n_tau, layer)
        
        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1)*cos_x).view(batch_size*num_tau, self.cos_layer_out)           
        x = self.model(x)

        if self.layer_type == "noisy": # pas de batchnorm pour les noisy nets
            x_A = torch.relu(self.ff_1_A(x))
            x_V = torch.relu(self.ff_1_V(x))
        else:
            x_A = torch.relu(self.bn_A(self.ff_1_A(x)))
            x_V = torch.relu(self.bn_V(self.ff_1_V(x)))

        value = self.value(x_V)
        # value = value.expand(state.size(0), self.action_size)
        advantage = self.advantage(x_A)

        if self.use_batchnorm:
            Q = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            Q = value + advantage - advantage.mean()
        # return Q

        return Q.view(batch_size, num_tau, self.action_size)
    
    

class FPN(nn.Module):
    """Fraction proposal network"""
    def __init__(self, layer_size, seed, num_tau=8, device="cuda:0"):
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        super(FPN,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_tau = num_tau
        self.device = device
        self.ff = nn.Linear(layer_size, num_tau)
        self.softmax = nn.LogSoftmax(dim=1)
        weight_init_xavier([self.ff])
        
    def forward(self,x):
        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        q = self.softmax(self.ff(x)) 
        q_probs = q.exp()
        taus = torch.cumsum(q_probs, dim=1)
        taus = torch.cat((torch.zeros((q.shape[0], 1)).to(self.device), taus), dim=1)
        taus_ = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        
        entropy = -(q * q_probs).sum(dim=-1, keepdim=True)
        assert entropy.shape == (q.shape[0], 1), "instead shape {}".format(entropy.shape)
        
        return taus, taus_, entropy

## AM

class Attention(nn.Module):
    """
    Multi-head self-attention encoder with residual connection and LayerNorm.
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """
    def __init__(self, d_input, d_model, n_heads):
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.embedding = nn.Linear(d_input, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        x_embed = self.embedding(x)  # [n_pompiers, d_model]
        attn_output, _ = self.multihead_attn(x_embed, x_embed, x_embed)
        output = self.norm(attn_output + x_embed)  # résiduel + normalisation
        return output # .squeeze(0)  # [n_pompiers, d_model]



class FirefighterEncoder(nn.Module):
    """
    Transformer encoder for firefighter feature sequences (supports [B,N,F] or [B,T,N,F]).
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """
    def __init__(self, feature_size, d_model, n_heads, num_layers):
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        super().__init__()
        self.embedding = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):  # ff_lines: [B, 80, 40]

        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        if x.dim() == 3:
            # Input shape: [B, N, F]
            x = self.embedding(x)           # [B, N, d_model]
            x = self.encoder(x)             # [B, N, d_model]
            x = self.norm(x)
            return x                        # [B, N, d_model]

        elif x.dim() == 4:
            # Input shape: [B, T, N, F]
            B, T, N, F = x.shape
            x = self.embedding(x)           # [B, T, N, d_model]
            x = x.view(B * T, N, -1)        # [B*T, N, d_model]
            x = self.encoder(x)             # [B*T, N, d_model]
            x = self.norm(x)
            x = x.view(B, T, N, -1)         # [B, T, N, d_model]
            return x

        else:
            raise ValueError(f"Unsupported input shape {x.shape}, expected [B, N, F] or [B, T, N, F]")

### Decision Transformer Network

class DecisionTransformer(nn.Module):
    """
    Transformer encoder used as a backbone for Decision Transformer-style policies.
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """
    def __init__(self, d_model, n_heads, num_layers):
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, key_padding_mask):
        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(x)

def pad_and_mask(seqs, pad_value=0):
    """
    Pad variable-length sequences and build a boolean mask.
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """
    lengths = [seq.size(0) for seq in seqs]
    max_len = max(lengths)
    padded = torch.full((len(seqs), max_len, *seqs[0].size()[1:]), pad_value, dtype=seqs[0].dtype)
    mask = torch.zeros(len(seqs), max_len, dtype=torch.bool)
    for i, seq in enumerate(seqs):
        padded[i, :seq.size(0)] = seq
        mask[i, :seq.size(0)] = 1
    return padded, mask

class DT_Network(nn.Module):

    """
    Decision Transformer network for action prediction from (RTG, state, action) tokenization.
    
    Parameters
    ----------
    See function/class signature.
    
    Returns
    -------
    See implementation.
    """

    def __init__(self, state_size, action_size, feature_size, layer_size, num_layers, max_len, seed):
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.feature_size = feature_size
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.max_len = max_len
    
        # AM
        # self.n_heads = 4
        # self.d_model = 128
        # self.ff_layers = 4
        # self.ff_encoder = FirefighterEncoder(self.feature_size, self.d_model, self.n_heads, self.ff_layers)
    
        # # infos NN
        # self.role_encoder = nn.Sequential(
        #                                     nn.Linear(self.feature_size, self.d_model),
        #                                     nn.ReLU(),
        #                                     nn.Linear(self.d_model, self.d_model),
        #                                     nn.ReLU(),
        #                                     nn.Linear(self.d_model, self.d_model)
        #                                 )
        # self.infos_encoder = nn.Sequential(
        #                                     nn.Linear(self.feature_size, self.d_model),
        #                                     nn.ReLU(),
        #                                     nn.Linear(self.d_model, self.d_model),
        #                                     nn.ReLU(),
        #                                     nn.Linear(self.d_model, self.d_model)
        #                                 )

        # DT

        # self.layer_size = 1024 #256
        
        self.transformer = DecisionTransformer(self.layer_size, self.n_heads, self.num_layers)
        
        # version sans AM
        self.state_embed = nn.Linear(self.state_size, self.layer_size)
        # version avec AM
        # self.state_embed = nn.Linear(82*self.d_model, self.layer_size)
        
        self.action_embed = nn.Embedding(self.action_size, self.layer_size)
        self.rtg_embed = nn.Linear(1, self.layer_size)
        self.time_embed = nn.Embedding(self.max_len, self.layer_size)


        self.predict_action = nn.Linear(self.layer_size, self.action_size)

    def forward(self, states, actions, returns_to_go, timesteps, mask):

        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        # print(states.shape)

        B, T, L, F = states.size()

        full_state = states

        # Enforce static expectations on invariant dimensions to provide
        # explicit shape constraints to the compiler. Only batch (B) and
        # sequence (T) dimensions may vary between calls.
        assert L == 82, f"expected 82 lines in state tensor, got {L}"
        assert F == self.feature_size, (
            f"expected last dimension {self.feature_size}, got {F}"
        )

        # info_lines = states[:, :, 0, :]  # [B, T, F]
        # role_lines = states[:, :, 1, :]    # [B, T, F]
        # ff_lines = states[:, :, 2:, :]    # [B, T, N, F]

        # info_emb = self.infos_encoder(info_lines)  # [B, T, hidden]
        # role_emb = self.role_encoder(role_lines)        # [B, T, hidden] 
        # ff_emb = self.ff_encoder(ff_lines)           # [B, T, N, hidden]


        # full_state = torch.cat([
        #     info_emb.unsqueeze(2),  # [B, T, 1, hidden]
        #     role_emb.unsqueeze(2),  # [B, T, 1, hidden]
        #     ff_emb  # [B, T, 80, hidden]
        # ], dim=2)  # [B, T, 82, hidden]

        state_flat = full_state.flatten(start_dim=2)  # [B, T, 82 * hidden]
        state_embeddings = self.state_embed(state_flat)
        # print("state_embeddings.shape", state_embeddings.shape)

        actions = actions.reshape(B, T)
        returns_to_go = returns_to_go.reshape(B, T, 1)
        timesteps = timesteps.reshape(B, T)

        action_embeddings = self.action_embed(actions)
        # print("action_embeddings.shape", action_embeddings.shape)
        rtg_embeddings = self.rtg_embed(returns_to_go)
        # print("rtg_embeddings.shape", rtg_embeddings.shape)
        time_embeddings = self.time_embed(timesteps)
        # print("time_embeddings.shape", time_embeddings.shape)


        tokens = torch.stack((rtg_embeddings, state_embeddings, action_embeddings), dim=2).view(B, 3 * T, -1)
        time_embeddings = time_embeddings.repeat(1, 1, 3).view(B, 3 * T, -1)

        x = tokens + time_embeddings
        key_padding_mask = ~mask.unsqueeze(2).repeat(1, 1, 3).view(B, 3 * T)
        # print("x.shape", x.shape)
        x = self.transformer(x, key_padding_mask=key_padding_mask)
        # print("x.shape", x.shape)
        x = x[:, 1::3]  # only use state outputs for action prediction
        return self.predict_action(x)

# PPO AM

class PPO_ActorCriticAM(nn.Module):
    """Shared network for PPO agent producing policy logits and value."""

    def __init__(self, state_size, action_size, layer_size, seed, num_layers=8, layer_type="ff", use_batchnorm=True):

        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        super(PPO_ActorCriticAM, self).__init__()
        self.state_size = state_size
        self.layer_size = layer_size
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.use_batchnorm = use_batchnorm

        # Attention module
        self.n_heads = 4
        self.d_model = 64
        self.d_input = 40  # roles + disp = Features
        self.attention = Attention(self.d_input, self.d_model, self.n_heads)

        # infos NN
        self.role_encoder = nn.Linear(self.d_input, self.d_model)
        self.infos_encoder = nn.Linear(self.d_input, self.d_model)

        # Standard feed-forward body
        self.emb_state_size = self.d_model * (action_size + 2)
        # layers = []
        # layers.append(nn.Linear(self.emb_state_size, self.layer_size))
        # if use_batchnorm:
        #     layers.append(SafeBatchNorm1d(self.layer_size))
        # layers.append(nn.ReLU())

        # for _ in range(num_layers - 1):
        #     layers.append(nn.Linear(layer_size, layer_size))
        #     if use_batchnorm:
        #         layers.append(SafeBatchNorm1d(layer_size))
        #     layers.append(nn.ReLU())

        # self.model = nn.Sequential(*layers)

        # Policy and value heads
        # self.policy = nn.Linear(layer_size, action_size)
        # self.value = nn.Linear(layer_size, 1)
        # weight_init([self.model, self.policy, self.value])

        self.actor_body = self._build_body()
        self.critic_body = self._build_body()

        self.policy_head = nn.Linear(layer_size, action_size)
        self.value_head = nn.Linear(layer_size, 1)

        weight_init(
            [
                self.actor_body,
                self.critic_body,
                self.policy_head,
                self.value_head,
            ]
        )

    def _build_body(self):
        """
         build body.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        layers = [nn.Linear(self.emb_state_size, self.layer_size)]
        if self.use_batchnorm:
            layers.append(SafeBatchNorm1d(self.layer_size))
        layers.append(nn.ReLU())

        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.layer_size, self.layer_size))
            if self.use_batchnorm:
                layers.append(SafeBatchNorm1d(self.layer_size))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, state):
        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        # if state.dim() == 1:
        #     state = state.unsqueeze(0)
        # elif state.dim() == 3:
        #     state = state.view(state.shape[0], -1)



        if state.dim() == 2:
            state = state.unsqueeze(0)

      

        B, L, F = state.shape

        infos_line = state[:, 0, :]
        role_line = state[:, 1, :]
        ff_state = state[:, 2:, :]

        attn_output = self.attention(ff_state)
        x_flat = attn_output.flatten(start_dim=1)

        infos_vec = self.infos_encoder(infos_line)
        role_vec = self.role_encoder(role_line)

        state = torch.cat([infos_vec, role_vec, x_flat], dim=1)

        actor_features = self.actor_body(state)
        critic_features = self.critic_body(state)

        policy_logits = self.policy_head(actor_features)
        value = self.value_head(critic_features).squeeze(-1)


        return policy_logits, value

class PPOActorCritic(nn.Module):
    """Actor-Critic model with independent dense networks for policy and value."""

    def __init__(
        """
          init  .
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        self,
        state_size,
        action_size,
        layer_size,
        seed,
        num_layers=8,
        use_batchnorm=True,
    ):
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm

        self.actor_body = self._build_body()
        self.critic_body = self._build_body()

        self.policy_head = nn.Linear(layer_size, action_size)
        self.value_head = nn.Linear(layer_size, 1)

        weight_init(
            [
                self.actor_body,
                self.critic_body,
                self.policy_head,
                self.value_head,
            ]
        )

    def _build_body(self):
        """
         build body.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """
        layers = [nn.Linear(self.state_size, self.layer_size)]
        if self.use_batchnorm:
            layers.append(SafeBatchNorm1d(self.layer_size))
        layers.append(nn.ReLU())

        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.layer_size, self.layer_size))
            if self.use_batchnorm:
                layers.append(SafeBatchNorm1d(self.layer_size))
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, state):

        """
        forward.
        
        Parameters
        ----------
        See function/class signature.
        
        Returns
        -------
        See implementation.
        """

        # print("dim", state.dim())
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() == 3:
            state = state.view(state.shape[0], -1)

        actor_features = self.actor_body(state)
        critic_features = self.critic_body(state)

        policy_logits = self.policy_head(actor_features)
        value = self.value_head(critic_features).squeeze(-1)

        return policy_logits, value

