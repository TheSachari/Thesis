"""
IntrinsicCuriosityModule.py

Intrinsic Curiosity Module (ICM) components for curiosity-driven exploration in RL.

This file defines three PyTorch modules:
- `Inverse`: inverse dynamics model (predicts action from (s, s'))
- `Forward`: forward dynamics model (predicts next-state features from (phi(s), a))
- `ICM`: wrapper that computes intrinsic prediction errors and performs joint updates

The implementation follows the classic ICM idea:
    Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction" (2017)

Key details in this implementation
---------------------------------
- States are embedded with an encoder in the `Inverse` model (`Inverse.encoder`).
- The `Forward` model expects (encoded_state, action) and returns predicted encoded_next_state.
- `ICM.calc_errors` returns per-sample forward and inverse prediction errors.
- `ICM.update_ICM` performs a gradient step on the weighted sum of the two losses.

Import safety
-------------
This module defines classes only and has no top-level execution logic. It is safe to import.

Notes / caveats
---------------
- Several methods assert CUDA usage (e.g., `Forward.forward` and `ICM.calc_errors`). If you want
  CPU support, remove or relax these asserts and ensure tensors are moved to the chosen device.
- `Inverse.forward` applies softmax and the `ICM` uses `CrossEntropyLoss` on those outputs.
  In PyTorch, `CrossEntropyLoss` expects **raw logits**, not probabilities. To be strictly correct,
  you should remove the softmax in `Inverse.forward` (or switch to `NLLLoss` on log-probabilities).
  This refactor keeps behavior unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class Inverse(nn.Module):
    """
    Inverse dynamics model for ICM.

    The inverse model predicts the action taken between two consecutive states.
    It also owns an `encoder` that maps raw states into a feature space `phi(s)`,
    which is reused by the forward model.

    Parameters
    ----------
    state_size : int
        Flattened dimension of the environment state.
    action_size : int
        Number of discrete actions.
    curiosity_size : int
        Feature dimension used by the encoder (phi).

    Attributes
    ----------
    encoder : nn.Sequential
        State encoder producing `curiosity_size`-dim features.
    layer1, layer2 : nn.Linear
        MLP layers used to predict action probabilities.
    softmax : nn.Softmax
        Softmax over actions (kept for behavior compatibility; see module notes).
    """

    def __init__(self, state_size: int, action_size: int, curiosity_size: int):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.curiosity_size = curiosity_size
        self.hidden_size = self.curiosity_size * 2

        # State encoder phi(s)
        self.encoder = nn.Sequential(
            nn.Linear(self.state_size, self.curiosity_size),
            nn.ELU(),
        )
        self.layer1 = nn.Linear(2 * self.curiosity_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.action_size)
        self.softmax = nn.Softmax(dim=1)

    def calc_input_layer(self) -> int:
        """
        Compute the encoder output dimension for a dummy zero state.

        Returns
        -------
        int
            Flattened encoder output size.
        """
        x = torch.zeros(self.state_size).unsqueeze(0)
        x = self.encoder(x)
        return int(x.flatten().shape[0])

    def forward(self, enc_state: torch.Tensor, enc_state1: torch.Tensor) -> torch.Tensor:
        """
        Predict the action distribution given encoded states.

        Parameters
        ----------
        enc_state : torch.Tensor
            Encoded current state phi(s), shape (B, curiosity_size).
        enc_state1 : torch.Tensor
            Encoded next state phi(s'), shape (B, curiosity_size).

        Returns
        -------
        torch.Tensor
            Action probabilities, shape (B, action_size).

        Notes
        -----
        Returns probabilities (softmax). The training loss in `ICM` uses CrossEntropyLoss,
        which typically expects logits; this is kept unchanged for compatibility.
        """
        x = torch.cat((enc_state, enc_state1), dim=1)
        x = torch.relu(self.layer1(x))
        x = self.softmax(self.layer2(x))
        return x


class Forward(nn.Module):
    """
    Forward dynamics model for ICM.

    Predicts next-state features phi(s') given current features phi(s) and action a.

    Parameters
    ----------
    state_size : int
        Unused in the current implementation (kept for API compatibility).
    action_size : int
        Number of discrete actions.
    output_size : int
        Feature dimension of phi(s) (must match the inverse encoder output).
    hidden_size : int, default=256
        Hidden dimension of the forward MLP.
    device : str, default="cuda:0"
        Device string used for one-hot action tensors and internal assertions.

    Notes
    -----
    - Actions are one-hot encoded and concatenated with the state embedding.
    - The method asserts CUDA usage (behavior preserved).
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        output_size: int,
        hidden_size: int = 256,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.action_size = action_size
        self.device = device
        self.forwardM = nn.Sequential(
            nn.Linear(output_size + self.action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next-state feature embedding.

        Parameters
        ----------
        state : torch.Tensor
            Encoded current state phi(s), shape (B, output_size).
        action : torch.Tensor
            Discrete action indices, shape (B, 1) or (B,).

        Returns
        -------
        torch.Tensor
            Predicted next embedding phi_hat(s'), shape (B, output_size).
        """
        ohe_action = torch.zeros(action.shape[0], self.action_size).to(self.device)
        indices = torch.stack(
            (torch.arange(action.shape[0]).to(self.device), action.squeeze().long()),
            dim=0,
        )
        indices = indices.tolist()
        ohe_action[indices] = 1.0

        x = torch.cat((state, ohe_action), dim=1)
        assert x.device.type == "cuda"
        return self.forwardM(x)


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module wrapper.

    Combines an inverse and forward model, computes forward/inverse prediction errors,
    and performs joint optimization of both networks.

    Parameters
    ----------
    inverse_model : Inverse
        Inverse dynamics model containing the shared encoder.
    forward_model : Forward
        Forward dynamics model in feature space.
    learning_rate : float, default=1e-3
        Optimizer learning rate (note: the original code uses a fixed 1e-3 internally).
    lambda_ : float, default=0.1
        Unused coefficient (kept for API compatibility).
    beta : float, default=0.2
        Weighting between inverse and forward loss: loss = (1-beta)*L_inv + beta*L_fwd.
    device : str, default="cuda:0"
        Device used to move submodules.

    Attributes
    ----------
    forward_loss : nn.MSELoss
        Per-sample MSE (reduction='none').
    inverse_loss : nn.CrossEntropyLoss
        Per-sample CE (reduction='none').
    optimizer : torch.optim.Optimizer
        Adam optimizer over parameters of both models.
    """

    def __init__(
        self,
        inverse_model: Inverse,
        forward_model: Forward,
        learning_rate: float = 1e-3,
        lambda_: float = 0.1,
        beta: float = 0.2,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.inverse_model = inverse_model.to(device)
        self.forward_model = forward_model.to(device)

        self.forward_scale = 1.0
        self.inverse_scale = 1e4
        self.lr = learning_rate
        self.beta = beta
        self.lambda_ = lambda_

        self.forward_loss = nn.MSELoss(reduction="none")
        self.inverse_loss = nn.CrossEntropyLoss(reduction="none")
        # Behavior preserved: lr fixed to 1e-3 (ignores `learning_rate` argument in original code).
        self.optimizer = optim.Adam(
            list(self.forward_model.parameters()) + list(self.inverse_model.parameters()),
            lr=1e-3,
        )

    def calc_errors(
        self,
        state1: torch.Tensor,
        state2: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-sample forward and inverse prediction errors.

        Parameters
        ----------
        state1 : torch.Tensor
            Current states, shape (B, ...) and on CUDA (asserted).
        state2 : torch.Tensor
            Next states, shape (B, ...) and on CUDA (asserted).
        action : torch.Tensor
            Actions taken, shape (B, 1) or (B,) and on CUDA (asserted).

        Returns
        -------
        forward_pred_err : torch.Tensor
            Per-sample forward error, shape (B, 1).
        inverse_pred_err : torch.Tensor
            Per-sample inverse error, shape (B, 1).
        """
        assert (
            state1.device.type == "cuda"
            and state2.device.type == "cuda"
            and action.device.type == "cuda"
        )

        state1 = state1.view(state1.shape[0], -1)
        state2 = state2.view(state1.shape[0], -1)

        enc_state1 = self.inverse_model.encoder(state1)
        enc_state2 = self.inverse_model.encoder(state2)

        forward_pred = self.forward_model(enc_state1.detach(), action)
        forward_pred_err = (
            1 / 2 * self.forward_loss(forward_pred, enc_state2.detach()).sum(dim=1).unsqueeze(dim=1)
        )

        pred_action = self.inverse_model(enc_state1, enc_state2)
        inverse_pred_err = self.inverse_loss(pred_action, action.flatten().long()).unsqueeze(dim=1)

        return forward_pred_err, inverse_pred_err

    def update_ICM(self, forward_err: torch.Tensor, inverse_err: torch.Tensor) -> float:
        """
        Update ICM parameters from error terms.

        Parameters
        ----------
        forward_err : torch.Tensor
            Per-sample forward errors, shape (B, 1).
        inverse_err : torch.Tensor
            Per-sample inverse errors, shape (B, 1).

        Returns
        -------
        float
            Scalar loss value (mean over batch) as a Python float.
        """
        self.optimizer.zero_grad()
        loss = ((1.0 - self.beta) * inverse_err + self.beta * forward_err).mean()
        loss.backward(retain_graph=True)

        clip_grad_norm_(self.inverse_model.parameters(), 1)
        clip_grad_norm_(self.forward_model.parameters(), 1)
        self.optimizer.step()
        return float(loss.detach().cpu().numpy())
