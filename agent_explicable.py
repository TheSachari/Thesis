"""
agent_explicable.py

Explainable reinforcement learning agent based on the Fully Parameterized Quantile Function (FQF) algorithm, with optional extensions such as Munchausen RL, Intrinsic Curiosity (ICM), n-step returns and Prioritized Experience Replay (PER).
"""
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import random
from networks import *
from ReplayBuffers import *
from IntrinsicCuriosityModule import *
from collective_functions import get_potential_actions


class FQF_Agent():
    """
    Fully Parameterized Quantile Function (FQF) agent for constrained decision-making.

    Learns a full return distribution and adapts quantile fractions dynamically.
    Designed for environments with a changing feasible action set.
    """
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 layer_type,
                 layer_size,
                 num_layers,
                 use_batchnorm,
                 AM,
                 n_steps,
                 batch_size,
                 buffer_size,
                 lr,
                 lr_dec,
                 tau,
                 gamma,
                 munchausen,
                 curiosity,
                 curiosity_size,
                 per,
                 rdm,
                 entropy_tau,
                 entropy_tau_coeff,
                 lo,
                 alpha ,
                 N,
                 entropy_coeff,
                 update_every,
                 max_train_steps,
                 decay_update,
                 device,
                 seed):
        """
        Initialize the explainable FQF agent.
        
        Builds networks, replay buffers and optional components (PER, ICM, Munchausen),
        and sets exploration and optimization parameters.
        """

        self.state_size = state_size
        self.action_size = action_size
        self.layer_type = layer_type
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm
        self.AM = AM
        self.seed = seed
        self.tseed = torch.manual_seed(seed)
        self.device = device
        self.tau = tau
        self.gamma = gamma
        self.update_every = update_every
        self.t_step = 0
        self.batch_size = batch_size
        self.Q_updates = 1 # to match with decay update
        self.n_steps = n_steps
        self.entropy_coeff = entropy_coeff
        self.N = N
        self.lr = lr
        self.lr_dec = lr_dec
        self.per = per
        self.rdm = rdm
        # munchausen params
        self.munchausen = munchausen
        self.curiosity = curiosity
        self.curiosity_size = curiosity_size
        self.eta = .1
        self.entropy_tau = entropy_tau
        self.entropy_tau_coeff = entropy_tau_coeff
        self.lo = lo
        self.alpha = alpha   
        self.max_train_steps = max_train_steps # 80k for 10k resp, for lr decay
        self.decay_update =  decay_update # Q updates % decay update => lr decay
        print("lr decay:", self.lr_dec, "decay_update:", self.decay_update, "PER", self.per)
        self.grad_clip = 1 #1, 10 ?

        # FQF-Network
        self.qnetwork_local = QVN(state_size, action_size,layer_size, AM, n_steps, device, seed, N, num_layers, layer_type, use_batchnorm).to(device)
        self.qnetwork_target = QVN(state_size, action_size,layer_size, AM, n_steps,device, seed, N, num_layers, layer_type, use_batchnorm).to(device)
        self.optimizer = optim.AdamW(self.qnetwork_local.parameters(), lr=lr)
        print(self.qnetwork_local)

        self.FPN = FPN(layer_size, seed, N, device).to(device)
        print(self.FPN)
        self.frac_optimizer = optim.RMSprop(self.FPN.parameters(), lr=lr*0.000001, alpha=0.95, eps=0.00001)
        
        # Replay memory Standard (random simple)
        if self.per == 0:
            self.memory = ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps, rdm)
        # Replay memory PER
        elif self.per == 1:
            self.memory = PrioritizedReplay(buffer_size, batch_size, seed, gamma, n_steps)
        # Replay memory PER Sum Tree
        elif self.per == 2:
            self.memory = N_Steps_Prioritized_ReplayBuffer(buffer_size, batch_size, seed, gamma, n_steps)
            
        # Curiosity
        if self.curiosity != 0:
            inverse_m = Inverse(self.state_size, self.action_size, self.curiosity_size)
            forward_m = Forward(self.state_size, self.action_size, inverse_m.calc_input_layer(), device=device)
            self.ICM = ICM(inverse_m, forward_m).to(device)
            print(inverse_m, forward_m)

    def step(self, state, action, reward, next_state, done):
        
        state = torch.from_numpy(state).float()
        next_state = torch.from_numpy(next_state).float() 
        
        # Save experience in replay memory

        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        
        if (self.t_step) % self.update_every == 0:

            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample() 

                if self.per == 0:
                    loss, entropy, icm_loss = self.learn(experiences)
                else:
                    loss, entropy = self.learn_per(experiences)
                self.Q_updates += 1

                return loss.item()

        else:
            return None

                
    def act(self, state, all_ff_waiting, eps=0.):
        """
        Record a transition and trigger learning when appropriate.
        
        Stores the transition in replay memory, manages n-step return aggregation,
        and periodically calls `learn()` to update the networks.
        """

        potential_actions, potential_skills = get_potential_actions(state, all_ff_waiting)


        state = torch.from_numpy(state.flatten()).float().to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            embedding = self.qnetwork_local.forward(state)
            taus, taus_, _ = self.FPN(embedding)
            F_Z = self.qnetwork_local.get_quantiles(state, taus_, embedding)
            q = ((taus[:, 1:].unsqueeze(-1) - taus[:, :-1].unsqueeze(-1)) * F_Z).sum(1)                   
        self.qnetwork_local.train()
        
        q_list = q.cpu().numpy().flatten().tolist()   
        action = filter_q_values(q_list, potential_actions)


        

        skill_lvl = potential_skills[potential_actions.index(action)]

        return action, skill_lvl, potential_actions
        
    def learn(self, experiences):
        
        states, actions, rewards, next_states, dones = experiences
        
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        
        embedding = self.qnetwork_local.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())

        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N, 1))
        assert Q_expected.shape == (self.batch_size, self.N, 1)
        
        # calc fractional loss 
        with torch.no_grad():
        """
        Update the agent networks using a batch sampled from replay memory.
        
        Computes distributional Bellman targets (FQF), applies optional Munchausen
        reward shaping and intrinsic curiosity rewards, and performs a gradient step.
        """
            F_Z_tau = self.qnetwork_local.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N-1, 1))
            
        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus)
        entropy_loss = self.entropy_coeff * entropy.mean() 
        frac_loss += entropy_loss
        
        # calculate icm loss
        if self.curiosity != 0:
            
            forward_pred_err, inverse_pred_err = self.ICM.calc_errors(state1=states, state2=next_states, action=actions)
            r_i = self.eta * forward_pred_err
            assert r_i.shape == rewards.shape, "r_ and r_e have not the same shape"
            if self.curiosity == 1:
                rewards += r_i.detach()
            else:
                rewards = r_i.detach()        
        icm_loss = 0
        if self.curiosity != 0:
            icm_loss = self.ICM.update_ICM(forward_pred_err, inverse_pred_err)
        
        # Calculate Q_targets without munchausen 
        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            with torch.no_grad():
                
                next_state_embedding_loc = self.qnetwork_local.forward(next_states)  
                n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
                F_Z_next = self.qnetwork_local.get_quantiles(next_states, n_taus_, next_state_embedding_loc)  
                Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))*F_Z_next).sum(1)
                action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)
                
                next_state_embedding = self.qnetwork_target.forward(next_states)
                F_Z_next = self.qnetwork_target.get_quantiles(next_states, taus_, next_state_embedding)
                Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.batch_size, self.N, 1)).transpose(1,2)
                Q_targets = rewards.unsqueeze(-1) + (self.gamma**self.n_steps * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
  
        
        # Calculate Q_targets with munchausen
        else:
            ns_embedding = self.qnetwork_target.forward(next_states).detach()
            ns_taus, ns_taus_, ns_entropy = self.FPN(ns_embedding.detach())
            ns_taus = ns_taus.detach()

            ns_entropy = ns_entropy.detach()
            m_quantiles = self.qnetwork_target.get_quantiles(next_states, ns_taus_, ns_embedding).detach()
            m_Q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(1)
            # calculate log-pi 
            logsum = torch.logsumexp(\
                (m_Q - m_Q.max(1)[0].unsqueeze(-1))/(ns_entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (m_Q - m_Q.max(1)[0].unsqueeze(-1) - (ns_entropy*self.entropy_tau_coeff).mean().detach()*logsum).unsqueeze(1)
            
            pi_target = F.softmax(m_Q/(ns_entropy*self.entropy_tau_coeff).mean().detach(), dim=1).unsqueeze(1) 
            Q_target = (self.gamma**self.n_steps * (pi_target * (m_quantiles-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.batch_size, 1, self.N)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(states, taus_, embedding).detach()
            m_Q_targets = ((taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach()) * m_quantiles_targets).sum(1)
            v_k_target = m_Q_targets.max(1)[0].unsqueeze(-1) 
            tau_log_pik = m_Q_targets - v_k_target - (entropy*self.entropy_tau_coeff).mean().detach()*torch.logsumexp(\
                                                                    (m_Q_targets - v_k_target)/(entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1)
            assert tau_log_pik.shape == (self.batch_size, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.batch_size, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target
            
        
        # Compute standard loss
        # loss = F.mse_loss(Q_expected, Q_targets) 
        
        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        
        # assert td_error.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)

        quantil_l = abs(taus_.unsqueeze(-1) -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1) 
        loss = loss.mean()

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()
        
        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),1)
        if self.lr_dec != 0:
            self.optimizer.step()
        

        # Minimize standard loss
        # loss.backward()
        # clip_grad_norm_(self.qnetwork_local.parameters(),1)
        # self.optimizer.step()



        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        if (self.Q_updates % self.decay_update == 0):

            print("update lr decay")

            if self.lr_dec == 0:
                self.lr_decay_0()
            elif self.lr_dec == 1:
                self.lr_decay_1() 
            elif self.lr_dec == 2:
                self.lr_decay_2()
            elif self.lr_dec == 3:
                self.lr_decay_3()
                
        return loss.detach().cpu().numpy(),  entropy.mean().detach().cpu().numpy(), icm_loss

    def learn_per(self, experiences):

        states, actions, rewards, next_states, dones, idx, weights = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1) 
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        embedding = self.qnetwork_local.forward(states)
        taus, taus_, entropy = self.FPN(embedding.detach())
        
        # Get expected Q values from local model
        F_Z_expected = self.qnetwork_local.get_quantiles(states, taus_, embedding)
        Q_expected = F_Z_expected.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N, 1))
        assert Q_expected.shape == (self.batch_size, self.N, 1)
        # calc fractional loss
        with torch.no_grad():
            F_Z_tau = self.qnetwork_local.get_quantiles(states, taus[:, 1:-1], embedding.detach())
            FZ_tau = F_Z_tau.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.N-1, 1))
            
        frac_loss = calc_fraction_loss(Q_expected.detach(), FZ_tau, taus, weights)
        entropy_loss = self.entropy_coeff * entropy.mean() 
        frac_loss += entropy_loss

        if not self.munchausen:
            # Get max predicted Q values (for next states) from target model
            with torch.no_grad():
                
                next_state_embedding_loc = self.qnetwork_local.forward(next_states)  
                n_taus, n_taus_, _ = self.FPN(next_state_embedding_loc)
                F_Z_next = self.qnetwork_local.get_quantiles(next_states, n_taus_, next_state_embedding_loc)  
                Q_targets_next = ((n_taus[:, 1:].unsqueeze(-1) - n_taus[:, :-1].unsqueeze(-1))*F_Z_next).sum(1)
                action_indx = torch.argmax(Q_targets_next, dim=1, keepdim=True)
                
                next_state_embedding = self.qnetwork_target.forward(next_states)
                F_Z_next = self.qnetwork_target.get_quantiles(next_states, taus_, next_state_embedding)
                Q_targets_next = F_Z_next.gather(2, action_indx.unsqueeze(-1).expand(self.batch_size, self.N, 1)).transpose(1,2)
                Q_targets = rewards.unsqueeze(-1) + (self.gamma**self.n_steps * Q_targets_next.to(self.device) * (1. - dones.unsqueeze(-1)))
        else:
            ns_embedding = self.qnetwork_target.forward(next_states).detach()
            ns_taus, ns_taus_, ns_entropy = self.FPN(ns_embedding.detach())
            ns_taus = ns_taus.detach()

            ns_entropy = ns_entropy.detach()
            m_quantiles = self.qnetwork_target.get_quantiles(next_states, ns_taus_, ns_embedding).detach()
            m_Q = ((ns_taus[:, 1:].unsqueeze(-1) - ns_taus[:, :-1].unsqueeze(-1)) * m_quantiles).sum(1)
            # calculate log-pi 
            logsum = torch.logsumexp(\
                (m_Q - m_Q.max(1)[0].unsqueeze(-1))/(ns_entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1) #logsum trick
            assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (m_Q - m_Q.max(1)[0].unsqueeze(-1) - (ns_entropy*self.entropy_tau_coeff).mean().detach()*logsum).unsqueeze(1)
            
            pi_target = F.softmax(m_Q/(ns_entropy*self.entropy_tau_coeff).mean().detach(), dim=1).unsqueeze(1) 
            Q_target = (self.gamma**self.n_steps * (pi_target * (m_quantiles-tau_log_pi_next)*(1 - dones.unsqueeze(-1))).sum(2)).unsqueeze(1)
            assert Q_target.shape == (self.batch_size, 1, self.N)

            m_quantiles_targets = self.qnetwork_local.get_quantiles(states, taus_, embedding).detach()
            m_Q_targets = ((taus[:, 1:].unsqueeze(-1).detach() - taus[:, :-1].unsqueeze(-1).detach()) * m_quantiles_targets).sum(1)
            v_k_target = m_Q_targets.max(1)[0].unsqueeze(-1) 
            tau_log_pik = m_Q_targets - v_k_target - (entropy*self.entropy_tau_coeff).mean().detach()*torch.logsumexp(\
                                                                    (m_Q_targets - v_k_target)/(entropy*self.entropy_tau_coeff).mean().detach(), 1).unsqueeze(-1)
            assert tau_log_pik.shape == (self.batch_size, self.action_size), "shape instead is {}".format(tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)
            
            # calc munchausen reward:
            munchausen_reward = (rewards + self.alpha*torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            assert munchausen_reward.shape == (self.batch_size, 1, 1)
            # Compute Q targets for current states 
            Q_targets = munchausen_reward + Q_target


        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        # assert td_error.shape == (self.batch_size, self.N, self.N), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0)
        quantil_l = abs(taus_.unsqueeze(-1) -(td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True) * weights
        loss = loss.mean()
        

        # Minimize the frac loss
        self.frac_optimizer.zero_grad()
        frac_loss.backward(retain_graph=True)
        self.frac_optimizer.step()
        
        # Minimize the huber loss
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(),self.grad_clip)
        if self.lr_dec != 0:
            self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        
        if (self.Q_updates % self.decay_update == 0):

            print("update lr decay")

            if self.lr_dec == 0:
                self.lr_decay_0()
            elif self.lr_dec == 1:
                self.lr_decay_1() 
            elif self.lr_dec == 2:
                self.lr_decay_2()
            elif self.lr_dec == 3:
                self.lr_decay_3()
        
        # update priorities
        td_error = td_error.sum(dim=1).mean(dim=1,keepdim=True) # not sure about this -> test 
        
        self.memory.update_priorities(idx, td_error.data.cpu().numpy())

                
        return loss.detach().cpu().numpy(), entropy.mean().detach().cpu().numpy()    

    def soft_update(self, local_model, target_model):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

    def lr_decay_0(self):
        for p in self.optimizer.param_groups:
            lr_now = p['lr']
        print("step", self.t_step, "current lr :", lr_now)
        
    def lr_decay_1(self):
        lr_now = 0.9 * self.lr * (1 - self.t_step / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
        print("step", self.t_step, "current lr :", lr_now)

    def lr_decay_2(self):
        if (self.t_step % 5000 == 0):
            self.lr = self.lr / 2
            for p in self.optimizer.param_groups:
                p['lr'] = self.lr    
        print("step", self.t_step, "current lr :", self.lr)

    def lr_decay_3(self):
        self.lr = self.lr / 2
        for p in self.optimizer.param_groups:
            p['lr'] = self.lr      
        print("step", self.t_step, "current lr :", self.lr)

    
def calc_fraction_loss(FZ_,FZ, taus, weights=None):
    """calculate the loss for the fraction proposal network """
    
    gradients1 = FZ - FZ_[:, :-1]
    gradients2 = FZ - FZ_[:, 1:] 
    flag_1 = FZ > torch.cat([FZ_[:, :1], FZ[:, :-1]], dim=1)
    flag_2 = FZ < torch.cat([FZ[:, 1:], FZ_[:, -1:]], dim=1)
    gradients = (torch.where(flag_1, gradients1, - gradients1) + torch.where(flag_2, gradients2, -gradients2)).view(taus.shape[0], 31)
    assert not gradients.requires_grad
    if weights != None:
        loss = ((gradients * taus[:, 1:-1]).sum(dim=1)*weights).mean()
    else:
        loss = (gradients * taus[:, 1:-1]).sum(dim=1).mean()
    return loss 
    
def calculate_huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], 32, 32), "huber loss has wrong shape"
    return loss

def filter_q_values(q_list, potential_actions):
    if potential_actions != [79]:

        dic_q = {k:q_list[k] for k in potential_actions} # dic potential action : q value  

        return max(dic_q, key=dic_q.get) # action with the highest q value
    else:
        return 79
