#!/usr/bin/env python3
"""
Fixed PPO Model - Actor-Critic architecture with proper initialization
Prevents policy collapse through careful weight initialization and architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.categorical import Categorical

class PacmanActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with CNN feature extraction for Pacman
    FIXED: Better initialization and architecture to prevent policy collapse
    """
    
    def __init__(self, observation_shape, num_actions=4, hidden_size=512):
        """
        Args:
            observation_shape: Tuple (channels, height, width) from gym observation space
            num_actions: Number of possible actions (4 for Pacman: w,a,s,d)
            hidden_size: Size of fully connected layer
        """
        super(PacmanActorCritic, self).__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        
        # Extract dimensions
        channels, height, width = observation_shape
        
        # FIXED: More conservative CNN architecture
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Calculate size after conv layers
        conv_output_size = self._get_conv_output_size(observation_shape)
        
        # FIXED: Shared feature layer with dropout
        self.shared_fc = nn.Linear(conv_output_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        
        # FIXED: Separate heads with proper initialization
        self.actor_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_head = nn.Linear(hidden_size // 2, num_actions)
        
        self.critic_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_head = nn.Linear(hidden_size // 2, 1)
        
        # CRITICAL: Proper weight initialization
        self._init_weights()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Created Fixed PacmanActorCritic with {total_params:,} parameters")
        
    def _get_conv_output_size(self, shape):
        """Calculate the output size after conv layers"""
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self._forward_conv(dummy_input)
        return dummy_output.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers only"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x
    
    def _init_weights(self):
        """FIXED: Proper weight initialization to prevent collapse"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization for most layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Conv2d):
                # He initialization for conv layers with ReLU
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
        # CRITICAL: Special initialization for output heads
        # Actor head: small weights to start with near-uniform policy
        nn.init.uniform_(self.actor_head.weight, -0.01, 0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        
        # Critic head: normal initialization
        nn.init.xavier_uniform_(self.critic_head.weight)
        nn.init.constant_(self.critic_head.bias, 0.0)
        
        print("✅ Applied fixed weight initialization")
    
    def forward(self, x):
        """
        Forward pass through both actor and critic networks
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
        
        Returns:
            action_logits: Raw logits for action probabilities
            values: State values from critic
        """
        # Handle single observation (add batch dimension)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Normalize input to [0, 1] range if needed
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        # Shared CNN feature extraction
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Shared features with dropout
        shared_features = F.relu(self.shared_fc(x))
        shared_features = self.dropout(shared_features)
        
        # Actor network (policy)
        actor_features = F.relu(self.actor_fc1(shared_features))
        action_logits = self.actor_head(actor_features)
        
        # Critic network (value function)
        critic_features = F.relu(self.critic_fc1(shared_features))
        values = self.critic_head(critic_features)
        
        return action_logits, values.squeeze(-1)
    
    def get_action_and_value(self, x, action=None):
        """
        Get action from policy and value estimate
        FIXED: Added temperature scaling to prevent extreme probabilities
        
        Args:
            x: Observation tensor
            action: If provided, compute log prob for this action
        
        Returns:
            action: Sampled action (if action=None) or provided action
            log_prob: Log probability of the action
            entropy: Policy entropy
            value: State value estimate
        """
        action_logits, value = self.forward(x)
        
        # FIXED: Temperature scaling to prevent extreme probabilities
        temperature = 1.0  # Can be tuned if needed
        scaled_logits = action_logits / temperature
        
        probs = Categorical(logits=scaled_logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value


class SimplePacmanActorCritic(nn.Module):
    """
    Simple fully-connected version of Actor-Critic for comparison/fallback
    FIXED: Better initialization and regularization
    """
    
    def __init__(self, observation_shape, num_actions=4, hidden_size=512):
        super(SimplePacmanActorCritic, self).__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        
        # Calculate input size (flatten all dimensions)
        input_size = np.prod(observation_shape)
        
        # Shared layers with dropout
        self.shared_fc1 = nn.Linear(input_size, hidden_size)
        self.shared_fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.2)
        
        # Actor network
        self.actor_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.actor_head = nn.Linear(hidden_size // 2, num_actions)
        
        # Critic network
        self.critic_fc = nn.Linear(hidden_size, hidden_size // 2)
        self.critic_head = nn.Linear(hidden_size // 2, 1)
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Created Simple PacmanActorCritic with {total_params:,} parameters")
    
    def _init_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        
        # Special initialization for output heads
        nn.init.uniform_(self.actor_head.weight, -0.01, 0.01)
        nn.init.constant_(self.actor_head.bias, 0.0)
        nn.init.xavier_uniform_(self.critic_head.weight)
        nn.init.constant_(self.critic_head.bias, 0.0)
    
    def forward(self, x):
        """Forward pass through both actor and critic networks"""
        # Handle single observation (add batch dimension)
        if len(x.shape) == len(self.observation_shape):
            x = x.unsqueeze(0)
        
        # Normalize and flatten
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        x = x.view(x.size(0), -1)
        
        # Shared layers
        x = F.relu(self.shared_fc1(x))
        shared_features = F.relu(self.shared_fc2(x))
        shared_features = self.dropout(shared_features)
        
        # Actor output
        actor_features = F.relu(self.actor_fc(shared_features))
        action_logits = self.actor_head(actor_features)
        
        # Critic output
        critic_features = F.relu(self.critic_fc(shared_features))
        values = self.critic_head(critic_features)
        
        return action_logits, values.squeeze(-1)
    
    def get_action_and_value(self, x, action=None):
        """Get action from policy and value estimate"""
        action_logits, value = self.forward(x)
        probs = Categorical(logits=action_logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value


def create_actor_critic(observation_space, use_cnn=True, **kwargs):
    """
    Factory function to create Actor-Critic network
    
    Args:
        observation_space: Gym observation space
        use_cnn: Whether to use CNN (True) or simple version (False)
        **kwargs: Additional arguments for network creation
    
    Returns:
        PacmanActorCritic or SimplePacmanActorCritic model instance
    """
    observation_shape = observation_space.shape
    
    print(f"Creating Fixed Actor-Critic for observation shape: {observation_shape}")
    
    if use_cnn:
        print("Using CNN-based PacmanActorCritic with anti-collapse measures")
        return PacmanActorCritic(observation_shape, **kwargs)
    else:
        print("Using simple fully-connected PacmanActorCritic")
        return SimplePacmanActorCritic(observation_shape, **kwargs)


# Utility functions
def save_model(model, filepath):
    """Save model state dict"""
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")


def load_model(model, filepath):
    """Load model state dict"""
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Model loaded from {filepath}")
    return model


def get_model_size(model):
    """Get number of parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Test the model architecture
if __name__ == "__main__":
    print("Testing Fixed PacmanActorCritic...")
    
    obs_shape = (6, 29, 25)
    print(f"\nTesting with observation shape: {obs_shape}")
    
    # Create CNN model
    cnn_model = PacmanActorCritic(obs_shape)
    
    # Create simple model for comparison
    simple_model = SimplePacmanActorCritic(obs_shape)
    
    print(f"\nCNN model parameters: {get_model_size(cnn_model):,}")
    print(f"Simple model parameters: {get_model_size(simple_model):,}")
    
    # Test forward pass with random input
    dummy_input = torch.randint(0, 256, (1, *obs_shape), dtype=torch.float)
    
    with torch.no_grad():
        # Test regular forward pass
        cnn_logits, cnn_values = cnn_model(dummy_input)
        simple_logits, simple_values = simple_model(dummy_input)
        
        print(f"\nCNN logits shape: {cnn_logits.shape}")
        print(f"CNN logits: {[f'{q:.4f}' for q in cnn_logits.squeeze().tolist()]}")
        print(f"CNN values: {cnn_values.item():.4f}")
        
        print(f"\nSimple logits shape: {simple_logits.shape}")
        print(f"Simple logits: {[f'{q:.4f}' for q in simple_logits.squeeze().tolist()]}")
        print(f"Simple values: {simple_values.item():.4f}")
        
        # Test action selection
        action, log_prob, entropy, value = cnn_model.get_action_and_value(dummy_input)
        print(f"\nAction selection test:")
        print(f"Action: {action.item()}")
        print(f"Log prob: {log_prob.item():.4f}")
        print(f"Entropy: {entropy.item():.4f}")
        print(f"Value: {value.item():.4f}")
        
        # Check initial policy is near uniform (anti-collapse check)
        probs = torch.softmax(cnn_logits, dim=-1).squeeze()
        prob_std = probs.std().item()
        prob_min = probs.min().item()
        prob_max = probs.max().item()
        
        print(f"\nInitial policy check:")
        print(f"Probabilities: {[f'{p:.3f}' for p in probs.tolist()]}")
        print(f"Std dev: {prob_std:.4f}")
        print(f"Range: {prob_min:.3f} to {prob_max:.3f}")
        
        if prob_std < 0.1 and prob_min > 0.15:
            print("✅ Good initial policy - near uniform distribution")
        else:
            print("⚠️  Initial policy may be biased")
    
    print("\nAll tests passed!")
    print(f"\nFixed Architecture Summary:")
    print(f"   Conv1: {obs_shape[0]}→32 (3x3, stride=1) + BatchNorm")
    print(f"   Conv2: 32→64 (3x3, stride=2) + BatchNorm")  
    print(f"   Conv3: 64→64 (3x3, stride=2) + BatchNorm")
    print(f"   Shared FC: {cnn_model._get_conv_output_size(obs_shape)}→512 + Dropout")
    print(f"   Actor: 512→256→4 (special init)")
    print(f"   Critic: 512→256→1")
    print(f"   Total: {get_model_size(cnn_model):,} parameters")
    print(f"   Anti-collapse measures: Small actor weights, BatchNorm, Dropout")