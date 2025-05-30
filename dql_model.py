import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PacmanDQN(nn.Module):
    """
    Simple and effective DQN for Pacman based on successful paper architectures
    Architecture: 64→128→128 conv layers + 512 FC layer
    Proven to work well for Pacman environments
    """
    
    def __init__(self, observation_shape, num_actions=4, hidden_size=512):
        """
        Args:
            observation_shape: Tuple (channels, height, width) from gym observation space
            num_actions: Number of possible actions (4 for Pacman: w,a,s,d)
            hidden_size: Size of fully connected layer (default 512 like in papers)
        """
        super(PacmanDQN, self).__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        
        # Extract dimensions
        channels, height, width = observation_shape
        
        # Convolutional layers inspired by successful Pacman papers
        # Layer 1: Basic feature detection (walls, dots, ghosts)
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1)
        
        # Layer 2: Spatial reduction + feature combination
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        
        # Layer 3: Further spatial reduction + high-level features
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2)
        
        # Calculate size after conv layers for FC layer
        conv_output_size = self._get_conv_output_size(observation_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_actions)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Created PacmanDQN with {total_params:,} parameters")
        
    def _get_conv_output_size(self, shape):
        """Calculate the output size after conv layers"""
        # Create dummy input to calculate output size
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self._forward_conv(dummy_input)
        return dummy_output.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers only"""
        x = F.relu(self.conv1(x))  # Extract basic features
        x = F.relu(self.conv2(x))  # Reduce spatial size, combine features  
        x = F.relu(self.conv3(x))  # Final feature extraction
        return x
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (channels, height, width) for single observation
        
        Returns:
            Q-values for each action
        """
        # Handle single observation (add batch dimension)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        # Normalize input to [0, 1] range if needed
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        # Convolutional feature extraction
        x = self._forward_conv(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected processing
        x = F.relu(self.fc1(x))
        
        # Output Q-values (no activation on final layer)
        q_values = self.out(x)
        
        return q_values


def create_dqn(observation_space, use_cnn=True, **kwargs):
    """
    Factory function to create PacmanDQN
    
    Args:
        observation_space: Gym observation space
        use_cnn: Whether to use CNN (True) or simple version (False)
        **kwargs: Additional arguments for network creation
    
    Returns:
        PacmanDQN model instance
    """
    observation_shape = observation_space.shape
    
    print(f"Creating DQN for observation shape: {observation_shape}")
    
    if use_cnn:
        print("Using CNN-based PacmanDQN")
        return PacmanDQN(observation_shape, **kwargs)
    else:
        # Simple fallback version that flattens everything
        print("Using simple fully-connected DQN")
        return SimplePacmanDQN(observation_shape, **kwargs)


class SimplePacmanDQN(nn.Module):
    """
    Simple fully-connected version for comparison/fallback
    """
    
    def __init__(self, observation_shape, num_actions=4, hidden_size=512):
        super(SimplePacmanDQN, self).__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        
        # Calculate input size (flatten all dimensions)
        input_size = np.prod(observation_shape)
        
        # Simple fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, num_actions)
        
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Created Simple PacmanDQN with {total_params:,} parameters")
    
    def forward(self, x):
        # Handle single observation (add batch dimension)
        if len(x.shape) == len(self.observation_shape):
            x = x.unsqueeze(0)
        
        # Normalize and flatten
        if x.max() > 1.0:
            x = x.float() / 255.0
        
        x = x.view(x.size(0), -1)
        
        # Forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.out(x)
        
        return q_values


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


if __name__ == "__main__":
    # Test the model with your actual observation shape
    print("Testing PacmanDQN...")
    
    # Your MultiChannelObs shape
    obs_shape = (6, 29, 25)
    print(f"\nTesting with observation shape: {obs_shape}")
    
    # Create CNN model
    cnn_model = PacmanDQN(obs_shape)
    
    # Create simple model for comparison
    simple_model = SimplePacmanDQN(obs_shape)
    
    print(f"\nCNN model parameters: {get_model_size(cnn_model):,}")
    print(f"Simple model parameters: {get_model_size(simple_model):,}")
    
    # Test forward pass with random input
    dummy_input = torch.randint(0, 256, (1, *obs_shape), dtype=torch.float)
    
    with torch.no_grad():
        cnn_output = cnn_model(dummy_input)
        simple_output = simple_model(dummy_input)
    
    print(f"\nCNN output shape: {cnn_output.shape}")
    print(f"CNN Q-values: {[f'{q:.4f}' for q in cnn_output.squeeze().tolist()]}")
    
    print(f"\nSimple output shape: {simple_output.shape}")
    print(f"Simple Q-values: {[f'{q:.4f}' for q in simple_output.squeeze().tolist()]}")
    
    # Test single observation (without batch dimension)
    single_obs = torch.randint(0, 256, obs_shape, dtype=torch.float)
    
    with torch.no_grad():
        single_output = cnn_model(single_obs)
    
    print(f"\nSingle observation test:")
    print(f"Input shape: {single_obs.shape}")
    print(f"Output shape: {single_output.shape}")
    print(f"Q-values: {[f'{q:.4f}' for q in single_output.squeeze().tolist()]}")
    
    print("\nAll tests passed!")
    print(f"\nArchitecture Summary:")
    print(f"   Conv1: {obs_shape[0]}→64 (3x3, stride=1)")
    print(f"   Conv2: 64→128 (5x5, stride=2)")  
    print(f"   Conv3: 128→128 (5x5, stride=2)")
    print(f"   FC1: {cnn_model._get_conv_output_size(obs_shape)}→512")
    print(f"   Output: 512→4")
    print(f"   Total: {get_model_size(cnn_model):,} parameters")