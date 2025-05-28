import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PacmanDQN(nn.Module):
    """
    Deep Q-Network for Pacman
    Supports both MultiChannelObs (6 channels) and SingleChannelObs (1 channel)
    Uses CNN layers to process spatial game state
    """
    
    def __init__(self, observation_shape, num_actions=4, hidden_size=512):
        """
        Args:
            observation_shape: Tuple (channels, height, width) from gym observation space
            num_actions: Number of possible actions (4 for Pacman: w,a,s,d)
            hidden_size: Size of fully connected layer
        """
        super(PacmanDQN, self).__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        
        # Extract dimensions
        channels, height, width = observation_shape
        
        # Convolutional layers for spatial feature extraction
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate size after conv layers (no pooling, same padding)
        conv_output_size = self._get_conv_output_size(observation_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, num_actions)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def _get_conv_output_size(self, shape):
        """Calculate the output size after conv layers"""
        # Create dummy input to calculate output size
        dummy_input = torch.zeros(1, *shape)
        dummy_output = self._forward_conv(dummy_input)
        return dummy_output.view(1, -1).size(1)
    
    def _forward_conv(self, x):
        """Forward pass through convolutional layers only"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
        
        # Convolutional layers
        x = self._forward_conv(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output Q-values
        q_values = self.out(x)
        
        return q_values


class SimplePacmanDQN(nn.Module):
    """
    Simpler version that flattens observations instead of using CNN
    Good for initial testing or if CNN version is too complex
    """
    
    def __init__(self, observation_shape, num_actions=4, hidden_size=512):
        super(SimplePacmanDQN, self).__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        
        # Calculate input size (flatten all dimensions)
        input_size = np.prod(observation_shape)
        
        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.out = nn.Linear(hidden_size // 4, num_actions)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
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
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        q_values = self.out(x)
        return q_values


def create_dqn(observation_space, use_cnn=True, **kwargs):
    """
    Factory function to create appropriate DQN based on observation space
    
    Args:
        observation_space: Gym observation space
        use_cnn: Whether to use CNN (True) or simple FC layers (False)
        **kwargs: Additional arguments for network creation
    
    Returns:
        DQN model instance
    """
    observation_shape = observation_space.shape
    
    print(f"Creating DQN for observation shape: {observation_shape}")
    
    if use_cnn:
        print("Using CNN-based DQN")
        return PacmanDQN(observation_shape, **kwargs)
    else:
        print("Using simple fully-connected DQN")
        return SimplePacmanDQN(observation_shape, **kwargs)


# Utility functions for model operations
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
    # Test the models with different observation shapes
    print("Testing DQN models...")
    
    # Test with MultiChannelObs shape (6 channels)
    multi_shape = (6, 20, 20)  # Example shape
    print(f"\nTesting with MultiChannelObs shape: {multi_shape}")
    
    cnn_model = PacmanDQN(multi_shape)
    simple_model = SimplePacmanDQN(multi_shape)
    
    print(f"CNN model parameters: {get_model_size(cnn_model):,}")
    print(f"Simple model parameters: {get_model_size(simple_model):,}")
    
    # Test forward pass
    dummy_input = torch.randint(0, 256, (1, *multi_shape), dtype=torch.float)
    
    with torch.no_grad():
        cnn_output = cnn_model(dummy_input)
        simple_output = simple_model(dummy_input)
    
    print(f"CNN output shape: {cnn_output.shape}")
    print(f"Simple output shape: {simple_output.shape}")
    print(f"CNN Q-values: {cnn_output.squeeze().tolist()}")
    print(f"Simple Q-values: {simple_output.squeeze().tolist()}")
    
    # Test with SingleChannelObs shape (1 channel)
    single_shape = (1, 20, 20)
    print(f"\nTesting with SingleChannelObs shape: {single_shape}")
    
    single_cnn = PacmanDQN(single_shape)
    print(f"Single channel CNN parameters: {get_model_size(single_cnn):,}")
    
    print("\nAll tests passed!")