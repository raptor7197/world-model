"""
Interactive World Model - Train and Test Locally
This creates a world model that you can interact with in real-time!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

print("="*70)
print("INTERACTIVE WORLD MODEL")
print("Train a model to predict how CartPole behaves!")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Environment
    env_name = 'CartPole-v1'
    state_dim = 4
    action_dim = 1
    
    # Model architecture
    hidden_dim = 128
    
    # Training
    num_episodes = 50
    max_steps = 200
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 100
    
    # Visualization
    test_episodes = 3

config = Config()

# ============================================================================
# WORLD MODEL DEFINITION
# ============================================================================
class WorldModel(nn.Module):
    """
    A neural network that learns to predict:
    next_state = f(current_state, action)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(WorldModel, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, state_dim)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)
        next_state_pred = self.predictor(features)
        return next_state_pred
    
    def predict_trajectory(self, initial_state, actions):
        """Predict multiple steps ahead"""
        trajectory = [initial_state]
        state = initial_state
        
        with torch.no_grad():
            for action in actions:
                state = self.forward(state.unsqueeze(0), action.unsqueeze(0))
                trajectory.append(state.squeeze(0))
        
        return torch.stack(trajectory)


# ============================================================================
# DATA COLLECTION
# ============================================================================
def collect_data(env, num_episodes, max_steps):
    """Collect training data from the environment"""
    print(f"\n📊 Collecting data from {num_episodes} episodes...")
    
    states = []
    actions = []
    next_states = []
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Random policy for exploration
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            states.append(state)
            actions.append([float(action)])
            next_states.append(next_state)
            rewards.append(reward)
            
            episode_reward += reward
            state = next_state
            
            if terminated or truncated:
                break
        
        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{num_episodes} | Steps: {step+1} | Reward: {episode_reward:.0f}")
    
    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    
    print(f"\n✓ Collected {len(states):,} transitions")
    return states, actions, next_states


# ============================================================================
# TRAINING
# ============================================================================
def train_world_model(model, states, actions, next_states, config):
    """Train the world model"""
    print(f"\n🎓 Training world model for {config.num_epochs} epochs...")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    losses = []
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # Shuffle data
        indices = torch.randperm(len(states))
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(states), config.batch_size):
            batch_idx = indices[i:i+config.batch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_next_states = next_states[batch_idx]
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_states, batch_actions)
            loss = criterion(predictions, batch_next_states)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{config.num_epochs} | Loss: {avg_loss:.6f} | Best: {best_loss:.6f}")
    
    print(f"\n✓ Training complete! Final loss: {losses[-1]:.6f}")
    return losses


# ============================================================================
# VISUALIZATION
# ============================================================================
def visualize_predictions(model, env, num_episodes=3):
    """
    Run the environment and compare:
    - Actual trajectory (what really happened)
    - Predicted trajectory (what the model thought would happen)
    """
    print(f"\n🎬 Testing world model predictions...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        
        actual_trajectory = [state.copy()]
        predicted_trajectory = [state.copy()]
        
        actions_taken = []
        
        # Run episode
        for step in range(50):
            action = env.action_space.sample()
            actions_taken.append(action)
            
            # Get actual next state
            next_state, _, terminated, truncated, _ = env.step(action)
            actual_trajectory.append(next_state.copy())
            
            # Get predicted next state
            with torch.no_grad():
                state_tensor = torch.tensor([predicted_trajectory[-1]], dtype=torch.float32)
                action_tensor = torch.tensor([[float(action)]], dtype=torch.float32)
                pred_next = model(state_tensor, action_tensor).squeeze().numpy()
                predicted_trajectory.append(pred_next)
            
            if terminated or truncated:
                break
        
        # Convert to arrays
        actual_trajectory = np.array(actual_trajectory)
        predicted_trajectory = np.array(predicted_trajectory)
        
        # Calculate error
        mse = np.mean((actual_trajectory - predicted_trajectory) ** 2)
        
        print(f"\n  Episode {episode+1}:")
        print(f"    Steps taken: {len(actual_trajectory)-1}")
        print(f"    Prediction MSE: {mse:.6f}")
        
        # Plot comparison
        plot_trajectory_comparison(actual_trajectory, predicted_trajectory, episode+1)


def plot_trajectory_comparison(actual, predicted, episode_num):
    """Plot actual vs predicted trajectories"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Episode {episode_num}: Actual vs Predicted Trajectories', fontsize=14, fontweight='bold')
    
    labels = ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity']
    
    for i, (ax, label) in enumerate(zip(axes.flat, labels)):
        steps = range(len(actual))
        ax.plot(steps, actual[:, i], 'b-', label='Actual', linewidth=2, alpha=0.7)
        ax.plot(steps, predicted[:, i], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time Step')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate and show error
        error = np.mean((actual[:, i] - predicted[:, i])**2)
        ax.set_title(f'{label}\nMSE: {error:.4f}')
    
    plt.tight_layout()
    plt.savefig(f'trajectory_episode_{episode_num}.png', dpi=150, bbox_inches='tight')
    print(f"    Saved: trajectory_episode_{episode_num}.png")
    plt.close()


def plot_training_curve(losses):
    """Plot training loss over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('World Model Training Progress', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: training_curve.png")
    plt.close()


# ============================================================================
# INTERACTIVE TESTING
# ============================================================================
def interactive_test(model, env):
    """
    Let the model predict what happens with specific actions
    """
    print("\n" + "="*70)
    print("INTERACTIVE MODE: See what the model predicts!")
    print("="*70)
    
    state, _ = env.reset()
    print(f"\nInitial state:")
    print(f"  Position: {state[0]:7.4f}")
    print(f"  Velocity: {state[1]:7.4f}")
    print(f"  Angle:    {state[2]:7.4f}")
    print(f"  Ang.Vel:  {state[3]:7.4f}")
    
    # Test sequence of actions
    action_sequence = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]  # Sample action sequence
    
    print(f"\nAction sequence: {action_sequence}")
    print(f"(0 = Push left, 1 = Push right)")
    
    print("\n" + "-"*70)
    print("Step | Action | Actual Pos | Predicted Pos | Error")
    print("-"*70)
    
    current_state = state
    for step, action in enumerate(action_sequence):
        # Get actual next state
        actual_next, _, terminated, truncated, _ = env.step(action)
        
        # Get predicted next state
        with torch.no_grad():
            state_tensor = torch.tensor([current_state], dtype=torch.float32)
            action_tensor = torch.tensor([[float(action)]], dtype=torch.float32)
            pred_next = model(state_tensor, action_tensor).squeeze().numpy()
        
        # Show comparison
        error = abs(actual_next[0] - pred_next[0])
        print(f" {step+1:2d}  |   {action}    |   {actual_next[0]:7.4f}   |    {pred_next[0]:7.4f}    | {error:.4f}")
        
        current_state = actual_next
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step+1}")
            break
    
    print("-"*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("\n🚀 Starting World Model Training Pipeline\n")
    
    # Create environment
    env = gym.make(config.env_name)
    print(f"Environment: {config.env_name}")
    print(f"State dimension: {config.state_dim}")
    print(f"Action space: {env.action_space}")
    
    # Collect data
    states, actions, next_states = collect_data(env, config.num_episodes, config.max_steps)
    
    # Create and train model
    model = WorldModel(config.state_dim, config.action_dim, config.hidden_dim)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    losses = train_world_model(model, states, actions, next_states, config)
    
    # Plot training curve
    plot_training_curve(losses)
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        all_predictions = model(states, actions)
        final_mse = nn.MSELoss()(all_predictions, next_states)
        print(f"\nFinal evaluation MSE: {final_mse.item():.6f}")
    
    # Visualize predictions
    visualize_predictions(model, env, config.test_episodes)
    
    # Interactive testing
    interactive_test(model, env)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_mse': final_mse.item()
    }, 'world_model.pth')
    print(f"\n✓ Model saved to: world_model.pth")
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ ALL DONE!")
    print("="*70)
    print("\nGenerated files:")
    print("  • training_curve.png - Training progress")
    print("  • trajectory_episode_*.png - Prediction comparisons")
    print("  • world_model.pth - Trained model weights")
    print("\nThe model learned to predict CartPole dynamics!")
    print("You can now use this model for planning, simulation, or RL.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()