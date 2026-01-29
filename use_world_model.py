"""
Load and Use a Trained World Model
Run this AFTER training to load your saved model and use it
"""

import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np

# ============================================================================
# WORLD MODEL (same architecture as training)
# ============================================================================
class WorldModel(nn.Module):
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


# ============================================================================
# LOAD MODEL
# ============================================================================
def load_model(model_path='world_model.pth'):
    """Load a trained world model"""
    print(f"Loading model from {model_path}...")
    
    checkpoint = torch.load(model_path)
    
    # Create model
    model = WorldModel(state_dim=4, action_dim=1, hidden_dim=128)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully!")
    print(f"  Final MSE: {checkpoint['final_mse']:.6f}")
    
    return model


# ============================================================================
# USE THE MODEL
# ============================================================================
def simulate_with_model(model, initial_state, actions):
    """
    Use the world model to simulate what would happen
    without actually running the environment!
    """
    print(f"\n🔮 Simulating {len(actions)} steps with world model...")
    
    trajectory = [initial_state]
    state = torch.tensor([initial_state], dtype=torch.float32)
    
    with torch.no_grad():
        for i, action in enumerate(actions):
            action_tensor = torch.tensor([[float(action)]], dtype=torch.float32)
            next_state = model(state, action_tensor)
            
            trajectory.append(next_state.squeeze().numpy())
            state = next_state
            
            if (i + 1) % 5 == 0:
                pos = next_state[0, 0].item()
                angle = next_state[0, 2].item()
                print(f"  Step {i+1}: Position={pos:7.4f}, Angle={angle:7.4f}")
    
    return np.array(trajectory)


def compare_simulation_vs_reality(model, env, num_tests=3):
    """
    Compare model's predictions with what actually happens
    """
    print(f"\n🔬 Running {num_tests} comparison tests...\n")
    
    for test in range(num_tests):
        state, _ = env.reset()
        
        # Generate random action sequence
        num_steps = 20
        actions = [env.action_space.sample() for _ in range(num_steps)]
        
        # Simulate with model
        simulated = simulate_with_model(model, state, actions)
        
        # Run in actual environment
        actual = [state]
        current_state = state
        for action in actions:
            next_state, _, terminated, truncated, _ = env.step(action)
            actual.append(next_state)
            current_state = next_state
            if terminated or truncated:
                break
        
        actual = np.array(actual)
        
        # Compare
        min_len = min(len(simulated), len(actual))
        mse = np.mean((simulated[:min_len] - actual[:min_len])**2)
        
        print(f"\nTest {test+1}:")
        print(f"  Steps completed: {min_len-1}")
        print(f"  Prediction MSE: {mse:.6f}")
        print(f"  Position error: {np.mean((simulated[:min_len, 0] - actual[:min_len, 0])**2):.6f}")
        print(f"  Angle error: {np.mean((simulated[:min_len, 2] - actual[:min_len, 2])**2):.6f}")


def plan_best_action(model, state, lookahead_steps=5):
    """
    Use the world model to plan ahead!
    Try different actions and pick the one that keeps the pole upright longest
    """
    print(f"\n🧠 Planning best action with {lookahead_steps}-step lookahead...")
    
    best_action = None
    best_score = float('-inf')
    
    # Try both actions
    for action in [0, 1]:
        # Simulate this action repeated multiple times
        actions = [action] * lookahead_steps
        simulated = simulate_with_model(model, state, actions)
        
        # Score: prefer small angles (pole upright)
        angle_penalty = np.sum(np.abs(simulated[:, 2]))  # Sum of absolute angles
        score = -angle_penalty
        
        print(f"  Action {action} ({'left' if action == 0 else 'right'}): score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_action = action
    
    print(f"  → Best action: {best_action} ({'left' if best_action == 0 else 'right'})")
    return best_action


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*70)
    print("WORLD MODEL - INFERENCE MODE")
    print("="*70)
    
    # Load model
    try:
        model = load_model('world_model.pth')
    except FileNotFoundError:
        print("\n❌ Error: world_model.pth not found!")
        print("Please run interactive_world_model.py first to train a model.")
        return
    
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Test 1: Simple simulation
    print("\n" + "="*70)
    print("TEST 1: Pure Simulation (no environment)")
    print("="*70)
    state, _ = env.reset()
    actions = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0]
    trajectory = simulate_with_model(model, state, actions)
    print(f"\n✓ Simulated {len(trajectory)-1} steps successfully!")
    
    # Test 2: Compare with reality
    print("\n" + "="*70)
    print("TEST 2: Simulation vs Reality")
    print("="*70)
    compare_simulation_vs_reality(model, env, num_tests=3)
    
    # Test 3: Planning
    print("\n" + "="*70)
    print("TEST 3: Model-Based Planning")
    print("="*70)
    state, _ = env.reset()
    print(f"\nCurrent state:")
    print(f"  Position: {state[0]:7.4f}")
    print(f"  Velocity: {state[1]:7.4f}")
    print(f"  Angle:    {state[2]:7.4f}")
    print(f"  Ang.Vel:  {state[3]:7.4f}")
    
    best_action = plan_best_action(model, state, lookahead_steps=10)
    
    # Test 4: Model-based control
    print("\n" + "="*70)
    print("TEST 4: Using Model for Control")
    print("="*70)
    print("\nRunning episode with model-based planning...")
    
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(100):
        # Use model to choose action
        action = plan_best_action(model, state, lookahead_steps=5)
        
        # Take action in real environment
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        if step % 20 == 0:
            print(f"  Step {step}: Reward so far: {total_reward}")
        
        if terminated or truncated:
            break
    
    print(f"\n✓ Episode complete!")
    print(f"  Total steps: {step+1}")
    print(f"  Total reward: {total_reward}")
    
    env.close()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE!")
    print("="*70)
    print("\nYour world model can:")
    print("  • Simulate future states without running the environment")
    print("  • Predict outcomes of action sequences")  
    print("  • Plan ahead to choose better actions")
    print("  • Be used for model-based reinforcement learning")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()