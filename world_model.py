"""
World Models in PyTorch - Complete Working Implementation
Based on: https://www.codegenes.net/blog/world-models-pytorch/

This script demonstrates:
1. Simple deterministic world model
2. Data collection from CartPole environment
3. Training the world model
4. Model evaluation
5. Regularized world model with dropout
6. Model ensembling for improved predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("World Models in PyTorch - Demo")
print("=" * 60)

# ============================================================================
# 1. SIMPLE WORLD MODEL DEFINITION
# ============================================================================

class SimpleWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SimpleWorldModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, state_dim)
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        next_state = self.fc3(x)
        return next_state


print("\n1. Defining Simple World Model...")
# Test with dummy data
state_dim = 4  # CartPole has 4 state dimensions
action_dim = 1  # CartPole has 2 actions, but we'll use 1D encoding

world_model = SimpleWorldModel(state_dim, action_dim)
print(f"   Model created with {sum(p.numel() for p in world_model.parameters())} parameters")

# Test forward pass
test_state = torch.randn(1, state_dim)
test_action = torch.randn(1, action_dim)
test_output = world_model(test_state, test_action)
print(f"   Test forward pass successful: input shape {test_state.shape} -> output shape {test_output.shape}")


# ============================================================================
# 2. DATA COLLECTION FROM ENVIRONMENT
# ============================================================================

print("\n2. Collecting data from CartPole environment...")

env = gym.make('CartPole-v1')

states = []
actions = []
next_states = []

num_episodes = 50
max_steps_per_episode = 200

for episode in range(num_episodes):
    state, info = env.reset()
    
    for step in range(max_steps_per_episode):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        
        states.append(state)
        actions.append([float(action)])  # Convert discrete action to float
        next_states.append(next_state)
        
        state = next_state
        
        if terminated or truncated:
            break

print(f"   Collected {len(states)} transitions")

# Convert to tensors
states = torch.tensor(states, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.float32)
next_states = torch.tensor(next_states, dtype=torch.float32)

print(f"   States shape: {states.shape}")
print(f"   Actions shape: {actions.shape}")
print(f"   Next states shape: {next_states.shape}")


# ============================================================================
# 3. TRAINING THE WORLD MODEL
# ============================================================================

print("\n3. Training the World Model...")

# Initialize fresh model for training
world_model = SimpleWorldModel(state_dim, action_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(world_model.parameters(), lr=0.001)

num_epochs = 100
batch_size = 32
losses = []

for epoch in range(num_epochs):
    # Shuffle data
    indices = torch.randperm(len(states))
    
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, len(states), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_states = states[batch_indices]
        batch_actions = actions[batch_indices]
        batch_next_states = next_states[batch_indices]
        
        optimizer.zero_grad()
        predicted_next_states = world_model(batch_states, batch_actions)
        loss = criterion(predicted_next_states, batch_next_states)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f'   Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print(f"   Final training loss: {losses[-1]:.4f}")


# ============================================================================
# 4. MODEL EVALUATION
# ============================================================================

print("\n4. Evaluating the World Model...")

with torch.no_grad():
    predicted_next_states = world_model(states, actions)
    mse = criterion(predicted_next_states, next_states)
    print(f"   Mean Squared Error on training data: {mse.item():.4f}")
    
    # Calculate per-dimension errors
    per_dim_error = torch.mean((predicted_next_states - next_states) ** 2, dim=0)
    print(f"   Per-dimension MSE: {per_dim_error.numpy()}")


# ============================================================================
# 5. REGULARIZED WORLD MODEL
# ============================================================================

print("\n5. Training Regularized World Model with Dropout...")

class RegularizedWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, dropout_rate=0.2):
        super(RegularizedWorldModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(32, state_dim)
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        next_state = self.fc3(x)
        return next_state

regularized_model = RegularizedWorldModel(state_dim, action_dim)
optimizer_reg = optim.Adam(regularized_model.parameters(), lr=0.001, weight_decay=0.0001)

reg_losses = []

for epoch in range(num_epochs):
    indices = torch.randperm(len(states))
    epoch_loss = 0
    num_batches = 0
    
    for i in range(0, len(states), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_states = states[batch_indices]
        batch_actions = actions[batch_indices]
        batch_next_states = next_states[batch_indices]
        
        optimizer_reg.zero_grad()
        predicted_next_states = regularized_model(batch_states, batch_actions)
        loss = criterion(predicted_next_states, batch_next_states)
        loss.backward()
        optimizer_reg.step()
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    reg_losses.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f'   Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

print(f"   Final regularized model loss: {reg_losses[-1]:.4f}")


# ============================================================================
# 6. MODEL ENSEMBLING
# ============================================================================

print("\n6. Training Ensemble of World Models...")

num_models = 5
models = [SimpleWorldModel(state_dim, action_dim) for _ in range(num_models)]
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

ensemble_losses = [[] for _ in range(num_models)]

for epoch in range(num_epochs):
    indices = torch.randperm(len(states))
    
    for model_idx in range(num_models):
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(states), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            batch_next_states = next_states[batch_indices]
            
            optimizers[model_idx].zero_grad()
            predicted_next_states = models[model_idx](batch_states, batch_actions)
            loss = criterion(predicted_next_states, batch_next_states)
            loss.backward()
            optimizers[model_idx].step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        ensemble_losses[model_idx].append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        avg_ensemble_loss = np.mean([losses[-1] for losses in ensemble_losses])
        print(f'   Epoch [{epoch+1}/{num_epochs}], Avg Ensemble Loss: {avg_ensemble_loss:.4f}')

# Evaluate ensemble
print("\n7. Evaluating Ensemble Model...")

with torch.no_grad():
    predictions = [model(states, actions) for model in models]
    ensemble_prediction = torch.mean(torch.stack(predictions), dim=0)
    ensemble_mse = criterion(ensemble_prediction, next_states)
    print(f"   Ensemble Mean Squared Error: {ensemble_mse.item():.4f}")
    
    # Compare individual models
    individual_mses = [criterion(pred, next_states).item() for pred in predictions]
    print(f"   Individual model MSEs: {[f'{mse:.4f}' for mse in individual_mses]}")
    print(f"   Average individual MSE: {np.mean(individual_mses):.4f}")
    print(f"   Ensemble improvement: {(np.mean(individual_mses) - ensemble_mse.item()):.4f}")


# ============================================================================
# 8. VISUALIZATION
# ============================================================================

print("\n8. Generating training curves...")

plt.figure(figsize=(12, 4))

# Plot 1: Training loss
plt.subplot(1, 3, 1)
plt.plot(losses, label='Simple Model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Simple World Model Training')
plt.legend()
plt.grid(True)

# Plot 2: Regularized model loss
plt.subplot(1, 3, 2)
plt.plot(reg_losses, label='Regularized Model', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Regularized World Model Training')
plt.legend()
plt.grid(True)

# Plot 3: Ensemble models
plt.subplot(1, 3, 3)
for i, model_losses in enumerate(ensemble_losses):
    plt.plot(model_losses, alpha=0.5, label=f'Model {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Ensemble Models Training')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('world_models_training.png', dpi=150)
print("   Saved training curves to 'world_models_training.png'")


# ============================================================================
# 9. TEST PREDICTIONS
# ============================================================================

print("\n9. Testing predictions on sample trajectories...")

# Get a test trajectory
env = gym.make('CartPole-v1')
test_state, info = env.reset()

test_trajectory = []
predicted_trajectory = []

for step in range(20):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Predict using ensemble
    with torch.no_grad():
        state_tensor = torch.tensor(test_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([[float(action)]], dtype=torch.float32)
        
        predictions = [model(state_tensor, action_tensor) for model in models]
        predicted_next = torch.mean(torch.stack(predictions), dim=0).squeeze().numpy()
    
    test_trajectory.append(next_state)
    predicted_trajectory.append(predicted_next)
    
    test_state = next_state
    
    if terminated or truncated:
        break

test_trajectory = np.array(test_trajectory)
predicted_trajectory = np.array(predicted_trajectory)

print(f"   Tested {len(test_trajectory)} steps")
print(f"   Prediction error: {np.mean((test_trajectory - predicted_trajectory)**2):.4f}")

env.close()

print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
print("\nSummary:")
print(f"  - Simple model final MSE: {losses[-1]:.4f}")
print(f"  - Regularized model final MSE: {reg_losses[-1]:.4f}")
print(f"  - Ensemble model MSE: {ensemble_mse.item():.4f}")
print(f"  - Training data collected: {len(states)} transitions")
print(f"  - Models trained for {num_epochs} epochs")
print("\nFiles created:")
print("  - world_models_training.png (training curves)")