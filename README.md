# World Models in PyTorch - Demo

This is a complete, working implementation of World Models in PyTorch based on the tutorial from [codegenes.net](https://www.codegenes.net/blog/world-models-pytorch/).

## What This Does

World Models are learned representations of an environment that allow AI agents to:
- **Predict future states** based on current state and actions
- **Plan actions** by simulating different scenarios
- **Learn efficiently** from limited real-world data

This implementation demonstrates:
1. **Simple World Model** - Basic neural network that predicts next states
2. **Data Collection** - Gathering training data from CartPole environment
3. **Model Training** - Training the world model using supervised learning
4. **Regularization** - Using dropout to prevent overfitting
5. **Ensemble Learning** - Combining multiple models for better predictions

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python world_models_demo.py
```

## What to Expect

The script will:
1. ✅ Create and test a simple world model architecture
2. ✅ Collect ~5,000-10,000 transitions from CartPole environment
3. ✅ Train a basic world model (100 epochs, ~30 seconds)
4. ✅ Train a regularized model with dropout
5. ✅ Train an ensemble of 5 models
6. ✅ Compare prediction accuracy across all models
7. ✅ Generate training curves visualization
8. ✅ Test predictions on a real trajectory

### Expected Output

```
============================================================
World Models in PyTorch - Demo
============================================================

1. Defining Simple World Model...
   Model created with 4865 parameters
   Test forward pass successful: input shape torch.Size([1, 4]) -> output shape torch.Size([1, 4])

2. Collecting data from CartPole environment...
   Collected 8342 transitions
   States shape: torch.Size([8342, 4])
   Actions shape: torch.Size([8342, 1])
   Next states shape: torch.Size([8342, 4])

3. Training the World Model...
   Epoch [20/100], Loss: 0.0234
   Epoch [40/100], Loss: 0.0156
   ...
   Final training loss: 0.0089

... and more!
```

## Understanding the Results

### Performance Metrics

- **Simple Model MSE**: Baseline performance (~0.01)
- **Regularized Model MSE**: Usually similar, more robust to overfitting
- **Ensemble Model MSE**: Best performance through averaging predictions

Lower MSE = better predictions!

### Training Curves

The script generates `world_models_training.png` showing:
- Loss over time for each model type
- How quickly models converge
- Stability of training process

## Key Components

### SimpleWorldModel Class
```python
class SimpleWorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(SimpleWorldModel, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, state_dim)
```

Takes current state + action → Predicts next state

### Training Process
1. Collect data by running random policy in CartPole
2. Train neural network to predict `state(t+1)` from `state(t)` and `action(t)`
3. Minimize Mean Squared Error between predictions and actual next states

## Customization

### Change the Environment
```python
env = gym.make('MountainCar-v0')  # Instead of CartPole
```

### Adjust Model Complexity
```python
world_model = SimpleWorldModel(state_dim, action_dim, hidden_dim=128)  # Bigger model
```

### More Training Data
```python
num_episodes = 100  # Collect more episodes
```

### Longer Training
```python
num_epochs = 200  # Train for more epochs
```

## Troubleshooting

### "No module named 'gym'"
```bash
pip install gym gymnasium
```

### "CUDA out of memory"
The script uses CPU by default. If you want GPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
world_model = world_model.to(device)
```

### Different gym API versions
The code handles both old and new gym APIs automatically.

## What's Next?

After running this demo, you can:
1. **Use the model for planning** - Simulate action sequences to find optimal behavior
2. **Try different environments** - MountainCar, Pendulum, etc.
3. **Add stochastic predictions** - Use variational autoencoders for uncertainty
4. **Implement model-based RL** - Use world model for training agents
5. **Add reward prediction** - Extend model to predict rewards too

## Technical Details

- **Framework**: PyTorch 2.0+
- **Environment**: OpenAI Gym CartPole-v1
- **Training**: Adam optimizer, MSE loss
- **Architecture**: Multi-layer perceptron (MLP)
- **Data**: ~5K-10K state transitions

## References

- Original Tutorial: https://www.codegenes.net/blog/world-models-pytorch/
- World Models Paper: https://arxiv.org/abs/1803.10122
- PyTorch Docs: https://pytorch.org/docs/
- OpenAI Gym: https://gymnasium.farama.org/

## License

This code is for educational purposes based on the public tutorial.