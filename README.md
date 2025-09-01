# PPO Traffic Light Control System

This project implements a Proximal Policy Optimization (PPO) reinforcement learning algorithm for intelligent traffic light control using a custom Gymnasium environment that interfaces with NetLogo simulation.

## Project Overview

The system uses PPO to learn optimal traffic light timing strategies for a 4-way junction traffic simulation. The environment observes traffic densities from 8 sensors and controls traffic light combinations and timing to minimize traffic congestion.

## Features

- Custom Gymnasium environment for traffic simulation
- PPO algorithm implementation using Stable-Baselines3
- NetLogo integration for realistic traffic simulation
- Real-time traffic density monitoring
- Configurable traffic light combinations and timing
- Model training, testing, and evaluation capabilities

## Requirements

- Python 3.8+
- Java JDK (for NetLogo integration)
- NetLogo 6.4.0 or compatible version
- Required Python packages (see requirements.txt)

## Installation

### Option 1: Using venv (Recommended)

1. Clone or download this repository
2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment
   python -m venv traffic_env
   
   # Activate virtual environment
   # On Windows:
   traffic_env\Scripts\activate
   # On macOS/Linux:
   source traffic_env/bin/activate
   ```
3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using uv (Fast Alternative)

1. Clone or download this repository
2. Install uv if you haven't already:
   ```bash
   pip install uv
   ```
3. Create and activate a virtual environment with uv:
   ```bash
   # Create virtual environment
   uv venv traffic_env
   
   # Activate virtual environment
   # On Windows:
   traffic_env\Scripts\activate
   # On macOS/Linux:
   source traffic_env/bin/activate
   ```
4. Install dependencies with uv (much faster):
   ```bash
   uv pip install -r requirements.txt
   ```

### Final Setup Steps

1. Ensure Java JDK is installed and the path is correctly set in the code
2. **NetLogo Model Setup**: The NetLogo model file 4Way-Junction-Traffic-Simulation-SriLanka.nlogo is not included in this repository. please [download the model from the companion repository](https://github.com/srimalonline/4Way-Junction-Traffic-Simulation-SriLanka) and place it in the same directory as your Python files.  

**Note**: If you move the Python files to a different location, make sure to also copy the `4Way-Junction-Traffic-Simulation-SriLanka.nlogo` file to the same directory, or update the file path in the code:

```python
# In ppo_traffic_simulation.py, line ~51
self.netlogo.load_model(r'4Way-Junction-Traffic-Simulation-SriLanka.nlogo')
```

### Deactivating Virtual Environment

When you're done working with the project:
```bash
deactivate
```

## Usage

### Basic Training and Testing

Run the main script to train and test the PPO model:

```bash
python ppo_traffic_simulation.py
```

### Custom Training

```python
from ppo_traffic_simulation import train_ppo_model

# Train with custom parameters
model = train_ppo_model(
    timesteps=50000,
    log_dir='custom_logs',
    model_path='custom_model_path'
)
```

### Testing a Trained Model

```python
from ppo_traffic_simulation import test_ppo_model

# Test the model for 100 steps
test_ppo_model(
    model_path='Training/PPO_Traffic_Model',
    num_steps=100
)
```

### Model Evaluation

```python
from ppo_traffic_simulation import evaluate_ppo_model

# Evaluate model performance
mean_reward, std_reward = evaluate_ppo_model(
    model_path='Training/PPO_Traffic_Model',
    n_eval_episodes=10
)
```

## Environment Details

### Observation Space
- **Type**: Box space with 8 continuous values
- **Range**: [0, ∞)
- **Description**: Traffic densities from 8 sensors (S1-S8)

### Action Space
- **Type**: MultiDiscrete
- **Structure**: [7, 36, 36, 36, 36]
- **Description**: 
  - First value: Traffic route combination (1-7)
  - Next 4 values: Green light durations for each side (5-40 seconds)

### Reward System
- **+1**: When traffic density standard deviation decreases (better traffic flow)
- **-1**: When traffic density standard deviation increases (worse traffic flow)
- **-10**: Deadlock detection (all densities are 0)

## Traffic Route Combinations

The system supports 7 different route combinations:

1. **com1**: R12, R34, R56, R78
2. **com2**: R12, R37, R48, R56
3. **com3**: R14, R26, R37, R58
4. **com4**: R14, R27, R36, R58
5. **com5**: R15, R26, R34, R78
6. **com6**: R15, R26, R37, R48
7. **com7**: R15, R27, R36, R48

Each route connects specific sensors (e.g., R12 connects S1 to S2).

## File Structure

```
├── ppo_traffic_simulation.py          # Main PPO implementation
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── SB3-PPO.ipynb                     # Original Jupyter notebook
├── 4Way-Junction-Traffic-Simulation-SriLanka.nlogo  # NetLogo model
└── Training/                         # Training outputs
    ├── Logs/                         # TensorBoard logs
    └── PPO_Traffic_Model             # Saved model files
```

## Configuration

### NetLogo Connection
Update the JVM path in the code to match your Java installation:

```python
self.netlogo = pynetlogo.NetLogoLink(
    jvm_path=r"C:\Program Files\Java\jdk-19\bin\server\jvm.dll",
    gui=True,
)
```

### Training Parameters
Modify training parameters in the `train_ppo_model` function:

- `timesteps`: Number of training steps
- `log_dir`: Directory for TensorBoard logs
- `model_path`: Path to save the trained model

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir=Training/Logs
```

## Troubleshooting

### Common Issues

1. **Java/NetLogo Connection Issues**
   - Ensure Java JDK is properly installed
   - Verify the JVM path in the code
   - Check that the NetLogo model file exists

2. **PyTorch Installation Issues**
   - Reinstall PyTorch with compatible versions
   - Check CUDA compatibility if using GPU

3. **Memory Issues**
   - Reduce the number of training timesteps
   - Use smaller batch sizes in PPO configuration

### Dependencies Conflicts

If you encounter dependency conflicts:

```bash
pip install --upgrade stable-baselines3
pip install --force-reinstall torch torchvision torchaudio
```

## Performance Optimization

- **CPU Training**: Default configuration works well for CPU training
- **GPU Training**: Ensure CUDA-compatible PyTorch installation
- **Memory Usage**: Monitor memory usage during long training sessions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:
```
**PPO Traffic Light Control System**  
Srimal Fernando / NSBM Green University  
September 2025
```
If yiu prefer BibTex
```
@misc{fernando2025ppo,
  title     = {PPO Traffic Light Control System},
  author    = {Srimal Fernando},
  institution = {NSBM Green University},
  year      = {2025},
  url       = {https://github.com/srimalonline/sb3-ppo-traffic-optimization}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Stable-Baselines3 for the PPO implementation
- OpenAI Gymnasium for the environment framework
- NetLogo for traffic simulation capabilities
