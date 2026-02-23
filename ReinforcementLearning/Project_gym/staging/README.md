# Staging Directory

This is the **staging environment** for development and testing of the CRM DQN agent.

## Directory Structure

```
staging/
├── src/
│   ├── data_gen.py           # Mock data generation
│   ├── validator.py          # CRM validation rules
│   └── __init__.py
├── logs/
│   ├── logger_config.py      # Logging configuration
│   └── gym_debug.log         # Training logs
├── output/
│   ├── checkpoints/          # Saved models
│   └── tensorboard_logs/     # Training metrics
├── crm_env.py                # Gymnasium environment
├── train.py                  # Training script
├── inference.py              # Testing script
├── tensorboard_runner.py     # TensorBoard launcher
├── dqn_config.py             # Hyperparameters
├── enviroment.env            # Configuration
└── .env.staging              # Staging-specific config
```

## Usage

### 1. Train the Model
```bash
cd staging
python3 train.py
```

### 2. Monitor Training
```bash
python3 tensorboard_runner.py
# http://localhost:6007
```

### 3. Test the Model
```bash
python3 inference.py
```

## Configuration

Edit `.env.staging` to customize:
- `log_level`: DEBUG (shows all messages)
- `api_port`: 8001 (staging port)
- `training_timesteps`: 10,000 (for quick tests)

## Notes

- This environment is for **development and testing**
- Uses more verbose logging for debugging
- Quick training cycles (10K timesteps)
- Do NOT use for production traffic
