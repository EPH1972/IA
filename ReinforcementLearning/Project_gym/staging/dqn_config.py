"""
DQN Hyperparameters Configuration
"""

# ============================================
# DQN Training Parameters
# ============================================
LEARNING_RATE = 1e-3
BUFFER_SIZE = 10000
EXPLORATION_FRACTION = 0.1
EXPLORATION_INITIAL_EPS = 1.0
EXPLORATION_FINAL_EPS = 0.05
TRAIN_FREQ = 1  # Train every step
TARGET_UPDATE_INTERVAL = 1000
LEARNING_STARTS = 1000

# ============================================
# Training Configuration
# ============================================
TOTAL_TIMESTEPS = 10000
BATCH_SIZE = 32
GAMMA = 0.99  # Discount factor
SEED = 42

# ============================================
# Checkpoint & Logging
# ============================================
CHECKPOINT_FREQ = 5000
LOG_INTERVAL = 100
EVAL_EPISODES = 10

# ============================================
# Environment Configuration
# ============================================
MAX_STEPS_PER_EPISODE = 10
RENDER_MODE = None
