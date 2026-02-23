import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from crm_env import CRMDataCleaningEnv
from logs.logger_config import get_logger

logger = get_logger("Trainer")

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log episode info when an episode ends
        if "dones" in self.locals and self.locals["dones"][0]:
            info = self.locals.get("infos", [{}])[0]
            logger.debug(f"Episode ended - Info: {info}")
        
        return True


class DQNTrainer:
    """
    Trainer class for DQN agent on CRM Data Cleaning environment.
    """
    
    def __init__(self, env_id="CRMDataCleaning-v0", learning_rate=1e-3, buffer_size=10000):
        self.env_id = env_id
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.model = None
        self.env = None
        
    def create_env(self):
        """Create the training environment."""
        self.env = CRMDataCleaningEnv()
        logger.info(f"Environment created: {self.env_id}")
        logger.info(f"Action space: {self.env.action_space}")
        logger.info(f"Observation space: {self.env.observation_space}")
        return self.env
    
    def train(self, total_timesteps=50000, log_dir="./output"):
        """
        Train the DQN agent.
        
        Args:
            total_timesteps: Total number of environment steps to train for.
            log_dir: Directory for saving logs and checkpoints.
        """
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_log = os.path.join(log_dir, "tensorboard_logs")
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(tensorboard_log, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create environment
        self.create_env()
        
        # Initialize DQN model
        logger.info("Initializing DQN model...")
        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            train_freq=(1, "step"),
            target_update_interval=1000,
            learning_starts=1000,
            verbose=1,
            tensorboard_log=tensorboard_log,
            seed=42
        )
        
        logger.info(f"Starting training for {total_timesteps} timesteps...")
        
        # Create callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,
            save_path=checkpoint_dir,
            name_prefix="dqn_model"
        )
        tensorboard_callback = TensorboardCallback()
        
        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, tensorboard_callback],
            log_interval=100,
            progress_bar=True
        )
        
        # Save the model
        model_path = os.path.join(checkpoint_dir, "dqn_crm_agent_final")
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        return self.model
    
    def evaluate(self, num_episodes=10):
        """
        Evaluate the trained model.
        
        Args:
            num_episodes: Number of episodes to run for evaluation.
        """
        if self.model is None:
            logger.error("Model not trained yet. Call train() first.")
            return
        
        logger.info(f"Evaluating model for {num_episodes} episodes...")
        
        total_rewards = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            step = 0
            
            while not terminated and not truncated:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                step += 1
            
            total_rewards.append(episode_reward)
            logger.info(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        logger.info(f"Evaluation Summary - Mean Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        return total_rewards
    
    def load_and_play(self, model_path, num_episodes=5):
        """
        Load a trained model and play episodes.
        
        Args:
            model_path: Path to the saved model.
            num_episodes: Number of episodes to run.
        """
        logger.info(f"Loading model from {model_path}...")
        
        if self.env is None:
            self.create_env()
        
        self.model = DQN.load(model_path, env=self.env)
        logger.info("Model loaded successfully.")
        
        return self.evaluate(num_episodes=num_episodes)


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("DQN Training Session Started")
    logger.info("=" * 80)
    
    # Initialize trainer
    trainer = DQNTrainer(
        env_id="CRMDataCleaning-v0",
        learning_rate=1e-3,
        buffer_size=10000
    )
    
    # Train the agent
    logger.info("Phase 1: Training Agent")
    model = trainer.train(total_timesteps=10000, log_dir="./output")
    
    # Evaluate the trained agent
    logger.info("\nPhase 2: Evaluating Agent")
    trainer.evaluate(num_episodes=10)
    
    logger.info("=" * 80)
    logger.info("Training Session Complete")
    logger.info("=" * 80)
