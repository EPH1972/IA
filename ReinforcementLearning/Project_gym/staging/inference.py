"""
Inference script for the trained DQN agent.
Use this to load a trained model and run it on new data.
"""

import os
import sys
from crm_env import CRMDataCleaningEnv
from stable_baselines3 import DQN
from logs.logger_config import get_logger

logger = get_logger("Inference")


def infer_on_sample(model_path, num_episodes=5):
    """
    Load a trained model and run inference on sample data.
    
    Args:
        model_path: Path to the saved DQN model.
        num_episodes: Number of episodes to run.
    """
    
    logger.info(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path + ".zip"):
        logger.error(f"Model file not found: {model_path}.zip")
        return
    
    # Create environment
    env = CRMDataCleaningEnv()
    
    # Load model
    model = DQN.load(model_path, env=env)
    logger.info("Model loaded successfully!")
    
    # Run inference
    logger.info(f"Running inference for {num_episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        action_names = [
            "No-Op", "Map First", "Map Last", "Map Email", "Map Phone",
            "Clean Phone", "Capitalize", "Commit", "Discard"
        ]
        
        logger.info(f"\n--- Episode {episode + 1} ---")
        
        while not terminated and not truncated and step < 20:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            
            logger.info(f"Step {step}: Action={action_names[action]}, Reward={reward:.2f}")
            
            if terminated or truncated:
                logger.info(f"Episode finished. Info: {info}")
                break
            
            step += 1
        
        episode_rewards.append(episode_reward)
        logger.info(f"Episode Reward: {episode_reward:.2f}")
    
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    logger.info(f"\nAverage Reward: {avg_reward:.2f}")


if __name__ == "__main__":
    # Default model path
    model_path = "./output/checkpoints/dqn_crm_agent_final"
    
    # Allow command-line override
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    infer_on_sample(model_path, num_episodes=5)
