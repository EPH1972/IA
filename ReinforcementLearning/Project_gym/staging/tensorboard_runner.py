"""
Script to run tensorboard for monitoring DQN training.
Usage: python tensorboard_runner.py
Then open http://localhost:6006 in your browser.
"""

import subprocess
import os
from logs.logger_config import get_logger

logger = get_logger("TensorBoard")


def run_tensorboard(log_dir="./output/tensorboard_logs", port=6006):
    """
    Start tensorboard to visualize training progress.
    
    Args:
        log_dir: Directory containing tensorboard logs.
        port: Port to serve tensorboard on.
    """
    
    if not os.path.exists(log_dir):
        logger.error(f"Log directory not found: {log_dir}")
        logger.info("Make sure to train the model first with train.py")
        return
    
    logger.info(f"Starting TensorBoard on port {port}...")
    logger.info(f"Open http://localhost:{port} in your browser")
    logger.info("Press Ctrl+C to stop")
    
    cmd = f"tensorboard --logdir={log_dir} --port={port}"
    
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        logger.info("TensorBoard stopped.")


if __name__ == "__main__":
    run_tensorboard()
