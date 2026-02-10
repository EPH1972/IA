import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import sys
from PIL import Image

# Define the Custom Pac-Man Environment
class SimplePacmanEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None):
        super(SimplePacmanEnv, self).__init__()
        
        # 0: Empty, 1: Wall, 2: Dot, 3: Pacman, 4: Ghost
        self.map_layout = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
            [1, 2, 1, 2, 1, 2, 1, 1, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 2, 1, 1, 0, 0, 1, 1, 2, 1],
            [1, 2, 2, 2, 0, 0, 2, 2, 2, 1],
            [1, 2, 1, 2, 1, 1, 2, 1, 2, 1],
            [1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        
        self.rows, self.cols = self.map_layout.shape
        self.action_space = spaces.Discrete(4) # Up, Right, Down, Left
        # Observation is the grid
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.rows, self.cols), dtype=np.int32)
        
        self.render_mode = render_mode
        self.pacman_pos = None
        self.ghost_pos = None
        self.dots_left = 0
        self.max_steps = 300
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.state = self.map_layout.copy()
        # Initialize Pacman and Ghost positions
        # Find empty spots or predefined spots
        self.pacman_pos = [1, 1]
        self.ghost_pos = [5, 4] # Center-ish
        
        # Reset dots
        self.dots_left = np.sum(self.state == 2)
        
        # Place agents
        self.state[self.pacman_pos[0], self.pacman_pos[1]] = 3
        # Ensure ghost doesn't overwrite a wall, though map suggests 0 there
        self.state[self.ghost_pos[0], self.ghost_pos[1]] = 4
        
        self.current_step = 0
        
        return self.state.copy(), {}

    def step(self, action):
        self.current_step += 1
        
        # Move Pacman
        moves = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        dr, dc = moves[action]
        new_r, new_c = self.pacman_pos[0] + dr, self.pacman_pos[1] + dc
        
        reward = -1 # Step penalty
        terminated = False
        truncated = False
        
        if self.map_layout[new_r, new_c] != 1: # Not a wall
            # Move logic
            # Clear old pos. If it was a dot originally, now it's empty (0)
            # We need to track dots collected separately or modify state permanently
            # Here we modify state.
            
            # Check content of new cell in CURRENT state (to see if dot or ghost)
            cell_content = self.state[new_r, new_c]
            
            if cell_content == 4: # Hit Ghost
                reward = -50
                terminated = True
            elif cell_content == 2: # Eat Dot
                reward = 10
                self.dots_left -= 1
                if self.dots_left == 0:
                    reward += 100 # BIG Win bonus
                    terminated = True
            
            # Update grid
            self.state[self.pacman_pos[0], self.pacman_pos[1]] = 0 # Empty now
            self.pacman_pos = [new_r, new_c]
            self.state[new_r, new_c] = 3
            
        else:
            reward = -5 # Higher Hit wall penalty to force efficient movement
            
        # Move Ghost (Randomly)
        if not terminated:
            g_action = random.choice(list(moves.keys()))
            g_dr, g_dc = moves[g_action]
            g_new_r, g_new_c = self.ghost_pos[0] + g_dr, self.ghost_pos[1] + g_dc
            
            # Ghost respects walls but doesn't eat dots/pacman logic deeply (just kills if collision)
            if self.map_layout[g_new_r, g_new_c] != 1:
                # restore what was under ghost? 
                # For simplicity, let's say ghost floats over things. 
                # But to keep observation simple, we just overwrite. 
                # To do it right: we need to remember what was under the ghost.
                pass 
                # Ignoring complex ghost rendering for this simple version, 
                # assume ghost stays put or swaps for simplicity of state representation validation.
                # Let's make ghost static or bounce for now to ensure stability of simple code
                
                # Simple Ghost Move:
                prev_content_at_ghost = self.map_layout[self.ghost_pos[0], self.ghost_pos[1]] 
                if prev_content_at_ghost != 1: # It's a valid tile type
                     # We can't easily track 'what was under' without a separate memory grid.
                     # Let's just make the ghost roam on empty space/dots without destroying them visually in state?
                     # No, state must reflect 'Ghost is Here'.
                     # Let's just assume Ghost eats dots temporarily or we don't care about restoring dots for this demo.
                     
                     # Better: Ghost only moves to empty squares (0) for now.
                     if self.state[g_new_r, g_new_c] == 0:
                        self.state[self.ghost_pos[0], self.ghost_pos[1]] = 0
                        self.ghost_pos = [g_new_r, g_new_c]
                        self.state[g_new_r, g_new_c] = 4
                     elif self.state[g_new_r, g_new_c] == 3:
                        # Moves onto pacman
                        reward = -50
                        self.state[self.ghost_pos[0], self.ghost_pos[1]] = 0
                        self.ghost_pos = [g_new_r, g_new_c]
                        self.state[g_new_r, g_new_c] = 4
                        terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        return self.state.copy(), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human" or self.render_mode == "ansi":
            outfile = sys.stdout
            chars = {0: " ", 1: "#", 2: ".", 3: "P", 4: "G"}
            print("-" * (self.cols + 2))
            for row in self.state:
                print("|" + "".join([chars[x] for x in row]) + "|")
            print("-" * (self.cols + 2))

# DQN Agent
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        # Input shape is now flattened one-hot: Rows * Cols * Categories(5)
        # 9 * 10 * 5 = 450
        input_dim = input_shape[0] * input_shape[1] * 5 
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.fc(x)

class Agent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=5000) # Larger memory
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995 # Slower decay
        self.learning_rate = 0.0005 # Slower learning rate for stability
        
        # Robust Device Selection
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"\n[DEVICE] CUDA is AVAILABLE. Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"[DEVICE] CUDA Version: {torch.version.cuda}")
        else:
            self.device = torch.device("cpu")
            print("\n[DEVICE] CUDA is NOT available. Using CPU.")
            print(f"[DEVICE] PyTorch Version: {torch.__version__}")
            # Try to force check just in case
            try:
                torch.cuda.init()
            except:
                pass

        self.model = DQN(state_shape, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def preprocess(self, state):
        # Handle single state: (9, 10) -> (1, 450)
        # Create directly on device if possible
        state_tensor = torch.as_tensor(state, dtype=torch.long, device=self.device).unsqueeze(0) # (1, 9, 10)
        one_hot = torch.nn.functional.one_hot(state_tensor, num_classes=5) # (1, 9, 10, 5)
        return one_hot.float().view(1, -1) # (1, 450)

    def preprocess_batch(self, states):
        # Handle batch of states: (B, 9, 10) -> (B, 450)
        states_np = np.array(states)
        # Create directly on device
        state_tensor = torch.as_tensor(states_np, dtype=torch.long, device=self.device) # (B, 9, 10)
        one_hot = torch.nn.functional.one_hot(state_tensor, num_classes=5) # (B, 9, 10, 5)
        return one_hot.float().view(state_tensor.shape[0], -1) # (B, 450)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_t = self.preprocess(state)
        with torch.no_grad():
            act_values = self.model(state_t)
        return torch.argmax(act_values[0]).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        # Efficient batch processing
        states = [i[0] for i in minibatch]
        next_states = [i[3] for i in minibatch]
        
        states_t = self.preprocess_batch(states)
        next_states_t = self.preprocess_batch(next_states)
        
        actions = torch.LongTensor(np.array([i[1] for i in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).to(self.device)

        # Q(s', a') for next states
        with torch.no_grad():
            next_q_values = self.model(next_states_t).max(1)[0]
        
        # Target Q = r + gamma * max Q(s', a')
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Current Q output
        current_q_values = self.model(states_t)
        
        # We only want to update the Q-value for the action we took
        # Gather q_values for the specific actions taken
        current_q_values_action = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.criterion(current_q_values_action, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def save_game_gif(agent, env, filename="pacman_gameplay.gif"):
    print(f"Generating replay GIF: {filename}...")
    frames = []
    state, _ = env.reset()
    terminated = False
    truncated = False
    
    # Color map: 0=Empty(Black), 1=Wall(Blue), 2=Dot(Pink), 3=Pacman(Yellow), 4=Ghost(Red)
    colors = {
        0: [0, 0, 0],       # Empty
        1: [0, 0, 139],     # Wall (Dark Blue)
        2: [255, 183, 174], # Dot (Pink)
        3: [255, 215, 0],   # Pacman (Gold)
        4: [255, 0, 0]      # Ghost (Red)
    }
    
    steps = 0
    # Run a full episode
    while not (terminated or truncated) and steps < 300:
        # 1. Render State to Image
        h, w = state.shape
        # Create an RGB array
        img_data = np.zeros((h, w, 3), dtype=np.uint8)
        for r in range(h):
            for c in range(w):
                pixel_val = state[r, c]
                img_data[r, c] = colors.get(pixel_val, [0,0,0])
        
        # 2. Convert to PIL Image and Upscale
        img = Image.fromarray(img_data, 'RGB')
        # Scale up 30x so it's visible (from ~10x10 pixels to 300x300)
        img = img.resize((w * 30, h * 30), Image.NEAREST)
        frames.append(img)
        
        # 3. Agent Step
        action = agent.act(state)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
        
    # Save GIF
    if frames:
        frames[0].save(filename, save_all=True, append_images=frames[1:], duration=150, loop=0)
        print(f"Saved {len(frames)} frames to {filename}")

# Analysis of the Environment (Step 2 of Instructions)
def analyze_environment(env):
    print("=" * 50)
    print("ANALYSIS OF THE ENVIRONMENT")
    print("=" * 50)
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    
    # Answers to questions
    print("\nQuestions to answer:")
    print(f"- Observation dimensions: {env.observation_space.shape}")
    print(f"- Number of actions: {env.action_space.n}")
    print("- When does an episode end? When Pacman eats all dots (Win) or hits a Ghost (Loss).")
    print("- Reward structure: +10 per dot, -1 per step, -50 collision, +50 win.")
    
    print("=" * 50)

# Evaluation (Step 5 of Instructions)
def evaluate_agent(env, agent, n_episodes=10):
    print("\nStarting Evaluation (Greedy Policy)...")
    rewards = []
    
    for e in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Greedy action only
            action = agent.act(state) # Note: epsilon should be 0 before calling this
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
        
        rewards.append(total_reward)
        print(f"Eval Episode {e+1}: Reward = {total_reward}")
    
    return rewards

def main():
    try:
        env = SimplePacmanEnv(render_mode="ansi")
        analyze_environment(env)
        
        state_shape = env.observation_space.shape
        action_size = env.action_space.n
        agent = Agent(state_shape, action_size)
        
        episodes = 500
        batch_size = 64
        scores = []

        print("\nStarting training on Custom Pac-Man Grid...")
        
        for e in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.remember(state, action, reward, next_state, terminated)
                state = next_state
                total_reward += reward
                
                agent.replay(batch_size)

            scores.append(total_reward)
            if (e+1) % 50 == 0: # Log every 50
                print(f"Episode: {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")

        # Plotting results (Step 4)
        plt.figure(figsize=(10, 5))
        plt.plot(scores)
        plt.title('Learning Curve: Reward vs Episode')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('pacman_training_scores.png')
        print("\nTraining finished. Learning curve saved to pacman_training_scores.png")

        # Set agent to full exploitation for evaluation
        agent.epsilon = 0.0

        # Evaluation (Step 5)
        evaluate_agent(env, agent, n_episodes=5)

        # Generate GIF emulation
        save_game_gif(agent, env, "pacman_gameplay.gif")

        print("\nDemonstration Run (Text):") 
        state, _ = env.reset()
        terminated = False
        truncated = False
        env.render()
        steps = 0
        while not (terminated or truncated) and steps < 20:
            action = agent.act(state) # Agent might still be exploring slightly or exploited
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            print(f"Step {steps+1}, Action: {['Up','Right','Down','Left'][action]}")
            env.render()
            steps += 1
            if terminated:
                if reward > 0:
                    print("WIN!")
                else:
                    print("DIED!")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
