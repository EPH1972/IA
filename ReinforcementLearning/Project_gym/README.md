# Project Gym: CRM DQN Agent - Multi-Event Type Support

**Status**: ✅ Development with Multi-Event Support

This project implements a Deep Q-Network agent to intelligently clean and transform CRM data from multiple sources into properly formatted events for Zoho CRM.

**Key Feature**: Supports multiple event types (offer, contract, new_lead, maintenance) with different output schemas.

## Directory Structure

```
Project_gym/
├── staging/              ← Development & Testing Environment
│   ├── src/
│   ├── logs/
│   ├── output/
│   ├── README.md
│   ├── .env.staging
│   └── [core scripts]
│
├── Apuntes               ← Project Notes & Requirements
├── enviroment.env        ← Global Configuration
├── instructions.md       ← Event Type Schemas & Validation Rules
├── project_env/          ← Python Virtual Environment
└── README.md             ← This File
```

## Quick Start

### Training & Development (Staging)

```bash
cd staging

# Train the DQN model
python3 train.py

# Monitor training in real-time
python3 tensorboard_runner.py     # http://localhost:6007

# Test the trained model
python3 inference.py
```

### How It Works

1. **Input**: Receives dirty/incomplete data from Gimlet, Bonasera, or Hubspot in JSON format
2. **Event Type Detection**: Identifies the event type (offer, contract, new_lead, maintenance)
3. **RL-Guided Cleaning**: DQN agent decides the optimal cleaning/transformation sequence
4. **Validation**: Ensures output matches event-type schema
5. **Output**: Properly formatted JSON ready for Zoho CRM

### Supported Event Types

| Event Type | Schema | Example Output |
|------------|--------|-----------------|
| `new_lead` | Basic lead data | First_Name, Last_Name, Email, Phone |
| `offer` | Lead + offer details | Lead fields + Offer_Amount, Offer_Date |
| `contract` | Full contract data | Lead fields + Contract_Terms, Contract_Value |
| `maintenance` | Maintenance ticket | Lead fields + Ticket_ID, Issue_Type |

## Architecture

### Processing Pipeline

```
Input JSON (Gimlet/Bonasera/Hubspot)
           ↓
[Event Type Detection]
           ↓
[State Vectorization]
           ↓
[DQN Agent Decision Loop]
  - Map fields from source format
  - Clean/normalize values
  - Handle missing fields
  - Validate schema compliance
           ↓
[Output Formatter - Event Type Specific]
           ↓
Zoho CRM Ready JSON
```

## File Organization

The staging environment contains:
- **`src/`** - Core source code
  - `data_gen.py` - Multi-source & multi-event mock data generation
  - `validator.py` - Event-type-specific validation rules + reward oracle
- **`logs/`** - Logging infrastructure
  - `logger_config.py` - Centralized logging setup
  - `gym_debug.log` - Training & execution logs
- **`output/`** - Model artifacts
  - `checkpoints/` - Saved DQN models
  - `tensorboard_logs/` - Training metrics & curves
- **Main Scripts**
  - `crm_env.py` - Custom Gymnasium environment with event-type support
  - `train.py` - DQN training pipeline
  - `inference.py` - Model testing & inference
  - `tensorboard_runner.py` - TensorBoard visualization server
  - `dqn_config.py` - Hyperparameter configuration
- **Configuration**
  - `enviroment.env` - Global CRM config & event schemas
  - `.env.staging` - Staging-specific settings
  - `instructions.md` - Event validation rules

## Configuration Management

### Global Configuration (`enviroment.env`)

Unified CRM field mapping, validation rules, and event schemas:
- **Required Fields** (per event type)
- **Validation Rules**: Email/Phone regex patterns
- **Data Sources**: Field mappings for Gimlet, Bonasera, Hubspot
- **Event Type Schemas**: Offer, Contract, New_Lead, Maintenance
- **Deduplication Rules**

### Event Type Handling

Each event type has its own validation schema in `instructions.md`:
- **new_lead**: First_Name, Last_Name, Email, Phone
- **offer**: new_lead fields + Offer_Amount, Offer_Date, Expected_Revenue
- **contract**: offer fields + Contract_Terms, Contract_Value, Contract_Date
- **maintenance**: Lead fields + Ticket_ID, Issue_Type, Priority

### Staging Configuration (`.env.staging`)

```bash
environment=staging
debug=true
log_level=DEBUG
training_timesteps=10000
tensorboard_port=6007
```

## Development Workflow

### 1. Prepare Data

Create training data with event types:
```bash
cd staging
python3 -c "
from src.data_gen import generate_batch
batch = generate_batch(size=100, event_types=['new_lead', 'offer', 'contract', 'maintenance'])
print(f'Generated {len(batch)} records')
"
```

### 2. Train the Agent

```bash
cd staging
python3 train.py
```

This will:
- Create environment with event-type support
- Initialize DQN agent
- Train for 10,000 timesteps
- Save checkpoints to `output/checkpoints/`
- Log to `logs/gym_debug.log`

### 3. Monitor Training

```bash
python3 tensorboard_runner.py
```

Open http://localhost:6007 to view:
- Reward curves per event type
- Episode efficiency
- Loss convergence
- Action distribution

### 4. Evaluate Performance

```bash
python3 inference.py
```

Tests the model on representative events from each type.

## Monitoring

### TensorBoard in Staging

```bash
cd staging
python3 tensorboard_runner.py
# http://localhost:6007
```

Metrics tracked:
- **Reward**: Episode reward by event type
- **Episode Length**: Steps to complete per event type
- **Q-Value Loss**: Model learning progress
- **Exploration**: Epsilon decay over time

## Troubleshooting

### Model not converging
- Check data distribution is balanced across event types
- Verify reward shaping in `validator.py`
- Review logs: `tail -f staging/logs/gym_debug.log`

### Event type not recognized
- Ensure event_type is included in input JSON
- Check event_type against allowed values in `enviroment.env`
- Verify `validator.py` has handling for the event type

### Out of memory during training
- Reduce `BUFFER_SIZE` in `dqn_config.py`
- Reduce `TOTAL_TIMESTEPS`
- Reduce batch generation size

### TensorBoard shows no data
- Ensure at least one episode completed
- Check `staging/output/tensorboard_logs/` exists
- Verify logging is enabled in `.env.staging`

## Performance Expectations

- **Mean Reward**: ~4.5-5.0 (valid leads)
- **Episode Length**: ~1-2 steps (efficient)
- **Accuracy**: ~90%+
- **Throughput**: 100-200 leads/sec (with batching)

## Next Steps

1. **Add Real Event Type Handling**: Map all 4 event types with their specific schemas
2. **Optimize Event Type Specific Rewards**: Fine-tune reward function per event type
3. **Multi-Source Testing**: Generate realistic Gimlet/Bonasera/Hubspot data samples
4. **Production Gateway**: Build API wrapper for Zoho CRM integration
5. **Performance Benchmarking**: Test on actual CRM data

## Support

For issues or questions:
- Check `README.md` in each environment directory
- Review logs in `logs/gym_debug.log`
- See `instructions.md` for CRM validation rules
- View `enviroment.env` for configuration

---

## Technical Details

### DQN Architecture

- **Network**: 2-layer MLP (64 → 64 → action_space)
- **Learning Rate**: 1e-3
- **Buffer Size**: 10,000 transitions
- **Exploration**: 1.0 → 0.05 decay over 10% of training
- **Target Update**: Every 1,000 steps
- **Batch Size**: 32 transitions per update

### Observation Space

9-dimensional state vector:
- [0-3]: Field presence flags (First_Name, Last_Name, Email, Phone)
- [4-5]: Validity flags (Email_Valid, Phone_Valid)
- [6]: Duplicate flag (0=new, 1=duplicate)
- [7]: Source ID (0=Gimlet, 1=Bonasera, 2=Hubspot, 3=Unknown)
- [8]: Event type encoded (0=new_lead, 1=offer, 2=contract, 3=maintenance)

### Action Space

9 discrete actions:
- 0: No-op (hold state)
- 1-4: Map source fields to target schema
- 5: Clean phone number
- 6: Capitalize/normalize names
- 7: Commit to Zoho CRM
- 8: Discard invalid record

### Reward Function

```
Base reward: 0
+ 10 if commit & schema valid
+ 20 if commit & no duplicates
+ 5 if commit & clean data
- 10 if discard & valid lead (penalize false negatives)
- 5 per step (encourage efficiency)
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- Virtual environment activated: `source project_env/bin/activate`
- Packages installed: gymnasium, stable-baselines3, torch, numpy, tensorboard

### 1. Install Environment

```bash
cd /home/iticbcn/IA/ReinforcementLearning/Project_gym
source project_env/bin/activate

# Install or verify dependencies
pip install gymnasium stable-baselines3 torch numpy tensorboard tqdm rich
```

### 2. Configure Environment

Review and customize:
- `enviroment.env` - Global CRM config and event schemas
- `staging/.env.staging` - Staging-specific settings

### 3. Training

```bash
cd staging
python3 train.py
```

Expected output:
```
Training DQN agent...
Timestep 10000/10000
Evaluation: Mean Reward = 4.52 ± 0.89
Model saved to: output/checkpoints/dqn_crm_agent_10000.zip
```

### 4. Monitor Training

```bash
python3 tensorboard_runner.py
# Open http://localhost:6007
```

### 5. Test Trained Model

```bash
python3 inference.py
```

Shows:
- Actions taken per event type
- Rewards achieved
- Processing efficiency

## Hyperparameter Configuration

Edit `dqn_config.py` to adjust:
- **Learning Rate**: 1e-3 (default) - Lower = slower learning
- **Buffer Size**: 10,000 - Larger = more stable but more memory
- **Exploration**: 1.0 → 0.05 - Epsilon decay rate
- **Total Timesteps**: 10,000 - Training duration
- **Target Update Interval**: 1,000 - Stability parameter

## Event Type Configuration

Define per event type in `enviroment.env`:
```
EVENT_TYPES=new_lead,offer,contract,maintenance

NEW_LEAD_REQUIRED_FIELDS=First_Name,Last_Name,Email,Phone
OFFER_REQUIRED_FIELDS=First_Name,Last_Name,Email,Phone,Offer_Amount,Expected_Revenue
CONTRACT_REQUIRED_FIELDS=First_Name,Last_Name,Email,Phone,Contract_Terms,Contract_Value
MAINTENANCE_REQUIRED_FIELDS=First_Name,Last_Name,Ticket_ID,Issue_Type,Priority
```

## Data Sources

Supports three input sources with automatic field mapping:
- **Gimlet**: `{first, last, contact_email, phone_number}`
- **Bonasera**: `{fname, lname, email, tel}`
- **Hubspot**: `{firstName, lastName, email, phone}`

Validator automatically maps these to canonical schema.

## Training Workflow

1. **Data Generation**: 
   - Mock data generated from all 3 sources (Gimlet, Bonasera, Hubspot)
   - Intentional errors: missing fields, wrong formats, duplicates
   - Random event type assignment
   
2. **Episode Loop** (per event):
   - Agent receives dirty record with event_type
   - State vectorized: field presence, validity, source, event type
   
3. **Decision Making**:
   - Agent observes state (9 dimensions)
   - DQN selects action based on learned Q-values
   - Action transforms record (map fields, clean phone, etc.)
   
4. **Event-Specific Validation**:
   - Validator checks against event-type schema
   - Reward based on outcome and event requirements
   
5. **Learning**:
   - Experience stored in replay buffer
   - Batch of 32 random transitions used for training
   - Q-values updated via temporal difference error

## Expected Performance

- **Random Agent**: ~0 reward (baseline)
- **Trained Agent**: ~4-5 reward (after 10K timesteps)
- **Episode Efficiency**: ~1-2 steps average
- **Accuracy**: ~85-90% valid outputs

## Support & Issues

- Check logs: `tail -f staging/logs/gym_debug.log`
- Review Apuntes for requirements
- See instructions.md for validation rules
- Inspect enviroment.env for configuration
