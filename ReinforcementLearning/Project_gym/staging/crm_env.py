
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Local imports
from src.data_gen import generate_batch, TARGET_KEYS, EVENT_TYPES
from src.validator import CRMValidator

try:
    from logs.logger_config import get_logger
    logger = get_logger("CRM_Env")
except ImportError:
    logging.basicConfig()
    logger = logging.getLogger("CRM_Env")

class CRMDataCleaningEnv(gym.Env):
    """
    Gymnasium Environment for CRM Data Cleaning with Multi-Event Support.
    
    The agent receives a raw JSON record (observation) with an event type
    and must choose actions to transform it into a valid Zoho CRM format.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=None):
        super(CRMDataCleaningEnv, self).__init__()

        # Validator & Data Source
        self.validator = CRMValidator()
        self.raw_data_queue = [] # Buffer for simulated incoming data
        self.current_record = {}
        self.current_event_type = "new_lead"  # Default event type
        
        # Action Space definition
        # 0: No-Op
        # 1: Map 'fname'/'firstname'/'Nombre' -> 'First_Name'
        # 2: Map 'lname'/'lastname'/'Apellidos' -> 'Last_Name'
        # 3: Map 'email'/'Correo' -> 'Email'
        # 4: Map 'phone'/'contact'/'Telefono' -> 'Phone'
        # 5: Clean Phone Format (Strip non-numeric, add +1 defaults)
        # 6: Capitalize Names
        # 7: Commit to CRM (Submit)
        # 8: Discard Record (Spam/Duplicate)
        self.action_space = spaces.Discrete(9)

        # Observation Space definition (10 dimensions with event type)
        # Features:
        # [0]: Has 'First_Name'? (0/1)
        # [1]: Has 'Last_Name'?
        # [2]: Has 'Email'?
        # [3]: Has 'Phone'?
        # [4]: Email format valid? (0/1 regex check)
        # [5]: Phone format valid? (0/1 regex check)
        # [6]: Is Duplicate? (0/1 - checked against DB)
        # [7]: Source ID (0=Unknown, 1=Gimlet, 2=Bonasera, 3=Hubspot)
        # [8]: Event Type (0=new_lead, 1=offer, 2=contract, 3=maintenance)
        # [9]: Step Count (normalized)
        
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1, 1, 1, 3, 3, 1], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Mapping for event type encoding
        self.event_type_map = {et: i for i, et in enumerate(EVENT_TYPES)}

        self.steps_taken = 0
        self.max_steps = 10
        self.reset()

    def _get_obs(self) -> np.ndarray:
        """
        Converts the current JSON state and event type into the observation vector.
        """
        obs = np.zeros(10, dtype=np.float32)
        
        # Check target keys presence
        obs[0] = 1.0 if 'First_Name' in self.current_record else 0.0
        obs[1] = 1.0 if 'Last_Name' in self.current_record else 0.0
        obs[2] = 1.0 if 'Email' in self.current_record else 0.0
        obs[3] = 1.0 if 'Phone' in self.current_record else 0.0
        
        # Validity Checks (using validator's regex logic but on current fields)
        email_val = self.current_record.get('Email', '')
        obs[4] = 1.0 if self.validator.validate_email(email_val) else 0.0
        
        phone_val = self.current_record.get('Phone', '')
        obs[5] = 1.0 if self.validator.validate_phone(phone_val) else 0.0
        
        # Duplicate Check
        temp_rec = {"Email": email_val}
        obs[6] = 1.0 if self.validator.is_duplicate(temp_rec) else 0.0
        
        # Source ID (Heuristic based on keys)
        if 'fname' in self.current_record: obs[7] = 1.0 # Gimlet
        elif 'Nombre' in self.current_record: obs[7] = 2.0 # Bonasera
        elif 'firstname' in self.current_record: obs[7] = 3.0 # Hubspot
        else: obs[7] = 0.0

        # Event Type encoding
        obs[8] = float(self.event_type_map.get(self.current_event_type, 0))

        # Step count (normalized)
        obs[9] = self.steps_taken / self.max_steps
        
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps_taken = 0
        
        # Replenish queue if empty
        if not self.raw_data_queue:
            self.raw_data_queue = generate_batch(20)
            
        self.current_record = self.raw_data_queue.pop(0)
        
        # Extract event_type from record
        self.current_event_type = self.current_record.pop('event_type', 'new_lead')
        
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps_taken += 1
        reward = -0.1 # Slight penalty for each step to encourage efficiency
        terminated = False
        truncated = False
        info = {}
        info['event_type'] = self.current_event_type
        
        record = self.current_record

        if action == 0: # No-Op
            pass
            
        elif action == 1: # Map First Name
            for k in ['fname', 'firstname', 'Nombre']:
                if k in record:
                    record['First_Name'] = record.pop(k)
                    reward += 1.0 # Small reward for successful mapping
                    break
                    
        elif action == 2: # Map Last Name
            for k in ['lname', 'lastname', 'Apellidos']:
                if k in record:
                    record['Last_Name'] = record.pop(k)
                    reward += 1.0
                    break

        elif action == 3: # Map Email
            for k in ['email_address', 'email', 'Correo']:
                if k in record:
                    record['Email'] = record.pop(k)
                    reward += 1.0
                    break

        elif action == 4: # Map Phone
            for k in ['contact', 'phone', 'Telefono']:
                if k in record:
                    record['Phone'] = record.pop(k)
                    reward += 1.0
                    break

        elif action == 5: # Clean Phone
            if 'Phone' in record:
                p = str(record['Phone'])
                # Remove spaces, parens
                p_clean = "".join(filter(lambda x: x.isdigit() or x == '+', p))
                
                # Check formatting needs
                if len(p_clean) == 10 and not p_clean.startswith('+'): 
                    p_clean = "+1-" + p_clean[:3] + "-" + p_clean[3:6] + "-" + p_clean[6:]
                
                record['Phone'] = p_clean
                reward += 2.0 # Good job

        elif action == 6: # Capitalize Names
            for field in ['First_Name', 'Last_Name']:
                if field in record and isinstance(record[field], str) and record[field]:
                    if not record[field][0].isupper():
                        record[field] = record[field].capitalize()
                        reward += 0.5 

        elif action == 7: # Commit
            terminated = True
            
            # Use Validator to assess risk/reward (pass event_type)
            risk_reward, reason = self.validator.assess_record(
                record, 
                event_type=self.current_event_type
            )
            
            reward += risk_reward
            info['reason'] = reason
            
            if risk_reward > 0:
                self.validator.commit(record) # Add to DB

        elif action == 8: # Discard
            terminated = True
            # If the record WAS valid and we discarded it -> Penalty (Lost Opportunity)
            # If the record WAS INVALID -> Reward (Risk Avoided)
            
            potential_reward, reason = self.validator.assess_record(
                record, 
                event_type=self.current_event_type
            )
            
            if potential_reward > 0:
                # We threw away a good lead!
                reward -= 20.0 
                info['reason'] = "Discarded valid lead!"
            else:
                # Good call, it was garbage.
                reward += 5.0
                info['reason'] = "Correctly discarded bad lead."

        # Check max steps
        if self.steps_taken >= self.max_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        print(f"Current Record State: {self.current_record} | Event Type: {self.current_event_type}")


