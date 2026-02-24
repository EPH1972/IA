import re
import logging
from typing import Dict, Tuple

# Import the logger configuration
try:
    from logs.logger_config import get_logger
except ImportError:
    # Fallback to basic logging if config not found
    logging.basicConfig()
    get_logger = logging.getLogger

logger = get_logger("Validator")

class CRMValidator:
    """
    Validates CRM records with event-type-specific schemas.
    Acts as the 'Reward Oracle' for the RL Environment.
    """
    
    def __init__(self):
        self.email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        self.phone_regex = r"^\+\d{1,3}-\d{3}-\d{3}-\d{4}$" # E.g., +1-555-010-9999
        
        # Schema definitions are now handled dynamically or will be replaced.
        # Clearing old hardcoded requirements to avoid conflicts with new JSON structure.
        self.required_fields = {} 
        self.base_required = {}
        
        # Simple in-memory database simulation for de-duplication
        self.mock_database = set()  

    def validate_email(self, email: str) -> bool:
        if not email: return False
        return bool(re.match(self.email_regex, email))

    def validate_phone(self, phone: str) -> bool:
        if not phone: return False
        return bool(re.match(self.phone_regex, phone))

    def check_schema(self, record: Dict, event_type: str = "new_lead") -> Tuple[bool, str]:
        """Checks if all required fields for the event type are present and not empty."""
        required = self.required_fields.get(event_type, self.base_required)
        # Extract field names from schema dict
        required_fields = list(required.keys()) if isinstance(required, dict) else required
        missing = [f for f in required_fields if f not in record or not record[f]]
        if missing:
            return False, f"Missing fields for {event_type}: {missing}"
        return True, "Schema OK"

    def is_duplicate(self, record: Dict) -> bool:
        """
        Checks if the record (by Email) already exists in our mock DB.
        """
        email = record.get("Email")
        if not email: return False
        return email in self.mock_database

    def assess_record(self, record: Dict, event_type: str = "new_lead") -> Tuple[float, str]:
        """
        Calculates the Reward for a given record state based on event type.
        Returns: (Reward, Reason)
        """
        
        # 1. Schema Check (event-type specific)
        schema_ok, msg = self.check_schema(record, event_type=event_type)
        if not schema_ok:
            return -10.0, f"Invalid Schema: {msg}"

        # 2. Get required fields for this event type
        required_schema = self.required_fields.get(event_type, self.base_required)
        required = list(required_schema.keys()) if isinstance(required_schema, dict) else required_schema
        
        # 3. Format Checks only for fields that are:
        #    a) Present in the record AND
        #    b) Required for this event type
        if "Email" in required:
            email = record.get('Email', '')
            if email and not self.validate_email(email):
                return -5.0, "Invalid Email Format"
        
        if "Phone" in required:
            phone = record.get('Phone', '')
            if phone and not self.validate_phone(phone):
                return -2.0, f"Invalid Phone Format: {phone}"

        # 4. Duplicate Check (only for events that have Email field as required)
        if "Email" in required:
            email = record.get('Email', '')
            if email and self.is_duplicate(record):
                return -20.0, "Duplicate Record Detected (Risk: Dirty Data)"

        # 5. Event-specific validation reward boost
        bonus = 0.0
        if event_type == "offer" and record.get("Offer_Amount") and record.get("Expected_Revenue"):
            bonus = 5.0  # Offer data complete
        elif event_type == "contract" and record.get("Contract_Terms") and record.get("Contract_Value"):
            bonus = 5.0  # Contract data complete
        elif event_type == "maintenance" and record.get("Ticket_ID") and record.get("Priority"):
            bonus = 3.0  # Maintenance ticket complete

        # Success!
        return 10.0 + bonus, f"Valid {event_type} ready for CRM"

    def commit(self, record: Dict, event_type: str = "new_lead") -> bool:
        """
        'Saves' the record to the mock database if valid.
        """
        reward, reason = self.assess_record(record, event_type=event_type)
        if reward > 0:
            self.mock_database.add(record.get("Email"))
            logger.info(f"Committed {event_type}: {record.get('Email')}")
            return True
        else:
            logger.warning(f"Failed to commit: {reason}")
            return False

if __name__ == "__main__":
    v = CRMValidator()
    
    # Test Good New Lead
    good_lead = {
        "First_Name": "John",
        "Last_Name": "Doe",
        "Email": "john@example.com",
        "Phone": "+1-555-010-9999",
        "Company": "Test Co"
    }
    print(f"Good Lead: {v.assess_record(good_lead, event_type='new_lead')}")
    
    # Test Good Offer
    good_offer = {
        "First_Name": "John",
        "Last_Name": "Doe",
        "Email": "john@example.com",
        "Phone": "+1-555-010-9999",
        "Offer_Amount": "$50000",
        "Expected_Revenue": "$100000"
    }
    print(f"Good Offer: {v.assess_record(good_offer, event_type='offer')}")
    
    # Test Bad Record
    bad = {
        "First_Name": "Jane", 
        # Missing Last Name
        "Email": "jane.com", # Bad Email
        "Phone": "123"
    }
    print(f"Bad: {v.assess_record(bad, event_type='new_lead')}")


