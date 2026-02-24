import json
import random
import string
import os
from typing import List, Dict

# Define paths to input examples
INPUT_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "input_data")

# Load base templates from JSON files
def load_templates():
    templates = {}
    try:
        for filename in os.listdir(INPUT_EXAMPLES_DIR):
            if filename.endswith(".json"):
                event_type = filename.replace(".json", "")
                with open(os.path.join(INPUT_EXAMPLES_DIR, filename), "r") as f:
                    templates[event_type] = json.load(f)
    except FileNotFoundError:
        # Fallback empty templates if directory not found yet
        pass
    return templates

EVENT_TEMPLATES = load_templates()

# Target Schema (Zoho CRM) - kept for reference, but data structure is changing
TARGET_KEYS = ["Deal_Name", "Email", "Phone", "Company"]

# Event Types
EVENT_TYPES = list(EVENT_TEMPLATES.keys()) if EVENT_TEMPLATES else ["new_lead", "offer", "contract", "maintenance"]


# Source Variations
SOURCES = ["Gimlet", "Bonasera", "Hubspot"]

SOURCE_MAPPINGS = {
    "Gimlet": {
        "First_Name": "fname",
        "Last_Name": "lname",
        "Email": "email_address",
        "Phone": "contact",
        "Company": "org_name",
        "Lead_Source": "source"
    },
    "Bonasera": {
        "First_Name": "Nombre",
        "Last_Name": "Apellidos",
        "Email": "Correo",
        "Phone": "Telefono",
        "Company": "Empresa",
        "Lead_Source": "Origen"
    },
    "Hubspot": {
        "First_Name": "firstname",
        "Last_Name": "lastname",
        "Email": "email",
        "Phone": "phone",
        "Company": "company",
        "Lead_Source": "hs_analytics_source"
    }
}

def random_string(length=8):
    return ''.join(random.choices(string.ascii_letters, k=length))

def random_phone():
    formats = [
        f"+1-{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}",
        f"({random.randint(200,999)}) {random.randint(200,999)}-{random.randint(1000,9999)}",
        f"{random.randint(200,999)}{random.randint(200,999)}{random.randint(1000,9999)}", # No format
        f"invalid-phone-{random.randint(100,999)}" # Invalid
    ]
    return random.choice(formats)

def random_email(name):
    domains = ["example.com", "test.org", "company.net", "inv@alid"]
    if random.random() < 0.1: # 10% chance of bad email
        return f"{name}at{random.choice(domains)}"
    return f"{name}@{random.choice(domains)}"

def generate_golden_record(event_type: str = None):
    """Generates a record based on event templates."""
    if event_type is None:
        event_type = random.choice(EVENT_TYPES)
    
    # Check if we have a template for this event type
    if event_type not in EVENT_TEMPLATES:
        # Fallback to old behavior if no template
        return {} 

    # Deep copy the template to avoid modifying the global constant
    template = json.loads(json.dumps(EVENT_TEMPLATES[event_type]))
    
    # Fill with random data
    fname = random_string(5).capitalize()
    
    # Update Lead Info (common to all)
    if "eventData" in template and "Lead" in template["eventData"]:
        template["eventData"]["Lead"]["Deal_Name"] = fname
        template["eventData"]["Lead"]["Email"] = random_email(fname.lower())
        template["eventData"]["Lead"]["Phone"] = random_phone()
        
    return template

# Legacy function kept for compatibility if needed, but not used for new format
def dirty_record(record: Dict, source: str = "Gimlet") -> Dict:
    """
    Simulates noise on the new JSON structure. 
    """
    # Simply return the record as-is for now, or add robust noise logic for nested dicts later
    return record

def generate_batch(size=10, event_types=None) -> List[Dict]:
    """Generates a batch of raw input data using templates."""
    if event_types is None:
        event_types = EVENT_TYPES
    
    batch = []
    for _ in range(size):
        event_type = random.choice(event_types)
        clean_rec = generate_golden_record(event_type=event_type)
        # We can add noise here if needed
        batch.append(clean_rec)
    return batch

if __name__ == "__main__":
    # Test generation
    data = generate_batch(3)
    print(json.dumps(data, indent=2))
