import json
import random
import string
from typing import List, Dict

# Target Schema (Zoho CRM)
TARGET_KEYS = ["First_Name", "Last_Name", "Email", "Phone", "Company", "Lead_Source"]

# Event Types
EVENT_TYPES = ["new_lead", "offer", "contract", "maintenance"]

# Event-Specific Required Fields with Data Types
EVENT_SCHEMAS = {
    "new_lead": {
        "First_Name": "string",
        "Last_Name": "string",
        "Email": "string",
        "Phone": "string"
    },
    "offer": {
        "First_Name": "string",
        "Last_Name": "string",
        "Email": "string",
        "Phone": "string",
        "Offer_Amount": "string",
        "Expected_Revenue": "string"
    },
    "contract": {
        "First_Name": "string",
        "Last_Name": "string",
        "Email": "string",
        "Phone": "string",
        "Contract_Terms": "string",
        "Contract_Value": "string"
    },
    "maintenance": {
        "First_Name": "string",
        "Last_Name": "string",
        "Ticket_ID": "string",
        "Issue_Type": "string",
        "Priority": "string"
    }
}

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
    """Generates a perfect, valid record with event-specific fields."""
    if event_type is None:
        event_type = random.choice(EVENT_TYPES)
    
    fname = random_string(5).capitalize()
    record = {
        "First_Name": fname,
        "Last_Name": random_string(6).capitalize(),
        "Email": random_email(fname.lower()),
        "Phone": "+1-555-010-9999",  # Standard format
        "Company": random_string(10) + " Inc.",
        "Lead_Source": "Generated",
        "event_type": event_type
    }
    
    # Add event-specific fields
    if event_type in ["offer", "contract", "new_lead"]:
        if event_type == "offer":
            record["Offer_Amount"] = f"${random.randint(1000, 100000)}"
            record["Expected_Revenue"] = f"${random.randint(1000, 500000)}"
        elif event_type == "contract":
            record["Contract_Terms"] = f"{random.randint(6, 36)} months"
            record["Contract_Value"] = f"${random.randint(10000, 1000000)}"
        # new_lead uses base fields only
    
    elif event_type == "maintenance":
        record.pop("Company", None)  # Maintenance may not need company
        record["Ticket_ID"] = f"TKT-{random.randint(100000, 999999)}"
        record["Issue_Type"] = random.choice(["bug", "feature", "support", "billing"])
        record["Priority"] = random.choice(["low", "medium", "high", "critical"])
    
    return record

def dirty_record(record: Dict, source: str) -> Dict:
    """
    Takes a clean record and transforms it into a 'dirty' source format.
    Preserves event_type and event-specific fields.
    """
    mapping = SOURCE_MAPPINGS[source]
    dirty = {}
    
    # Preserve event_type and event-specific fields
    event_type = record.get('event_type', 'new_lead')
    dirty['event_type'] = event_type
    
    # Preserve event-specific fields (they don't get mapped)
    for field in ['Offer_Amount', 'Expected_Revenue', 'Contract_Terms', 'Contract_Value', 
                  'Ticket_ID', 'Issue_Type', 'Priority']:
        if field in record:
            dirty[field] = record[field]
    
    for target_key, source_key in mapping.items():
        value = record.get(target_key)
        
        # Introduce noise
        if target_key == "Phone":
            value = random_phone()
        
        if random.random() < 0.05: # 5% chance of missing value
            value = ""
        elif random.random() < 0.05: # 5% chance of none
            value = None
            
        dirty[source_key] = value
        
    # Introduce extra unmapped fields
    dirty["_meta_id"] = random.randint(1000, 9999)
    dirty["timestamp"] = "2023-10-27T10:00:00Z"
    
    return dirty

def generate_batch(size=10, event_types=None) -> List[Dict]:
    """Generates a batch of raw input data with event types."""
    if event_types is None:
        event_types = EVENT_TYPES
    
    batch = []
    for _ in range(size):
        source = random.choice(SOURCES)
        event_type = random.choice(event_types)
        clean_rec = generate_golden_record(event_type=event_type)
        dirty_rec = dirty_record(clean_rec, source)
        batch.append(dirty_rec)
    return batch

if __name__ == "__main__":
    # Test generation
    data = generate_batch(3)
    print(json.dumps(data, indent=2))
