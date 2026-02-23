# Zoho CRM Data Integration Instructions

This document defines the required format for JSON data to be successfully created in Zoho CRM by event type.

## Event Types

### 1. New Lead (`new_lead`)
**Purpose**: Import a new prospect into the CRM.

**Required Fields**:
- `First_Name` (string)
- `Last_Name` (string)
- `Email` (string) - valid email format
- `Phone` (string) - valid international format

**Optional Fields**:
- `Company` (string)
- `Lead_Source` (string) - e.g., 'Gimlet', 'Bonasera', 'Hubspot'

---

### 2. Sales Offer (`offer`)
**Purpose**: Create an offer record linked to an existing lead.

**Required Fields** (All new_lead fields plus):
- `Offer_Amount` (string) - e.g., "$50,000"
- `Expected_Revenue` (string) - e.g., "$100,000"

**Optional Fields**:
- `Offer_Date` (date) - ISO Date String
- `Validity_Period` (number) - Days
- `Company` (string)

---

### 3. Contract (`contract`)
**Purpose**: Record a signed contract or agreement.

**Required Fields** (All new_lead fields plus):
- `Contract_Terms` (string) - e.g., "12 months", "24 months"
- `Contract_Value` (string) - e.g., "$500,000"

**Optional Fields**:
- `Contract_Date` (date) - ISO Date String
- `Start_Date` (date) - ISO Date String
- `End_Date` (date) - ISO Date String
- `Company` (string)

---

### 4. Maintenance Ticket (`maintenance`)
**Purpose**: Create a support or maintenance ticket.

**Required Fields** (Modified from new_lead):
- `First_Name` (string) - Customer name
- `Last_Name` (string)
- `Ticket_ID` (string) - Unique ticket identifier
- `Issue_Type` (string) - e.g., "bug", "feature", "support", "billing"
- `Priority` (string) - "low", "medium", "high", "critical"

**Optional Fields**:
- `Email` (string) - Contact email
- `Phone` (string) - Contact phone
- `Description` (string)
- `Assigned_To` (string)
- `Created_Date` (date) - ISO Date String

---

## Data Validation Rules

1.  **Email Format**: Must match standard regex for valid email addresses (e.g., `user@domain.com`).
   - Regex: `/^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$/`

2.  **Phone Format**: Should be normalized to format like `+1-555-010-9999` (international).
   - Regex: `/^\+\d{1,3}-\d{3}-\d{3}-\d{4}$/`

3.  **Deduplication** (new_lead & offer): Records with duplicate `Email` are rejected or merged.

4.  **Event Type Detection**: The incoming JSON should include or the system will infer an `event_type` field.

---

## Transformation Logic (Agent Policy)

The RL agent is responsible for:

1.  **Detection**: Identify event type from input metadata or data structure.
2.  **Mapping**: Identify source-specific keys and rename to target keys.
   - Example: `fname` → `First_Name`, `contact` → `Phone`
3.  **Cleaning**: 
   - Remove invalid characters from phone numbers
   - Capitalize names (First letter uppercase)
   - Trim whitespace
4.  **Validation**: Ensure all required fields for the event type are present and valid.
5.  **Filtration**: Discard records that cannot be fixed (missing critical fields).

---

## Data Source Mappings

### Gimlet Format
```json
{
  "fname": "John",
  "lname": "Doe",
  "email_address": "john@example.com",
  "contact": "+1-555-0101",
  "org_name": "ACME Corp",
  "source": "Gimlet"
}
```

### Bonasera Format
```json
{
  "Nombre": "Juan",
  "Apellidos": "García",
  "Correo": "juan@example.com",
  "Telefono": "555-0101",
  "Empresa": "ACME",
  "Origen": "Bonasera"
}
```

### Hubspot Format
```json
{
  "firstname": "Jane",
  "lastname": "Smith",
  "email": "jane@example.com",
  "phone": "5550101",
  "company": "ACME Inc",
  "hs_analytics_source": "Hubspot"
}
```

---

## Example Processing

**Input (Gimlet - new_lead)**:
```json
{
  "fname": "john ",
  "lname": "doe",
  "email_address": "john@example.com",
  "contact": "5550101",
  "org_name": "ACME",
  "event_type": "new_lead"
}
```

**After Agent Processing**:
```json
{
  "First_Name": "John",
  "Last_Name": "Doe",
  "Email": "john@example.com",
  "Phone": "+1-555-010-1",
  "Company": "ACME",
  "Lead_Source": "Gimlet",
  "event_type": "new_lead"
}
```

**Output (Ready for Zoho CRM)**:
```json
{
  "First_Name": "John",
  "Last_Name": "Doe",
  "Email": "john@example.com",
  "Phone": "+1-555-010-1",
  "Company": "ACME"
}
```


