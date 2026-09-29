"""
Creates the ReqLens AI test SRS PDF with intentional inconsistencies.
Run once: python data/create_srs_pdf.py
Requires: pip install pymupdf
"""
import fitz  # PyMuPDF
import os

SRS_TEXT = """ReqLens AI Test SRS - Inconsistency Recommendation Test
Version 1.0

1. Introduction
This Software Requirements Specification defines requirements for a university
event management system. It intentionally includes contradictions and
inconsistencies for testing the ReqLens AI analysis pipeline.

2. User Registration Requirements

REQ-001: Students shall be allowed to register for public university events.

REQ-002: Only administrators shall be allowed to register students for public
university events.

REQ-003: The system shall allow registered users to view all public events.

REQ-004: The system shall allow registered users to view only events they are
enrolled in.

3. Performance Requirements

REQ-017: The system shall process all search queries within 500 milliseconds
under normal operating conditions.

REQ-018: The system shall process all search queries within 200 milliseconds
to ensure acceptable user experience.

REQ-019: The system shall support at least 1000 concurrent users.

REQ-020: The system shall support at least 500 concurrent users during peak hours.

4. Response Time Requirements

REQ-031: The system shall respond to user requests within 300 ms.

REQ-032: The system shall respond to API requests within 500 ms.

5. Authentication Requirements

REQ-033: Users shall authenticate using a username and password before
accessing any restricted content.

REQ-034: The system shall require two-factor authentication for all user
logins to protected areas.

REQ-035: Guest users shall be allowed to view public event listings without
authentication.

6. Data Retention Requirements

REQ-036: The system shall retain user data for a minimum of 1 year after
account deletion.

REQ-037: The system shall delete all user data within 30 days of account
deletion to comply with GDPR.

REQ-038: The system shall log all user actions for audit purposes.

REQ-038: The system shall log only failed authentication attempts to reduce
storage costs.

7. Payment Requirements

REQ-039: All event registrations shall require payment before confirmation.

REQ-040: Free events shall allow registration without payment.

REQ-041: The system shall notify users of successful payment within 5 seconds.

REQ-042: The system shall notify users of successful payment within 2 seconds.

8. Access Control Requirements

REQ-043: Faculty members shall be allowed to create and manage university events.

REQ-044: Only administrators shall be allowed to create university events.

REQ-045: Students shall not be allowed to modify event registrations after
the registration deadline has passed.

REQ-046: Students shall be allowed to cancel their event registration at any
time, including after the deadline.
"""

output_path = os.path.join(os.path.dirname(__file__),
                           "ReqLens_SRS_Inconsistency_Recommendation_Test.pdf")

doc = fitz.open()
page = doc.new_page()

# Write text in chunks that fit on the page
fontsize = 10
margin = 50
y = margin
line_height = fontsize * 1.4

for line in SRS_TEXT.split("\n"):
    if y > page.rect.height - margin:
        page = doc.new_page()
        y = margin
    page.insert_text((margin, y), line, fontsize=fontsize)
    y += line_height

doc.save(output_path)
doc.close()
print(f"SRS PDF created: {output_path}")
print(f"File size: {os.path.getsize(output_path)} bytes")
