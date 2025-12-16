"""
PDF Report Generator - Medical Summary Template for Hospital XYZ

Generates professional medical triage reports in PDF format using ReportLab.
"""

from io import BytesIO
from datetime import datetime
from typing import Dict, Any, List
import textwrap

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY


# Color scheme
HOSPITAL_BLUE = HexColor("#1e40af")
HOSPITAL_LIGHT_BLUE = HexColor("#dbeafe")
HEADER_GRAY = HexColor("#374151")
TEXT_GRAY = HexColor("#4b5563")
RISK_COLORS = {
    "red": HexColor("#dc2626"),
    "amber": HexColor("#d97706"),
    "green": HexColor("#16a34a"),
}


def wrap_text(text: str, width: int = 80) -> List[str]:
    """Wrap text to specified width."""
    return textwrap.wrap(text, width=width)


def draw_header(c: canvas.Canvas, width: float, height: float):
    """Draw hospital header with logo placeholder."""
    # Header background
    c.setFillColor(HOSPITAL_BLUE)
    c.rect(0, height - 1.2*inch, width, 1.2*inch, fill=True, stroke=False)

    # Hospital name
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 22)
    c.drawString(0.75*inch, height - 0.55*inch, "HOSPITAL XYZ")

    # Subtitle
    c.setFont("Helvetica", 10)
    c.drawString(0.75*inch, height - 0.8*inch, "Emergency Department - Medical Triage Report")

    # Contact info on right
    c.setFont("Helvetica", 9)
    c.drawRightString(width - 0.75*inch, height - 0.55*inch, "Emergency: 911")
    c.drawRightString(width - 0.75*inch, height - 0.75*inch, "www.hospitalxyz.com")


def draw_risk_badge(c: canvas.Canvas, risk_band: str, x: float, y: float):
    """Draw colored risk level badge."""
    color = RISK_COLORS.get(risk_band.lower(), RISK_COLORS["amber"])

    # Badge background
    badge_width = 1.5*inch
    badge_height = 0.4*inch
    c.setFillColor(color)
    c.roundRect(x, y - badge_height/2, badge_width, badge_height, 5, fill=True, stroke=False)

    # Badge text
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(x + badge_width/2, y - 4, risk_band.upper())


def draw_section_header(c: canvas.Canvas, text: str, y: float, width: float) -> float:
    """Draw a section header with underline."""
    c.setFillColor(HOSPITAL_BLUE)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75*inch, y, text)

    # Underline
    c.setStrokeColor(HOSPITAL_BLUE)
    c.setLineWidth(1)
    c.line(0.75*inch, y - 3, width - 0.75*inch, y - 3)

    return y - 25


def draw_key_value(c: canvas.Canvas, key: str, value: str, x: float, y: float) -> float:
    """Draw a key-value pair."""
    c.setFillColor(TEXT_GRAY)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, f"{key}:")
    c.setFont("Helvetica", 10)
    c.drawString(x + 100, y, str(value))
    return y - 16


def draw_multiline_text(c: canvas.Canvas, text: str, x: float, y: float,
                        max_width: int = 85, line_height: int = 14) -> float:
    """Draw wrapped text and return new y position."""
    c.setFillColor(TEXT_GRAY)
    c.setFont("Helvetica", 10)

    lines = wrap_text(text, max_width)
    for line in lines:
        c.drawString(x, y, line)
        y -= line_height

    return y


def draw_bullet_list(c: canvas.Canvas, items: List[str], x: float, y: float,
                     line_height: int = 14) -> float:
    """Draw a bullet list."""
    c.setFillColor(TEXT_GRAY)
    c.setFont("Helvetica", 10)

    for item in items:
        # Wrap long items
        lines = wrap_text(item, 80)
        for i, line in enumerate(lines):
            if i == 0:
                c.drawString(x, y, f"  {line}")
            else:
                c.drawString(x + 15, y, line)
            y -= line_height

    return y


def draw_footer(c: canvas.Canvas, width: float, page_num: int, ticket_id: str):
    """Draw page footer."""
    # Footer line
    c.setStrokeColor(HOSPITAL_BLUE)
    c.setLineWidth(0.5)
    c.line(0.75*inch, 0.6*inch, width - 0.75*inch, 0.6*inch)

    # Footer text
    c.setFillColor(TEXT_GRAY)
    c.setFont("Helvetica", 8)
    c.drawString(0.75*inch, 0.4*inch, f"Ticket ID: {ticket_id}")
    c.drawCentredString(width/2, 0.4*inch, f"Page {page_num}")
    c.drawRightString(width - 0.75*inch, 0.4*inch,
                      f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    # Disclaimer
    c.setFont("Helvetica-Oblique", 7)
    c.drawCentredString(width/2, 0.25*inch,
                        "This document is for medical triage purposes only. Clinical judgment required.")


def generate_medical_report_pdf(
    ticket_id: str,
    risk_band: str,
    demographics: Dict[str, Any],
    summary: str,
    diagnosis: str,
    conclusion: str,
    key_flags: List[str],
    triggered_rules: List[Dict[str, str]],
    answers: Dict[str, Any],
    waiting_instruction: str
) -> bytes:
    """
    Generate a professional medical triage report PDF.

    Args:
        ticket_id: Unique ticket identifier
        risk_band: Risk level (red/amber/green)
        demographics: Patient demographic data
        summary: Clinical summary text
        diagnosis: Diagnosis section from LLM
        conclusion: Conclusion/recommendations from LLM
        key_flags: List of clinical red flags
        triggered_rules: List of triggered risk rules
        answers: Question-answer pairs from triage
        waiting_instruction: Instructions for patient

    Returns:
        PDF file as bytes
    """
    buffer = BytesIO()
    width, height = letter
    c = canvas.Canvas(buffer, pagesize=letter)

    # Page 1
    draw_header(c, width, height)

    y = height - 1.6*inch

    # Ticket info and risk level row
    c.setFillColor(HEADER_GRAY)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(0.75*inch, y, f"Triage Report - {ticket_id}")

    # Risk badge on the right
    draw_risk_badge(c, risk_band, width - 2.5*inch, y + 5)

    y -= 40

    # Patient Information Section
    y = draw_section_header(c, "PATIENT INFORMATION", y, width)

    age = demographics.get("age", "N/A")
    sex = demographics.get("sex", "N/A")
    pregnant = demographics.get("pregnant")

    y = draw_key_value(c, "Age", str(age), 0.75*inch, y)
    y = draw_key_value(c, "Sex", str(sex).capitalize(), 0.75*inch, y)
    if pregnant is not None:
        y = draw_key_value(c, "Pregnant", "Yes" if pregnant else "No", 0.75*inch, y)

    chief_complaint = answers.get("chief_complaint", "Not specified")
    y = draw_key_value(c, "Chief Complaint", str(chief_complaint).replace("_", " ").title(),
                       0.75*inch, y)

    y -= 15

    # Clinical Summary Section
    y = draw_section_header(c, "CLINICAL SUMMARY", y, width)
    y = draw_multiline_text(c, summary, 0.75*inch, y)
    y -= 15

    # Diagnosis Section
    if diagnosis:
        y = draw_section_header(c, "DIAGNOSIS", y, width)
        y = draw_multiline_text(c, diagnosis, 0.75*inch, y)
        y -= 15

    # Check if we need a new page
    if y < 3*inch:
        draw_footer(c, width, 1, ticket_id)
        c.showPage()
        draw_header(c, width, height)
        y = height - 1.6*inch

    # Key Flags Section
    if key_flags:
        y = draw_section_header(c, "KEY CLINICAL FLAGS", y, width)
        y = draw_bullet_list(c, key_flags, 0.75*inch, y)
        y -= 15

    # Check page break
    if y < 3*inch:
        draw_footer(c, width, 1, ticket_id)
        c.showPage()
        draw_header(c, width, height)
        y = height - 1.6*inch

    # Triggered Rules Section
    if triggered_rules:
        y = draw_section_header(c, "TRIGGERED RISK RULES", y, width)
        rules_text = [f"{r.get('rule', 'N/A')}: {r.get('description', '')}"
                      for r in triggered_rules]
        y = draw_bullet_list(c, rules_text, 0.75*inch, y)
        y -= 15

    # Check page break
    if y < 2.5*inch:
        draw_footer(c, width, 1, ticket_id)
        c.showPage()
        draw_header(c, width, height)
        y = height - 1.6*inch

    # Conclusion Section
    if conclusion:
        y = draw_section_header(c, "CONCLUSION & RECOMMENDATIONS", y, width)
        y = draw_multiline_text(c, conclusion, 0.75*inch, y)
        y -= 15

    # Instructions Box
    if y < 2*inch:
        draw_footer(c, width, 1, ticket_id)
        c.showPage()
        draw_header(c, width, height)
        y = height - 1.6*inch

    # Waiting Instructions Box
    box_height = 0.8*inch
    c.setFillColor(HOSPITAL_LIGHT_BLUE)
    c.roundRect(0.75*inch, y - box_height, width - 1.5*inch, box_height, 5,
                fill=True, stroke=False)

    c.setFillColor(HOSPITAL_BLUE)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(1*inch, y - 20, "PATIENT INSTRUCTIONS:")
    c.setFont("Helvetica", 10)
    c.setFillColor(TEXT_GRAY)

    # Wrap instruction text
    instruction_lines = wrap_text(waiting_instruction, 90)
    inst_y = y - 35
    for line in instruction_lines[:2]:  # Max 2 lines
        c.drawString(1*inch, inst_y, line)
        inst_y -= 14

    # Footer
    draw_footer(c, width, 1, ticket_id)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# Standalone test
if __name__ == "__main__":
    # Test PDF generation
    pdf_bytes = generate_medical_report_pdf(
        ticket_id="T-1234-ABC123",
        risk_band="amber",
        demographics={"age": 45, "sex": "male", "pregnant": None},
        summary="Patient presents with moderate chest discomfort radiating to left arm. "
                "No acute distress observed. Vital signs within normal limits.",
        diagnosis="Possible angina pectoris. Differential includes musculoskeletal pain, "
                  "anxiety-related chest tightness, or early cardiac event.",
        conclusion="Recommend ECG and cardiac enzyme panel. Patient should be monitored "
                   "closely. Consider cardiology consultation if symptoms persist or worsen.",
        key_flags=["Chest pain radiating to arm", "Age > 40", "Male patient"],
        triggered_rules=[
            {"rule": "CHEST_PAIN_ARM", "description": "Chest pain with arm radiation"},
            {"rule": "AGE_RISK", "description": "Age-related cardiac risk factor"}
        ],
        answers={"chief_complaint": "chest_discomfort", "pain_location": "left arm"},
        waiting_instruction="Please wait in the priority waiting area. You will be seen within 15 minutes."
    )

    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Test PDF generated: test_report.pdf")
