#!/usr/bin/env python3
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import textwrap

pdfmetrics.registerFont(TTFont('DejaVu', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVu-Bold', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

def create_disclosure_pdf(output_path: str = "vulnerability_disclosure.pdf"):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    
    margin = 1 * inch
    y = height - margin
    line_height = 14
    max_width = 85
    
    # Title
    c.setFont("DejaVu-Bold", 14)
    c.drawString(margin, y, "Vulnerability Disclosure Policy")
    y -= 0.4 * inch
    
    # Body text
    c.setFont("DejaVu", 11)
    main_text = """Megacorp One AI's vulnerability disclosure process begins with identification of potential security issues within systems, applications, or infrastructure. All discovered vulnerabilities must be documented with severity ratings, affected components, and reproduction steps. Security researchers are required to report findings through the designated vulnerability intake portal at security@megacorpone.ai. Initial triage occurs within 48 hours of submission. Megacorp One AI maintains a 90-day disclosure deadline from initial report to public notification, allowing adequate time for remediation efforts. The security team coordinates with affected product teams to develop patches. Researchers who follow the responsible disclosure policy are recognized in the security advisory acknowledgments section. Megacorp One AI commits to treating all vulnerability reports with confidentiality until patches are deployed to production environments. Security researchers demonstrating good faith in the disclosure process may be eligible for recognition in official security bulletins. The organization provides regular updates to reporters on remediation progress and expected patch timelines. Upon resolution, Megacorp One AI publishes a detailed security advisory documenting the vulnerability, its impact, affected versions, and mitigation strategies. Researchers are encouraged to verify patch effectiveness in staging environments before production release. This collaborative approach ensures vulnerabilities are addressed systematically while protecting enterprise security posture."""
    
    for paragraph in main_text.split('\n'):
        if paragraph.strip() == '':
            y -= line_height
            continue
        
        wrapped = textwrap.wrap(paragraph, width=max_width)
        for line in wrapped:
            if y < margin:
                c.showPage()
                c.setFont("DejaVu", 11)
                y = height - margin
            c.drawString(margin, y, line)
            y -= line_height
    
    c.save()

if __name__ == "__main__":
    create_disclosure_pdf("MC1_Vulnerability_Disclosure.pdf")
