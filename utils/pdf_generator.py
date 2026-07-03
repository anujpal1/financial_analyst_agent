import os
import re
from fpdf import FPDF
from config import SYSTEM_LOGGER as logger

class FinancialReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 12)
        # Using text= for compatibility with newer fpdf2
        self.cell(0, 10, text="INSTITUTIONAL RESEARCH DIRECTIVE - CONFIDENTIAL", border=0, align="C")
        # ln=1 in older fpdf is handled by ln() or move position, let's just use ln()
        self.ln(10)
        self.line(10, 18, 200, 18)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, text=f"Page {self.page_no()}/{{nb}}", border=0, align="C")

def export_markdown_to_pdf(markdown_content: str, output_path: str) -> str:
    """Converts the agent's raw Markdown report into a clean, formatted PDF document."""
    logger.info(f"Compiling PDF export generation for path: {output_path}")
    try:
        pdf = FinancialReportPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_font("Helvetica", size=10)
        
        # 1. Strip out long continuous markdown dividers that crash FPDF
        cleaned_text = re.sub(r'[-=_*]{5,}', '', markdown_content)
        
        # 2. Normalize structural tags 
        clean_lines = cleaned_text.replace("###", "").replace("##", "").replace("**", "").replace("#", "").split("\n")
        
        for line in clean_lines:
            if line.strip():
                # 3. FIX: Changed 'txt' to 'text' to match the new fpdf2 API requirements
                pdf.multi_cell(0, 6, text=line.encode('latin-1', 'replace').decode('latin-1'))
            else:
                pdf.ln(4)
                
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pdf.output(output_path)
        logger.info(f"PDF statement successfully built and written to disk at {output_path}")
        return f"Success: PDF generated at {output_path}"
    except Exception as e:
        logger.error(f"PDF compiler crashed: {str(e)}")
        return f"PDF generation error: {str(e)}"