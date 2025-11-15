from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def create_sample_pdf(file_path):
    """Creates a sample PDF for testing."""
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(72, 800, "Plan de Desarrollo Municipal")
    c.drawString(72, 780, "Este es un plan de desarrollo de prueba.")
    c.drawString(72, 760, "El diagnóstico cuantitativo muestra la necesidad de mejorar la infraestructura.")
    c.drawString(72, 740, "Se propone un plan de acción detallado con un presupuesto de $1000 millones.")
    c.drawString(72, 720, "La Secretaría de Planeación será la responsable de la ejecución.")
    c.save()

if __name__ == "__main__":
    create_sample_pdf("tests/sample_plan.pdf")
