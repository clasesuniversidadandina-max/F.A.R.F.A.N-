import pytest
from src.policy_processor import PolicyAnalysisPipeline, ProcessorConfig
import os

def test_policy_analysis_pipeline_initialization():
    """Tests the initialization of the PolicyAnalysisPipeline."""
    pipeline = PolicyAnalysisPipeline()
    assert pipeline is not None
    assert pipeline.config is not None

def test_policy_analysis_pipeline_simple_analysis():
    """Tests a simple analysis run with the pipeline."""
    config = ProcessorConfig(confidence_threshold=0.5)
    pipeline = PolicyAnalysisPipeline(config=config)
    text = "Este es un texto de prueba suficientemente largo para el diagnóstico cuantitativo. El análisis de la situación actual muestra una brecha en la capacidad institucional. Se necesita un plan de acción detallado con un cronograma de ejecución claro para abordar los problemas identificados en el diagnóstico situacional."
    results = pipeline.analyze_text(text)
    assert results is not None
    assert results["processing_status"] == "complete"
    assert "point_evidence" in results

def test_end_to_end_pdf_analysis():
    """Tests the full end-to-end analysis of a sample PDF."""
    pipeline = PolicyAnalysisPipeline()
    # Create a dummy PDF for testing
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    file_path = "tests/sample_plan.pdf"
    c = canvas.Canvas(file_path, pagesize=letter)
    c.drawString(72, 800, "Plan de Desarrollo Municipal")
    c.drawString(72, 780, "Este es un plan de desarrollo de prueba.")
    c.drawString(72, 760, "El diagnóstico cuantitativo muestra la necesidad de mejorar la infraestructura.")
    c.drawString(72, 740, "Se propone un plan de acción detallado con un presupuesto de $1000 millones.")
    c.drawString(72, 720, "La Secretaría de Planeación será la responsable de la ejecución.")
    c.save()

    results = pipeline.analyze_file(file_path)
    assert results is not None
    assert results["processing_status"] == "complete"
    assert "metadata" in results
    assert "point_evidence" in results
    assert "dimension_analysis" in results
    assert results["document_statistics"]["character_count"] > 100
    os.remove(file_path)
