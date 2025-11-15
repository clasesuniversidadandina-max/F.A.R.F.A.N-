import pytest
from decimal import Decimal
from src.financiero_viabilidad_tablas import QualityScore, FinancialIndicator, ResponsibleEntity

def test_quality_score_creation():
    """Tests the creation of a QualityScore object."""
    score = QualityScore(
        overall_score=8.5,
        financial_feasibility=9.0,
        indicator_quality=8.0,
        responsibility_clarity=7.5,
        temporal_consistency=8.0,
        pdet_alignment=9.5,
        causal_coherence=8.5,
        confidence_interval=(8.0, 9.0),
        evidence={}
    )
    assert score.overall_score == 8.5
    assert score.pdet_alignment == 9.5

def test_financial_indicator_creation():
    """Tests the creation of a FinancialIndicator object."""
    indicator = FinancialIndicator(
        source_text="$100 millones",
        amount=Decimal("100000000"),
        currency="COP",
        fiscal_year=2024,
        funding_source="SGP",
        budget_category="Educación",
        execution_percentage=None,
        confidence_interval=(0.9, 0.95),
        risk_level=0.1
    )
    assert indicator.amount == Decimal("100000000")
    assert indicator.funding_source == "SGP"

def test_responsible_entity_creation():
    """Tests the creation of a ResponsibleEntity object."""
    entity = ResponsibleEntity(
        name="Secretaría de Educación",
        entity_type="secretaría",
        specificity_score=0.9,
        mentioned_count=10,
        associated_programs=["Programa A"],
        associated_indicators=["Indicador X"],
        budget_allocated=Decimal("50000000")
    )
    assert entity.name == "Secretaría de Educación"
    assert entity.mentioned_count == 10
