import pytest
from src.Analyzer_one import MunicipalOntology, ValueChainLink

def test_value_chain_link_creation():
    """Tests the creation of a ValueChainLink object."""
    link = ValueChainLink(
        name="test_link",
        instruments=["inst1", "inst2"],
        mediators=["med1"],
        outputs=["out1"],
        outcomes=["outcome1"],
        bottlenecks=["b1"],
        lead_time_days=10,
        conversion_rates={"rate1": 0.5},
        capacity_constraints={"cap1": 0.8}
    )
    assert link.name == "test_link"
    assert link.lead_time_days == 10
    assert "inst1" in link.instruments

def test_municipal_ontology_creation():
    """Tests the creation and basic structure of the MunicipalOntology."""
    ontology = MunicipalOntology()
    assert "diagnostic_planning" in ontology.value_chain_links
    assert "economic_development" in ontology.policy_domains
    assert "governance" in ontology.cross_cutting_themes
    assert isinstance(ontology.value_chain_links["diagnostic_planning"], ValueChainLink)
