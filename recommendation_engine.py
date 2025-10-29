# recommendation_engine.py - Rule-Based Recommendation Engine
# coding=utf-8
"""
Recommendation Engine - Multi-Level Rule-Based Recommendations
================================================================

This module implements a rule-based recommendation engine that:
1. Loads and validates recommendation rules from JSON files
2. Evaluates conditions against score data at MICRO, MESO, and MACRO levels
3. Generates actionable recommendations with specific interventions
4. Renders templates with context-specific variable substitution

Supports three levels of recommendations:
- MICRO: Question-level recommendations (PA-DIM combinations)
- MESO: Cluster-level recommendations (CL01-CL04)
- MACRO: Plan-level strategic recommendations

Author: Integration Team
Version: 1.0.0
Python: 3.10+
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime, timezone
import jsonschema

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES FOR RECOMMENDATIONS
# ============================================================================

@dataclass
class Recommendation:
    """
    Structured recommendation with full intervention details
    """
    rule_id: str
    level: str  # MICRO, MESO, or MACRO
    problem: str
    intervention: str
    indicator: Dict[str, Any]
    responsible: Dict[str, Any]
    horizon: Dict[str, str]
    verification: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class RecommendationSet:
    """
    Collection of recommendations with metadata
    """
    level: str
    recommendations: List[Recommendation]
    generated_at: str
    total_rules_evaluated: int
    rules_matched: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'level': self.level,
            'recommendations': [r.to_dict() for r in self.recommendations],
            'generated_at': self.generated_at,
            'total_rules_evaluated': self.total_rules_evaluated,
            'rules_matched': self.rules_matched,
            'metadata': self.metadata
        }


# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class RecommendationEngine:
    """
    Core recommendation engine that evaluates rules and generates recommendations
    """
    
    def __init__(
        self,
        rules_path: str = "config/recommendation_rules.json",
        schema_path: str = "rules/recommendation_rules.schema.json"
    ):
        """
        Initialize recommendation engine
        
        Args:
            rules_path: Path to recommendation rules JSON file
            schema_path: Path to JSON schema for validation
        """
        self.rules_path = Path(rules_path)
        self.schema_path = Path(schema_path)
        self.rules: Dict[str, Any] = {}
        self.schema: Dict[str, Any] = {}
        self.rules_by_level: Dict[str, List[Dict[str, Any]]] = {
            'MICRO': [],
            'MESO': [],
            'MACRO': []
        }
        
        # Load rules and schema
        self._load_schema()
        self._load_rules()
        
        logger.info(
            f"Recommendation engine initialized with "
            f"{len(self.rules_by_level['MICRO'])} MICRO, "
            f"{len(self.rules_by_level['MESO'])} MESO, "
            f"{len(self.rules_by_level['MACRO'])} MACRO rules"
        )
    
    def _load_schema(self):
        """Load JSON schema for rule validation"""
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
            logger.info(f"Loaded recommendation rules schema from {self.schema_path}")
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
    
    def _load_rules(self):
        """Load and validate recommendation rules"""
        try:
            with open(self.rules_path, 'r', encoding='utf-8') as f:
                self.rules = json.load(f)
            
            # Validate against schema
            jsonschema.validate(instance=self.rules, schema=self.schema)
            
            # Organize rules by level
            for rule in self.rules.get('rules', []):
                level = rule.get('level')
                if level in self.rules_by_level:
                    self.rules_by_level[level].append(rule)
            
            logger.info(f"Loaded and validated {len(self.rules.get('rules', []))} rules from {self.rules_path}")
        except jsonschema.ValidationError as e:
            logger.error(f"Rule validation failed: {e.message}")
            raise
        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            raise
    
    def reload_rules(self):
        """Reload rules from disk (useful for hot-reloading)"""
        self.rules_by_level = {'MICRO': [], 'MESO': [], 'MACRO': []}
        self._load_rules()
    
    # ========================================================================
    # MICRO LEVEL RECOMMENDATIONS
    # ========================================================================
    
    def generate_micro_recommendations(
        self,
        scores: Dict[str, float],
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationSet:
        """
        Generate MICRO-level recommendations based on PA-DIM scores
        
        Args:
            scores: Dictionary mapping "PA##-DIM##" to scores (0.0-3.0)
            context: Additional context for template rendering
            
        Returns:
            RecommendationSet with matched recommendations
        """
        recommendations = []
        rules_evaluated = 0
        
        for rule in self.rules_by_level['MICRO']:
            rules_evaluated += 1
            
            # Extract condition
            when = rule.get('when', {})
            pa_id = when.get('pa_id')
            dim_id = when.get('dim_id')
            score_lt = when.get('score_lt')
            
            # Build score key
            score_key = f"{pa_id}-{dim_id}"
            
            # Check if condition matches
            if score_key in scores and scores[score_key] < score_lt:
                # Render template
                template = rule.get('template', {})
                rendered = self._render_micro_template(template, pa_id, dim_id, context)
                
                # Create recommendation
                rec = Recommendation(
                    rule_id=rule.get('rule_id'),
                    level='MICRO',
                    problem=rendered['problem'],
                    intervention=rendered['intervention'],
                    indicator=rendered['indicator'],
                    responsible=rendered['responsible'],
                    horizon=rendered['horizon'],
                    verification=rendered['verification'],
                    metadata={
                        'score_key': score_key,
                        'actual_score': scores[score_key],
                        'threshold': score_lt,
                        'gap': score_lt - scores[score_key]
                    }
                )
                recommendations.append(rec)
        
        return RecommendationSet(
            level='MICRO',
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_rules_evaluated=rules_evaluated,
            rules_matched=len(recommendations)
        )
    
    def _render_micro_template(
        self,
        template: Dict[str, Any],
        pa_id: str,
        dim_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Render MICRO template with variable substitution
        
        Variables supported:
        - {{PAxx}}: Policy area (e.g., PA01)
        - {{DIMxx}}: Dimension (e.g., DIM01)
        - {{Q###}}: Question number (from context)
        """
        ctx = context or {}
        
        # Build substitution map
        substitutions = {
            'PAxx': pa_id,
            'DIMxx': dim_id,
            'Q001': ctx.get('question_id', 'Q001'),  # Default or from context
        }
        
        rendered = {}
        for key, value in template.items():
            if isinstance(value, str):
                rendered[key] = self._substitute_variables(value, substitutions)
            else:
                rendered[key] = value
        
        return rendered
    
    # ========================================================================
    # MESO LEVEL RECOMMENDATIONS
    # ========================================================================
    
    def generate_meso_recommendations(
        self,
        cluster_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationSet:
        """
        Generate MESO-level recommendations based on cluster performance
        
        Args:
            cluster_data: Dictionary with cluster metrics:
                {
                    'CL01': {'score': 75.0, 'variance': 0.15, 'weak_pa': 'PA02'},
                    'CL02': {'score': 62.0, 'variance': 0.22, 'weak_pa': 'PA05'},
                    ...
                }
            context: Additional context for template rendering
            
        Returns:
            RecommendationSet with matched recommendations
        """
        recommendations = []
        rules_evaluated = 0
        
        for rule in self.rules_by_level['MESO']:
            rules_evaluated += 1
            
            # Extract condition
            when = rule.get('when', {})
            cluster_id = when.get('cluster_id')
            score_band = when.get('score_band')
            variance_level = when.get('variance_level')
            variance_threshold = when.get('variance_threshold')
            weak_pa_id = when.get('weak_pa_id')
            
            # Get cluster data
            cluster = cluster_data.get(cluster_id, {})
            cluster_score = cluster.get('score', 0)
            cluster_variance = cluster.get('variance', 0)
            cluster_weak_pa = cluster.get('weak_pa')
            
            # Check conditions
            if not self._check_meso_conditions(
                cluster_score, cluster_variance, cluster_weak_pa,
                score_band, variance_level, variance_threshold, weak_pa_id
            ):
                continue
            
            # Render template
            template = rule.get('template', {})
            rendered = self._render_meso_template(template, cluster_id, context)
            
            # Create recommendation
            rec = Recommendation(
                rule_id=rule.get('rule_id'),
                level='MESO',
                problem=rendered['problem'],
                intervention=rendered['intervention'],
                indicator=rendered['indicator'],
                responsible=rendered['responsible'],
                horizon=rendered['horizon'],
                verification=rendered['verification'],
                metadata={
                    'cluster_id': cluster_id,
                    'score': cluster_score,
                    'score_band': score_band,
                    'variance': cluster_variance,
                    'variance_level': variance_level,
                    'weak_pa': cluster_weak_pa
                }
            )
            recommendations.append(rec)
        
        return RecommendationSet(
            level='MESO',
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_rules_evaluated=rules_evaluated,
            rules_matched=len(recommendations)
        )
    
    def _check_meso_conditions(
        self,
        score: float,
        variance: float,
        weak_pa: Optional[str],
        score_band: str,
        variance_level: str,
        variance_threshold: Optional[float],
        weak_pa_id: Optional[str]
    ) -> bool:
        """Check if MESO conditions are met"""
        # Check score band
        if score_band == 'BAJO' and score >= 55:
            return False
        elif score_band == 'MEDIO' and (score < 55 or score >= 75):
            return False
        elif score_band == 'ALTO' and score < 75:
            return False
        
        # Check variance level
        if variance_level == 'BAJA' and variance >= 0.08:
            return False
        elif variance_level == 'MEDIA' and (variance < 0.08 or variance >= 0.18):
            return False
        elif variance_level == 'ALTA':
            if variance_threshold and variance < variance_threshold / 100:
                return False
            elif not variance_threshold and variance < 0.18:
                return False
        
        # Check weak PA if specified
        if weak_pa_id and weak_pa != weak_pa_id:
            return False
        
        return True
    
    def _render_meso_template(
        self,
        template: Dict[str, Any],
        cluster_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render MESO template with variable substitution"""
        ctx = context or {}
        
        substitutions = {
            'cluster_id': cluster_id,
        }
        
        rendered = {}
        for key, value in template.items():
            if isinstance(value, str):
                rendered[key] = self._substitute_variables(value, substitutions)
            else:
                rendered[key] = value
        
        return rendered
    
    # ========================================================================
    # MACRO LEVEL RECOMMENDATIONS
    # ========================================================================
    
    def generate_macro_recommendations(
        self,
        macro_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> RecommendationSet:
        """
        Generate MACRO-level strategic recommendations
        
        Args:
            macro_data: Dictionary with plan-level metrics:
                {
                    'macro_band': 'SATISFACTORIO',
                    'clusters_below_target': ['CL02', 'CL03'],
                    'variance_alert': 'MODERADA',
                    'priority_micro_gaps': ['PA01-DIM05', 'PA04-DIM04']
                }
            context: Additional context for template rendering
            
        Returns:
            RecommendationSet with matched recommendations
        """
        recommendations = []
        rules_evaluated = 0
        
        for rule in self.rules_by_level['MACRO']:
            rules_evaluated += 1
            
            # Extract condition
            when = rule.get('when', {})
            macro_band = when.get('macro_band')
            clusters_below = set(when.get('clusters_below_target', []))
            variance_alert = when.get('variance_alert')
            priority_gaps = set(when.get('priority_micro_gaps', []))
            
            # Get macro data
            actual_band = macro_data.get('macro_band')
            actual_clusters = set(macro_data.get('clusters_below_target', []))
            actual_variance = macro_data.get('variance_alert')
            actual_gaps = set(macro_data.get('priority_micro_gaps', []))
            
            # Check conditions
            if macro_band and macro_band != actual_band:
                continue
            if variance_alert and variance_alert != actual_variance:
                continue
            
            # Check if clusters match (subset or exact match)
            if clusters_below and not clusters_below.issubset(actual_clusters):
                # For MACRO, we want exact match or the rule's clusters to be present
                if clusters_below != actual_clusters and not actual_clusters.issubset(clusters_below):
                    continue
            
            # Check if priority gaps match (subset)
            if priority_gaps and not priority_gaps.issubset(actual_gaps):
                continue
            
            # Render template
            template = rule.get('template', {})
            rendered = self._render_macro_template(template, context)
            
            # Create recommendation
            rec = Recommendation(
                rule_id=rule.get('rule_id'),
                level='MACRO',
                problem=rendered['problem'],
                intervention=rendered['intervention'],
                indicator=rendered['indicator'],
                responsible=rendered['responsible'],
                horizon=rendered['horizon'],
                verification=rendered['verification'],
                metadata={
                    'macro_band': actual_band,
                    'clusters_below_target': list(actual_clusters),
                    'variance_alert': actual_variance,
                    'priority_micro_gaps': list(actual_gaps)
                }
            )
            recommendations.append(rec)
        
        return RecommendationSet(
            level='MACRO',
            recommendations=recommendations,
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_rules_evaluated=rules_evaluated,
            rules_matched=len(recommendations)
        )
    
    def _render_macro_template(
        self,
        template: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render MACRO template with variable substitution"""
        ctx = context or {}
        
        substitutions = {}
        
        rendered = {}
        for key, value in template.items():
            if isinstance(value, str):
                rendered[key] = self._substitute_variables(value, substitutions)
            else:
                rendered[key] = value
        
        return rendered
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _substitute_variables(self, text: str, substitutions: Dict[str, str]) -> str:
        """
        Substitute variables in text using {{variable}} syntax
        
        Args:
            text: Text with variables
            substitutions: Dictionary of variable_name -> value
            
        Returns:
            Text with variables substituted
        """
        result = text
        for var, value in substitutions.items():
            pattern = r'\{\{' + re.escape(var) + r'\}\}'
            result = re.sub(pattern, value, result)
        return result
    
    def generate_all_recommendations(
        self,
        micro_scores: Dict[str, float],
        cluster_data: Dict[str, Any],
        macro_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, RecommendationSet]:
        """
        Generate recommendations at all three levels
        
        Args:
            micro_scores: PA-DIM scores for MICRO recommendations
            cluster_data: Cluster metrics for MESO recommendations
            macro_data: Plan-level metrics for MACRO recommendations
            context: Additional context
            
        Returns:
            Dictionary with 'MICRO', 'MESO', and 'MACRO' recommendation sets
        """
        return {
            'MICRO': self.generate_micro_recommendations(micro_scores, context),
            'MESO': self.generate_meso_recommendations(cluster_data, context),
            'MACRO': self.generate_macro_recommendations(macro_data, context)
        }
    
    def export_recommendations(
        self,
        recommendations: Dict[str, RecommendationSet],
        output_path: str,
        format: str = 'json'
    ):
        """
        Export recommendations to file
        
        Args:
            recommendations: Dictionary of recommendation sets
            output_path: Path to output file
            format: Output format ('json' or 'markdown')
        """
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {level: rec_set.to_dict() for level, rec_set in recommendations.items()},
                    f,
                    indent=2,
                    ensure_ascii=False
                )
        elif format == 'markdown':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self._format_as_markdown(recommendations))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported recommendations to {output_path} in {format} format")
    
    def _format_as_markdown(self, recommendations: Dict[str, RecommendationSet]) -> str:
        """Format recommendations as Markdown"""
        lines = ["# Recomendaciones del Plan de Desarrollo\n"]
        
        for level in ['MICRO', 'MESO', 'MACRO']:
            rec_set = recommendations.get(level)
            if not rec_set:
                continue
            
            lines.append(f"\n## Nivel {level}\n")
            lines.append(f"**Generado:** {rec_set.generated_at}\n")
            lines.append(f"**Reglas evaluadas:** {rec_set.total_rules_evaluated}\n")
            lines.append(f"**Recomendaciones:** {rec_set.rules_matched}\n")
            
            for i, rec in enumerate(rec_set.recommendations, 1):
                lines.append(f"\n### {i}. {rec.rule_id}\n")
                lines.append(f"**Problema:** {rec.problem}\n")
                lines.append(f"\n**Intervención:** {rec.intervention}\n")
                lines.append(f"\n**Indicador:**")
                lines.append(f"- Nombre: {rec.indicator.get('name')}")
                lines.append(f"- Meta: {rec.indicator.get('target')} {rec.indicator.get('unit')}\n")
                lines.append(f"\n**Responsable:** {rec.responsible.get('entity')} ({rec.responsible.get('role')})\n")
                lines.append(f"**Socios:** {', '.join(rec.responsible.get('partners', []))}\n")
                lines.append(f"\n**Horizonte:** {rec.horizon.get('start')} → {rec.horizon.get('end')}\n")
                lines.append(f"\n**Verificación:**")
                for v in rec.verification:
                    lines.append(f"- {v}")
                lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_recommendation_engine(
    rules_path: str = "config/recommendation_rules.json",
    schema_path: str = "rules/recommendation_rules.schema.json"
) -> RecommendationEngine:
    """
    Convenience function to load recommendation engine
    
    Args:
        rules_path: Path to rules JSON
        schema_path: Path to schema JSON
        
    Returns:
        Initialized RecommendationEngine
    """
    return RecommendationEngine(rules_path=rules_path, schema_path=schema_path)


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    engine = load_recommendation_engine()
    
    # Example MICRO recommendations
    micro_scores = {
        'PA01-DIM01': 1.2,  # Below threshold of 1.65
        'PA02-DIM03': 1.8,  # Above threshold
        'PA03-DIM05': 1.4,  # Below threshold
    }
    
    micro_recs = engine.generate_micro_recommendations(micro_scores)
    print(f"\n=== MICRO Recommendations ===")
    print(f"Rules evaluated: {micro_recs.total_rules_evaluated}")
    print(f"Recommendations: {micro_recs.rules_matched}")
    
    for rec in micro_recs.recommendations[:2]:  # Show first 2
        print(f"\n{rec.rule_id}:")
        print(f"Problem: {rec.problem[:100]}...")
        print(f"Intervention: {rec.intervention[:100]}...")
    
    # Example MESO recommendations
    cluster_data = {
        'CL01': {'score': 72.0, 'variance': 0.25, 'weak_pa': 'PA02'},
        'CL02': {'score': 58.0, 'variance': 0.12},
    }
    
    meso_recs = engine.generate_meso_recommendations(cluster_data)
    print(f"\n=== MESO Recommendations ===")
    print(f"Rules evaluated: {meso_recs.total_rules_evaluated}")
    print(f"Recommendations: {meso_recs.rules_matched}")
    
    # Example MACRO recommendations
    macro_data = {
        'macro_band': 'SATISFACTORIO',
        'clusters_below_target': ['CL02', 'CL03'],
        'variance_alert': 'MODERADA',
        'priority_micro_gaps': ['PA01-DIM05', 'PA05-DIM04', 'PA04-DIM04', 'PA08-DIM05']
    }
    
    macro_recs = engine.generate_macro_recommendations(macro_data)
    print(f"\n=== MACRO Recommendations ===")
    print(f"Rules evaluated: {macro_recs.total_rules_evaluated}")
    print(f"Recommendations: {macro_recs.rules_matched}")
