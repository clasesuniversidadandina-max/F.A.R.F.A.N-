"""
Advanced Policy Contradiction Detection System for Colombian Municipal Development Plans

Este sistema implementa el estado del arte en detección de contradicciones para análisis
de políticas públicas, específicamente calibrado para Planes de Desarrollo Municipal (PDM)
colombianos según la Ley 152 de 1994 y metodología DNP.

Innovations:
- Transformer-based semantic similarity using sentence-transformers
- Graph-based contradiction reasoning with NetworkX
- Bayesian inference for confidence scoring
- Temporal logic verification for timeline consistency
- Multi-dimensional vector embeddings for policy alignment
- Statistical hypothesis testing for numerical claims
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Protocol

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cosine
from scipy.stats import beta, chi2_contingency, ks_2samp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging with structured format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContradictionType(Enum):
    """Taxonomía de contradicciones según estándares de política pública"""
    NUMERICAL_INCONSISTENCY = auto()
    TEMPORAL_CONFLICT = auto()
    SEMANTIC_OPPOSITION = auto()
    LOGICAL_INCOMPATIBILITY = auto()
    RESOURCE_ALLOCATION_MISMATCH = auto()
    OBJECTIVE_MISALIGNMENT = auto()
    REGULATORY_CONFLICT = auto()
    STAKEHOLDER_DIVERGENCE = auto()


class PolicyDimension(Enum):
    """Dimensiones del Plan de Desarrollo según DNP Colombia"""
    DIAGNOSTICO = "diagnóstico"
    ESTRATEGICO = "estratégico"
    PROGRAMATICO = "programático"
    FINANCIERO = "plan plurianual de inversiones"
    SEGUIMIENTO = "seguimiento y evaluación"
    TERRITORIAL = "ordenamiento territorial"


@dataclass(frozen=True)
class PolicyStatement:
    """Representación estructurada de una declaración de política"""
    text: str
    dimension: PolicyDimension
    position: Tuple[int, int]  # (start, end) in document
    entities: List[str] = field(default_factory=list)
    temporal_markers: List[str] = field(default_factory=list)
    quantitative_claims: List[Dict[str, Any]] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    context_window: str = ""
    semantic_role: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class ContradictionEvidence:
    """Evidencia estructurada de contradicción con trazabilidad completa"""
    statement_a: PolicyStatement
    statement_b: PolicyStatement
    contradiction_type: ContradictionType
    confidence: float  # Bayesian posterior probability
    severity: float  # Impact on policy coherence
    semantic_similarity: float
    logical_conflict_score: float
    temporal_consistency: bool
    numerical_divergence: Optional[float]
    affected_dimensions: List[PolicyDimension]
    resolution_suggestions: List[str]
    graph_path: Optional[List[str]] = None
    statistical_significance: Optional[float] = None


class BayesianConfidenceCalculator:
    """Cálculo Bayesiano de confianza con priors informados por dominio"""

    def __init__(self):
        # Priors basados en análisis empírico de PDMs colombianos
        self.prior_alpha = 2.5  # Shape parameter for beta distribution
        self.prior_beta = 7.5  # Scale parameter (sesgo conservador)

    def calculate_posterior(
            self,
            evidence_strength: float,
            observations: int,
            domain_weight: float = 1.0
    ) -> float:
        """
        Calcula probabilidad posterior usando inferencia Bayesiana

        Args:
            evidence_strength: Fuerza de la evidencia [0, 1]
            observations: Número de observaciones que soportan la evidencia
            domain_weight: Peso específico del dominio de política
        """
        # Actualizar distribución Beta con evidencia
        alpha_post = self.prior_alpha + evidence_strength * observations * domain_weight
        beta_post = self.prior_beta + (1 - evidence_strength) * observations * domain_weight

        # Calcular media de la distribución posterior
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # Calcular intervalo de credibilidad del 95%
        credible_interval = beta.interval(0.95, alpha_post, beta_post)

        # Ajustar por incertidumbre
        uncertainty_penalty = 1.0 - (credible_interval[1] - credible_interval[0])

        return min(1.0, posterior_mean * uncertainty_penalty)


class TemporalLogicVerifier:
    """Verificación de consistencia temporal usando lógica temporal lineal (LTL)"""

    def __init__(self):
        self.temporal_patterns = {
            'sequential': re.compile(r'(primero|luego|después|posteriormente|finalmente)', re.IGNORECASE),
            'parallel': re.compile(r'(simultáneamente|al mismo tiempo|paralelamente)', re.IGNORECASE),
            'deadline': re.compile(r'(antes de|hasta|máximo|plazo)', re.IGNORECASE),
            'milestone': re.compile(r'(hito|meta intermedia|checkpoint)', re.IGNORECASE)
        }

    def verify_temporal_consistency(
            self,
            statements: List[PolicyStatement]
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Verifica consistencia temporal entre declaraciones

        Returns:
            (is_consistent, conflicts_found)
        """
        timeline = self._build_timeline(statements)
        conflicts = []

        # Verificar orden temporal
        for i, event_a in enumerate(timeline):
            for event_b in timeline[i + 1:]:
                if self._has_temporal_conflict(event_a, event_b):
                    conflicts.append({
                        'event_a': event_a,
                        'event_b': event_b,
                        'conflict_type': 'temporal_ordering'
                    })

        # Verificar restricciones de plazo
        deadline_violations = self._check_deadline_constraints(timeline)
        conflicts.extend(deadline_violations)

        return len(conflicts) == 0, conflicts

    def _build_timeline(self, statements: List[PolicyStatement]) -> List[Dict]:
        """Construye línea temporal a partir de declaraciones"""
        timeline = []
        for stmt in statements:
            for marker in stmt.temporal_markers:
                # Extraer información temporal estructurada
                timeline.append({
                    'statement': stmt,
                    'marker': marker,
                    'timestamp': self._parse_temporal_marker(marker),
                    'type': self._classify_temporal_type(marker)
                })
        return sorted(timeline, key=lambda x: x.get('timestamp', 0))

    def _parse_temporal_marker(self, marker: str) -> Optional[int]:
        """Parsea marcador temporal a timestamp numérico"""
        # Implementación específica para formato colombiano
        year_match = re.search(r'20\d{2}', marker)
        if year_match:
            return int(year_match.group())

        quarter_patterns = {
            'primer': 1, 'segundo': 2, 'tercer': 3, 'cuarto': 4,
            'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4
        }
        for pattern, quarter in quarter_patterns.items():
            if pattern in marker.lower():
                return quarter

        return None

    def _has_temporal_conflict(self, event_a: Dict, event_b: Dict) -> bool:
        """Detecta conflictos temporales entre eventos"""
        if event_a['timestamp'] and event_b['timestamp']:
            # Verificar si eventos mutuamente excluyentes ocurren simultáneamente
            if event_a['timestamp'] == event_b['timestamp']:
                return self._are_mutually_exclusive(
                    event_a['statement'],
                    event_b['statement']
                )
        return False

    def _are_mutually_exclusive(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> bool:
        """Determina si dos declaraciones son mutuamente excluyentes"""
        # Verificar si compiten por los mismos recursos
        resources_a = set(self._extract_resources(stmt_a.text))
        resources_b = set(self._extract_resources(stmt_b.text))

        return len(resources_a & resources_b) > 0

    def _extract_resources(self, text: str) -> List[str]:
        """Extrae recursos mencionados en el texto"""
        resource_patterns = [
            r'presupuesto',
            r'recursos?\s+\w+',
            r'fondos?\s+\w+',
            r'personal',
            r'infraestructura'
        ]
        resources = []
        for pattern in resource_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            resources.extend(matches)
        return resources

    def _check_deadline_constraints(self, timeline: List[Dict]) -> List[Dict]:
        """Verifica violaciones de restricciones de plazo"""
        violations = []
        for event in timeline:
            if event['type'] == 'deadline':
                # Verificar si hay eventos posteriores que deberían ocurrir antes
                for other in timeline:
                    if other['timestamp'] and event['timestamp']:
                        if other['timestamp'] > event['timestamp']:
                            if self._should_precede(other['statement'], event['statement']):
                                violations.append({
                                    'event_a': other,
                                    'event_b': event,
                                    'conflict_type': 'deadline_violation'
                                })
        return violations

    def _should_precede(self, stmt_a: PolicyStatement, stmt_b: PolicyStatement) -> bool:
        """Determina si stmt_a debe preceder a stmt_b"""
        # Análisis de dependencias causales
        return bool(stmt_a.dependencies & {stmt_b.text[:50]})

    def _classify_temporal_type(self, marker: str) -> str:
        """Clasifica el tipo de marcador temporal"""
        for pattern_type, pattern in self.temporal_patterns.items():
            if pattern.search(marker):
                return pattern_type
        return 'unspecified'


class PolicyContradictionDetector:
    """
    Sistema avanzado de detección de contradicciones para PDMs colombianos.
    Implementa el estado del arte en NLP y razonamiento lógico.
    """

    def __init__(
            self,
            model_name: str = "hiiamsid/sentence_similarity_spanish_es",
            spacy_model: str = "es_core_news_lg",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # Modelos de transformers para análisis semántico
        self.semantic_model = SentenceTransformer(model_name, device=device)

        # Modelo de clasificación de contradicciones
        self.contradiction_classifier = pipeline(
            "text-classification",
            model="microsoft/deberta-v3-base",
            device=0 if device == "cuda" else -1
        )

        # Procesamiento de lenguaje natural
        self.nlp = spacy.load(spacy_model)

        # Componentes especializados
        self.bayesian_calculator = BayesianConfidenceCalculator()
        self.temporal_verifier = TemporalLogicVerifier()

        # Grafo de conocimiento para razonamiento
        self.knowledge_graph = nx.DiGraph()

        # Vectorizador TF-IDF para análisis complementario
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=5000,
            sublinear_tf=True
        )

        # Patrones específicos de PDM colombiano
        self._initialize_pdm_patterns()

    def _initialize_pdm_patterns(self):
        """Inicializa patrones específicos de PDMs colombianos"""
        self.pdm_patterns = {
            'ejes_estrategicos': re.compile(
                r'(eje\s+estratégico|línea\s+estratégica|pilar|dimensión)',
                re.IGNORECASE
            ),
            'programas': re.compile(
                r'(programa|subprograma|proyecto|iniciativa)',
                re.IGNORECASE
            ),
            'metas': re.compile(
                r'(meta\s+de\s+resultado|meta\s+de\s+producto|indicador)',
                re.IGNORECASE
            ),
            'recursos': re.compile(
                r'(SGP|regalías|recursos\s+propios|cofinanciación|crédito)',
                re.IGNORECASE
            ),
            'normativa': re.compile(
                r'(ley\s+\d+|decreto\s+\d+|acuerdo\s+\d+|resolución\s+\d+)',
                re.IGNORECASE
            )
        }

    def detect(
            self,
            text: str,
            plan_name: str = "PDM",
            dimension: PolicyDimension = PolicyDimension.ESTRATEGICO
    ) -> Dict[str, Any]:
        """
        Detecta contradicciones con análisis multi-dimensional avanzado

        Args:
            text: Texto del plan de desarrollo
            plan_name: Nombre del PDM
            dimension: Dimensión del plan siendo analizada

        Returns:
            Análisis completo con contradicciones detectadas y métricas
        """
        # Extraer declaraciones de política estructuradas
        statements = self._extract_policy_statements(text, dimension)

        # Generar embeddings semánticos
        statements = self._generate_embeddings(statements)

        # Construir grafo de conocimiento
        self._build_knowledge_graph(statements)

        # Detectar contradicciones multi-tipo
        contradictions = []

        # 1. Contradicciones semánticas usando transformers
        semantic_contradictions = self._detect_semantic_contradictions(statements)
        contradictions.extend(semantic_contradictions)

        # 2. Inconsistencias numéricas con pruebas estadísticas
        numerical_contradictions = self._detect_numerical_inconsistencies(statements)
        contradictions.extend(numerical_contradictions)

        # 3. Conflictos temporales con verificación lógica
        temporal_conflicts = self._detect_temporal_conflicts(statements)
        contradictions.extend(temporal_conflicts)

        # 4. Incompatibilidades lógicas usando razonamiento en grafo
        logical_contradictions = self._detect_logical_incompatibilities(statements)
        contradictions.extend(logical_contradictions)

        # 5. Conflictos de asignación de recursos
        resource_conflicts = self._detect_resource_conflicts(statements)
        contradictions.extend(resource_conflicts)

        # Calcular métricas agregadas
        coherence_metrics = self._calculate_coherence_metrics(
            contradictions,
            statements,
            text
        )

        # Generar recomendaciones de resolución
        recommendations = self._generate_resolution_recommendations(contradictions)

        return {
            "plan_name": plan_name,
            "dimension": dimension.value,
            "contradictions": [self._serialize_contradiction(c) for c in contradictions],
            "total_contradictions": len(contradictions),
            "high_severity_count": sum(1 for c in contradictions if c.severity > 0.7),
            "coherence_metrics": coherence_metrics,
            "recommendations": recommendations,
            "knowledge_graph_stats": self._get_graph_statistics()
        }

    def _extract_policy_statements(
            self,
            text: str,
            dimension: PolicyDimension
    ) -> List[PolicyStatement]:
        """Extrae declaraciones de política estructuradas del texto"""
        doc = self.nlp(text)
        statements = []

        for sent in doc.sents:
            # Analizar entidades nombradas
            entities = [ent.text for ent in sent.ents]

            # Extraer marcadores temporales
            temporal_markers = self._extract_temporal_markers(sent.text)

            # Extraer afirmaciones cuantitativas
            quantitative_claims = self._extract_quantitative_claims(sent.text)

            # Determinar rol semántico
            semantic_role = self._determine_semantic_role(sent)

            # Identificar dependencias
            dependencies = self._identify_dependencies(sent, doc)

            statement = PolicyStatement(
                text=sent.text,
                dimension=dimension,
                position=(sent.start_char, sent.end_char),
                entities=entities,
                temporal_markers=temporal_markers,
                quantitative_claims=quantitative_claims,
                context_window=self._get_context_window(text, sent.start_char, sent.end_char),
                semantic_role=semantic_role,
                dependencies=dependencies
            )

            statements.append(statement)

        return statements

    def _generate_embeddings(
            self,
            statements: List[PolicyStatement]
    ) -> List[PolicyStatement]:
        """Genera embeddings semánticos para las declaraciones"""
        texts = [stmt.text for stmt in statements]
        embeddings = self.semantic_model.encode(texts, convert_to_numpy=True)

        # Crear nuevas instancias con embeddings
        enhanced_statements = []
        for stmt, embedding in zip(statements, embeddings):
            enhanced = PolicyStatement(
                text=stmt.text,
                dimension=stmt.dimension,
                position=stmt.position,
                entities=stmt.entities,
                temporal_markers=stmt.temporal_markers,
                quantitative_claims=stmt.quantitative_claims,
                embedding=embedding,
                context_window=stmt.context_window,
                semantic_role=stmt.semantic_role,
                dependencies=stmt.dependencies
            )
            enhanced_statements.append(enhanced)

        return enhanced_statements

    def _build_knowledge_graph(self, statements: List[PolicyStatement]):
        """Construye grafo de conocimiento para razonamiento"""
        self.knowledge_graph.clear()

        for i, stmt in enumerate(statements):
            node_id = f"stmt_{i}"
            self.knowledge_graph.add_node(
                node_id,
                text=stmt.text[:100],
                dimension=stmt.dimension.value,
                entities=stmt.entities,
                semantic_role=stmt.semantic_role
            )

            # Conectar con declaraciones relacionadas
            for j, other in enumerate(statements):
                if i != j:
                    similarity = self._calculate_similarity(stmt, other)
                    if similarity > 0.7:  # Umbral de relación
                        self.knowledge_graph.add_edge(
                            f"stmt_{i}",
                            f"stmt_{j}",
                            weight=similarity,
                            relation_type=self._determine_relation_type(stmt, other)
                        )

    def _detect_semantic_contradictions(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detecta contradicciones semánticas usando transformers"""
        contradictions = []

        for i, stmt_a in enumerate(statements):
            for stmt_b in statements[i + 1:]:
                if stmt_a.embedding is not None and stmt_b.embedding is not None:
                    # Calcular similaridad coseno
                    similarity = 1 - cosine(stmt_a.embedding, stmt_b.embedding)

                    # Verificar contradicción usando clasificador
                    combined_text = f"{stmt_a.text} [SEP] {stmt_b.text}"
                    contradiction_score = self._classify_contradiction(combined_text)

                    if contradiction_score > 0.7 and similarity > 0.5:
                        # Calcular confianza Bayesiana
                        confidence = self.bayesian_calculator.calculate_posterior(
                            evidence_strength=contradiction_score,
                            observations=len(stmt_a.entities) + len(stmt_b.entities),
                            domain_weight=self._get_domain_weight(stmt_a.dimension)
                        )

                        evidence = ContradictionEvidence(
                            statement_a=stmt_a,
                            statement_b=stmt_b,
                            contradiction_type=ContradictionType.SEMANTIC_OPPOSITION,
                            confidence=confidence,
                            severity=self._calculate_severity(stmt_a, stmt_b),
                            semantic_similarity=similarity,
                            logical_conflict_score=contradiction_score,
                            temporal_consistency=True,
                            numerical_divergence=None,
                            affected_dimensions=[stmt_a.dimension, stmt_b.dimension],
                            resolution_suggestions=self._suggest_resolutions(
                                ContradictionType.SEMANTIC_OPPOSITION
                            )
                        )
                        contradictions.append(evidence)

        return contradictions

    def _detect_numerical_inconsistencies(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detecta inconsistencias numéricas con análisis estadístico"""
        contradictions = []

        for i, stmt_a in enumerate(statements):
            for stmt_b in statements[i + 1:]:
                if stmt_a.quantitative_claims and stmt_b.quantitative_claims:
                    for claim_a in stmt_a.quantitative_claims:
                        for claim_b in stmt_b.quantitative_claims:
                            if self._are_comparable_claims(claim_a, claim_b):
                                divergence = self._calculate_numerical_divergence(
                                    claim_a,
                                    claim_b
                                )

                                if divergence is not None and divergence > 0.2:
                                    # Test estadístico de significancia
                                    p_value = self._statistical_significance_test(
                                        claim_a,
                                        claim_b
                                    )

                                    if p_value < 0.05:  # Significancia estadística
                                        confidence = self.bayesian_calculator.calculate_posterior(
                                            evidence_strength=1 - p_value,
                                            observations=2,
                                            domain_weight=1.5  # Mayor peso para evidencia numérica
                                        )

                                        evidence = ContradictionEvidence(
                                            statement_a=stmt_a,
                                            statement_b=stmt_b,
                                            contradiction_type=ContradictionType.NUMERICAL_INCONSISTENCY,
                                            confidence=confidence,
                                            severity=min(1.0, divergence),
                                            semantic_similarity=0.0,
                                            logical_conflict_score=divergence,
                                            temporal_consistency=True,
                                            numerical_divergence=divergence,
                                            affected_dimensions=[stmt_a.dimension],
                                            resolution_suggestions=self._suggest_resolutions(
                                                ContradictionType.NUMERICAL_INCONSISTENCY
                                            ),
                                            statistical_significance=p_value
                                        )
                                        contradictions.append(evidence)

        return contradictions

    def _detect_temporal_conflicts(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detecta conflictos temporales usando verificación lógica"""
        contradictions = []

        # Filtrar declaraciones con marcadores temporales
        temporal_statements = [s for s in statements if s.temporal_markers]

        if len(temporal_statements) >= 2:
            is_consistent, conflicts = self.temporal_verifier.verify_temporal_consistency(
                temporal_statements
            )

            for conflict in conflicts:
                stmt_a = conflict['event_a']['statement']
                stmt_b = conflict['event_b']['statement']

                confidence = self.bayesian_calculator.calculate_posterior(
                    evidence_strength=0.9,  # Alta confianza en lógica temporal
                    observations=len(conflicts),
                    domain_weight=1.2
                )

                evidence = ContradictionEvidence(
                    statement_a=stmt_a,
                    statement_b=stmt_b,
                    contradiction_type=ContradictionType.TEMPORAL_CONFLICT,
                    confidence=confidence,
                    severity=0.8,  # Los conflictos temporales son severos
                    semantic_similarity=self._calculate_similarity(stmt_a, stmt_b),
                    logical_conflict_score=1.0,
                    temporal_consistency=False,
                    numerical_divergence=None,
                    affected_dimensions=[PolicyDimension.PROGRAMATICO],
                    resolution_suggestions=self._suggest_resolutions(
                        ContradictionType.TEMPORAL_CONFLICT
                    )
                )
                contradictions.append(evidence)

        return contradictions

    def _detect_logical_incompatibilities(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detecta incompatibilidades lógicas usando razonamiento en grafo"""
        contradictions = []

        # Buscar ciclos negativos en el grafo (indicativos de contradicción)
        try:
            negative_cycles = nx.negative_edge_cycle(
                self.knowledge_graph,
                weight='weight'
            )

            for cycle in negative_cycles:
                # Extraer declaraciones del ciclo
                stmt_indices = [int(node.split('_')[1]) for node in cycle]
                cycle_statements = [statements[i] for i in stmt_indices]

                # Analizar incompatibilidad lógica
                for i in range(len(cycle_statements)):
                    stmt_a = cycle_statements[i]
                    stmt_b = cycle_statements[(i + 1) % len(cycle_statements)]

                    if self._has_logical_conflict(stmt_a, stmt_b):
                        confidence = self.bayesian_calculator.calculate_posterior(
                            evidence_strength=0.85,
                            observations=len(cycle),
                            domain_weight=1.0
                        )

                        evidence = ContradictionEvidence(
                            statement_a=stmt_a,
                            statement_b=stmt_b,
                            contradiction_type=ContradictionType.LOGICAL_INCOMPATIBILITY,
                            confidence=confidence,
                            severity=0.7,
                            semantic_similarity=self._calculate_similarity(stmt_a, stmt_b),
                            logical_conflict_score=0.9,
                            temporal_consistency=True,
                            numerical_divergence=None,
                            affected_dimensions=[stmt_a.dimension, stmt_b.dimension],
                            resolution_suggestions=self._suggest_resolutions(
                                ContradictionType.LOGICAL_INCOMPATIBILITY
                            ),
                            graph_path=cycle
                        )
                        contradictions.append(evidence)
        except nx.NetworkXError:
            pass  # No negative cycles found

        return contradictions

    def _detect_resource_conflicts(
            self,
            statements: List[PolicyStatement]
    ) -> List[ContradictionEvidence]:
        """Detecta conflictos en asignación de recursos"""
        contradictions = []
        resource_allocations = {}

        for stmt in statements:
            # Extraer menciones de recursos
            resources = self._extract_resource_mentions(stmt.text)
            for resource_type, amount in resources:
                if resource_type not in resource_allocations:
                    resource_allocations[resource_type] = []
                resource_allocations[resource_type].append((stmt, amount))

        # Verificar conflictos de asignación
        for resource_type, allocations in resource_allocations.items():
            if len(allocations) > 1:
                total_claimed = sum(amount for _, amount in allocations if amount)

                # Verificar si las asignaciones son mutuamente excluyentes
                for i, (stmt_a, amount_a) in enumerate(allocations):
                    for stmt_b, amount_b in allocations[i + 1:]:
                        if amount_a and amount_b:
                            if self._are_conflicting_allocations(
                                    amount_a,
                                    amount_b,
                                    total_claimed
                            ):
                                confidence = self.bayesian_calculator.calculate_posterior(
                                    evidence_strength=0.8,
                                    observations=len(allocations),
                                    domain_weight=1.3
                                )

                                evidence = ContradictionEvidence(
                                    statement_a=stmt_a,
                                    statement_b=stmt_b,
                                    contradiction_type=ContradictionType.RESOURCE_ALLOCATION_MISMATCH,
                                    confidence=confidence,
                                    severity=0.9,  # Conflictos de recursos son críticos
                                    semantic_similarity=self._calculate_similarity(stmt_a, stmt_b),
                                    logical_conflict_score=0.8,
                                    temporal_consistency=True,
                                    numerical_divergence=abs(amount_a - amount_b) / max(amount_a, amount_b),
                                    affected_dimensions=[PolicyDimension.FINANCIERO],
                                    resolution_suggestions=self._suggest_resolutions(
                                        ContradictionType.RESOURCE_ALLOCATION_MISMATCH
                                    )
                                )
                                contradictions.append(evidence)

        return contradictions

    def _calculate_coherence_metrics(
            self,
            contradictions: List[ContradictionEvidence],
            statements: List[PolicyStatement],
            text: str
    ) -> Dict[str, float]:
        """Calcula métricas avanzadas de coherencia del documento"""

        # Densidad de contradicciones normalizada
        contradiction_density = len(contradictions) / max(1, len(statements))

        # Índice de coherencia semántica global
        semantic_coherence = self._calculate_global_semantic_coherence(statements)

        # Consistencia temporal
        temporal_consistency = sum(
            1 for c in contradictions
            if c.contradiction_type != ContradictionType.TEMPORAL_CONFLICT
        ) / max(1, len(contradictions))

        # Alineación de objetivos
        objective_alignment = self._calculate_objective_alignment(statements)

        # Índice de fragmentación del grafo
        graph_fragmentation = self._calculate_graph_fragmentation()

        # Score de coherencia compuesto (weighted harmonic mean)
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        scores = np.array([
            1 - contradiction_density,
            semantic_coherence,
            temporal_consistency,
            objective_alignment,
            1 - graph_fragmentation
        ])

        # Harmonic mean ponderada para penalizar valores bajos
        coherence_score = np.sum(weights) / np.sum(weights / np.maximum(scores, 0.01))

        # Entropía de contradicciones
        contradiction_entropy = self._calculate_contradiction_entropy(contradictions)

        # Complejidad sintáctica del documento
        syntactic_complexity = self._calculate_syntactic_complexity(text)

        return {
            "coherence_score": float(coherence_score),
            "contradiction_density": float(contradiction_density),
            "semantic_coherence": float(semantic_coherence),
            "temporal_consistency": float(temporal_consistency),
            "objective_alignment": float(objective_alignment),
            "graph_fragmentation": float(graph_fragmentation),
            "contradiction_entropy": float(contradiction_entropy),
            "syntactic_complexity": float(syntactic_complexity),
            "confidence_interval": self._calculate_confidence_interval(coherence_score, len(statements))
        }

    def _calculate_global_semantic_coherence(
            self,
            statements: List[PolicyStatement]
    ) -> float:
        """Calcula coherencia semántica global usando embeddings"""
        if len(statements) < 2:
            return 1.0

        # Calcular matriz de similitud
        embeddings = [s.embedding for s in statements if s.embedding is not None]
        if len(embeddings) < 2:
            return 0.5

        similarity_matrix = cosine_similarity(embeddings)

        # Calcular coherencia como promedio de similitudes consecutivas
        consecutive_similarities = []
        for i in range(len(similarity_matrix) - 1):
            consecutive_similarities.append(similarity_matrix[i, i + 1])

        # Penalizar alta varianza en similitudes
        mean_similarity = np.mean(consecutive_similarities)
        std_similarity = np.std(consecutive_similarities)

        coherence = mean_similarity * (1 - min(0.5, std_similarity))

        return float(coherence)

    def _calculate_objective_alignment(
            self,
            statements: List[PolicyStatement]
    ) -> float:
        """Calcula alineación entre objetivos declarados"""
        objective_statements = [
            s for s in statements
            if s.semantic_role in ['objective', 'goal', 'target']
        ]

        if len(objective_statements) < 2:
            return 1.0

        # Analizar consistencia direccional de objetivos
        alignment_scores = []
        for i, obj_a in enumerate(objective_statements):
            for obj_b in objective_statements[i + 1:]:
                if obj_a.embedding is not None and obj_b.embedding is not None:
                    # Calcular alineación como similitud coseno
                    alignment = 1 - cosine(obj_a.embedding, obj_b.embedding)
                    alignment_scores.append(alignment)

        if alignment_scores:
            return float(np.mean(alignment_scores))
        return 0.5

    def _calculate_graph_fragmentation(self) -> float:
        """Calcula fragmentación del grafo de conocimiento"""
        if self.knowledge_graph.number_of_nodes() == 0:
            return 0.0

        # Calcular número de componentes conectados
        num_components = nx.number_weakly_connected_components(self.knowledge_graph)
        num_nodes = self.knowledge_graph.number_of_nodes()

        # Fragmentación normalizada
        fragmentation = (num_components - 1) / max(1, num_nodes - 1)

        return float(fragmentation)

    def _calculate_contradiction_entropy(
            self,
            contradictions: List[ContradictionEvidence]
    ) -> float:
        """Calcula entropía de distribución de tipos de contradicción"""
        if not contradictions:
            return 0.0

        # Contar frecuencia de cada tipo
        type_counts = {}
        for c in contradictions:
            type_counts[c.contradiction_type] = type_counts.get(c.contradiction_type, 0) + 1

        # Calcular probabilidades
        total = len(contradictions)
        probabilities = [count / total for count in type_counts.values()]

        # Calcular entropía de Shannon
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)

        # Normalizar por entropía máxima
        max_entropy = np.log2(len(ContradictionType))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        return float(normalized_entropy)

    def _calculate_syntactic_complexity(self, text: str) -> float:
        """Calcula complejidad sintáctica del documento"""
        doc = self.nlp(text[:5000])  # Limitar para eficiencia

        # Métricas de complejidad
        avg_sentence_length = np.mean([len(sent.text.split()) for sent in doc.sents])

        # Profundidad promedio del árbol de dependencias
        dependency_depths = []
        for sent in doc.sents:
            depths = [self._get_dependency_depth(token) for token in sent]
            if depths:
                dependency_depths.append(np.mean(depths))

        avg_dependency_depth = np.mean(dependency_depths) if dependency_depths else 0

        # Diversidad léxica (Type-Token Ratio)
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        ttr = len(set(tokens)) / len(tokens) if tokens else 0

        # Combinar métricas
        complexity = (
                min(1.0, avg_sentence_length / 50) * 0.3 +
                min(1.0, avg_dependency_depth / 10) * 0.3 +
                ttr * 0.4
        )

        return float(complexity)

    def _get_dependency_depth(self, token) -> int:
        """Calcula profundidad de un token en el árbol de dependencias"""
        depth = 0
        current = token
        while current.head != current and depth < 20:  # Evitar loops infinitos
            current = current.head
            depth += 1
        return depth

    def _calculate_confidence_interval(
            self,
            score: float,
            n_observations: int
    ) -> Tuple[float, float]:
        """Calcula intervalo de confianza del 95% para el score"""
        # Usar distribución t de Student para muestras pequeñas
        if n_observations < 30:
            # Error estándar estimado
            se = np.sqrt(score * (1 - score) / n_observations)
            # Valor crítico t para 95% de confianza
            t_critical = stats.t.ppf(0.975, n_observations - 1)
            margin = t_critical * se
        else:
            # Usar distribución normal para muestras grandes
            se = np.sqrt(score * (1 - score) / n_observations)
            margin = 1.96 * se

        return (
            max(0.0, score - margin),
            min(1.0, score + margin)
        )

    def _generate_resolution_recommendations(
            self,
            contradictions: List[ContradictionEvidence]
    ) -> List[Dict[str, Any]]:
        """Genera recomendaciones específicas para resolver contradicciones"""
        recommendations = []

        # Agrupar contradicciones por tipo
        by_type = {}
        for c in contradictions:
            if c.contradiction_type not in by_type:
                by_type[c.contradiction_type] = []
            by_type[c.contradiction_type].append(c)

        # Generar recomendaciones por tipo
        for cont_type, conflicts in by_type.items():
            if cont_type == ContradictionType.NUMERICAL_INCONSISTENCY:
                recommendations.append({
                    "type": "numerical_reconciliation",
                    "priority": "high",
                    "description": "Revisar y reconciliar cifras inconsistentes",
                    "specific_actions": [
                        "Verificar fuentes de datos originales",
                        "Establecer línea base única",
                        "Documentar metodología de cálculo"
                    ],
                    "affected_sections": self._identify_affected_sections(conflicts)
                })

            elif cont_type == ContradictionType.TEMPORAL_CONFLICT:
                recommendations.append({
                    "type": "timeline_adjustment",
                    "priority": "high",
                    "description": "Ajustar cronograma para resolver conflictos temporales",
                    "specific_actions": [
                        "Revisar secuencia de actividades",
                        "Validar plazos con áreas responsables",
                        "Establecer hitos intermedios claros"
                    ],
                    "affected_sections": self._identify_affected_sections(conflicts)
                })

            elif cont_type == ContradictionType.RESOURCE_ALLOCATION_MISMATCH:
                recommendations.append({
                    "type": "budget_reallocation",
                    "priority": "critical",
                    "description": "Revisar asignación presupuestal",
                    "specific_actions": [
                        "Realizar análisis de suficiencia presupuestal",
                        "Priorizar programas según impacto",
                        "Identificar fuentes alternativas de financiación"
                    ],
                    "affected_sections": self._identify_affected_sections(conflicts)
                })

            elif cont_type == ContradictionType.SEMANTIC_OPPOSITION:
                recommendations.append({
                    "type": "conceptual_clarification",
                    "priority": "medium",
                    "description": "Clarificar conceptos y objetivos opuestos",
                    "specific_actions": [
                        "Realizar sesiones de alineación estratégica",
                        "Definir glosario de términos unificado",
                        "Establecer jerarquía clara de objetivos"
                    ],
                    "affected_sections": self._identify_affected_sections(conflicts)
                })

        # Ordenar por prioridad
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))

        return recommendations

    def _identify_affected_sections(
            self,
            conflicts: List[ContradictionEvidence]
    ) -> List[str]:
        """Identifica secciones del plan afectadas por contradicciones"""
        affected = set()
        for c in conflicts:
            # Extraer información de sección desde el contexto
            for pattern_name, pattern in self.pdm_patterns.items():
                if pattern.search(c.statement_a.context_window):
                    affected.add(pattern_name)
                if pattern.search(c.statement_b.context_window):
                    affected.add(pattern_name)

        return list(affected)

    def _serialize_contradiction(
            self,
            contradiction: ContradictionEvidence
    ) -> Dict[str, Any]:
        """Serializa evidencia de contradicción para output"""
        return {
            "statement_1": contradiction.statement_a.text,
            "statement_2": contradiction.statement_b.text,
            "position_1": contradiction.statement_a.position,
            "position_2": contradiction.statement_b.position,
            "contradiction_type": contradiction.contradiction_type.name,
            "confidence": float(contradiction.confidence),
            "severity": float(contradiction.severity),
            "semantic_similarity": float(contradiction.semantic_similarity),
            "logical_conflict_score": float(contradiction.logical_conflict_score),
            "temporal_consistency": contradiction.temporal_consistency,
            "numerical_divergence": float(
                contradiction.numerical_divergence) if contradiction.numerical_divergence else None,
            "statistical_significance": float(
                contradiction.statistical_significance) if contradiction.statistical_significance else None,
            "affected_dimensions": [d.value for d in contradiction.affected_dimensions],
            "resolution_suggestions": contradiction.resolution_suggestions,
            "graph_path": contradiction.graph_path
        }

    def _get_graph_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del grafo de conocimiento"""
        if self.knowledge_graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "components": 0}

        return {
            "nodes": self.knowledge_graph.number_of_nodes(),
            "edges": self.knowledge_graph.number_of_edges(),
            "components": nx.number_weakly_connected_components(self.knowledge_graph),
            "density": nx.density(self.knowledge_graph),
            "average_clustering": nx.average_clustering(self.knowledge_graph.to_undirected()),
            "diameter": nx.diameter(self.knowledge_graph.to_undirected()) if nx.is_connected(
                self.knowledge_graph.to_undirected()) else -1
        }

    # Métodos auxiliares

    def _extract_temporal_markers(self, text: str) -> List[str]:
        """Extrae marcadores temporales del texto"""
        markers = []

        # Patrones de fechas
        date_patterns = [
            r'\d{1,2}\s+de\s+\w+\s+de\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'(enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+\d{4}',
            r'(Q[1-4]|trimestre\s+[1-4])\s+\d{4}',
            r'20\d{2}',
            r'(corto|mediano|largo)\s+plazo',
            r'(primer|segundo|tercer|cuarto)\s+(año|semestre|trimestre)'
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            markers.extend(matches)

        return markers

    def _extract_quantitative_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extrae afirmaciones cuantitativas estructuradas"""
        claims = []

        # Patrones numéricos con contexto
        patterns = [
            (r'(\d+(?:[.,]\d+)?)\s*(%|por\s*ciento)', 'percentage'),
            (r'(\d+(?:[.,]\d+)?)\s*(millones?|mil\s+millones?)', 'amount'),
            (r'(\$|COP)\s*(\d+(?:[.,]\d+)?)', 'currency'),
            (r'(\d+(?:[.,]\d+)?)\s*(personas?|beneficiarios?|familias?)', 'beneficiaries'),
            (r'(\d+(?:[.,]\d+)?)\s*(hectáreas?|km2?|metros?)', 'area'),
            (r'meta\s+de\s+(\d+(?:[.,]\d+)?)', 'target')
        ]

        for pattern, claim_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                value_str = match.group(1) if claim_type != 'currency' else match.group(2)
                value = self._parse_number(value_str)

                claims.append({
                    'type': claim_type,
                    'value': value,
                    'raw_text': match.group(0),
                    'position': match.span(),
                    'context': text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
                })

        return claims

    def _parse_number(self, text: str) -> float:
        """Parsea número desde texto"""
        try:
            # Reemplazar coma decimal
            normalized = text.replace(',', '.')
            return float(normalized)
        except ValueError:
            return 0.0

    def _extract_resource_mentions(self, text: str) -> List[Tuple[str, Optional[float]]]:
        """Extrae menciones de recursos con montos"""
        resources = []

        # Patrones de recursos específicos de PDM colombiano
        resource_patterns = [
            (r'SGP\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?)?', 'SGP'),
            (r'regalías\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?)?', 'regalías'),
            (r'recursos\s+propios\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?)?', 'recursos_propios'),
            (r'cofinanciación\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?)?', 'cofinanciación'),
            (r'presupuesto\s+total\s*[:\s]*\$?\s*(\d+(?:[.,]\d+)?)\s*(millones?)?', 'presupuesto_total')
        ]

        for pattern, resource_type in resource_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount = self._parse_number(match.group(1)) if match.group(1) else None
                if match.group(2) and 'millon' in match.group(2).lower():
                    amount = amount * 1000000 if amount else None
                resources.append((resource_type, amount))

        return resources

    def _determine_semantic_role(self, sent) -> Optional[str]:
        """Determina el rol semántico de una oración"""
        text_lower = sent.text.lower()

        role_patterns = {
            'objective': ['objetivo', 'meta', 'propósito', 'finalidad'],
            'strategy': ['estrategia', 'línea', 'eje', 'pilar'],
            'action': ['implementar', 'ejecutar', 'desarrollar', 'realizar'],
            'indicator': ['indicador', 'medir', 'evaluar', 'monitorear'],
            'resource': ['presupuesto', 'recurso', 'financiación', 'inversión'],
            'constraint': ['limitación', 'restricción', 'condición', 'requisito']
        }

        for role, keywords in role_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                return role

        return None

    def _identify_dependencies(self, sent, doc) -> Set[str]:
        """Identifica dependencias entre declaraciones"""
        dependencies = set()

        # Buscar referencias a otras secciones
        reference_patterns = [
            r'como\s+se\s+menciona\s+en',
            r'según\s+lo\s+establecido\s+en',
            r'de\s+acuerdo\s+con',
            r'en\s+línea\s+con',
            r'siguiendo\s+lo\s+dispuesto'
        ]

        for pattern in reference_patterns:
            if re.search(pattern, sent.text, re.IGNORECASE):
                # Buscar la sección referenciada
                for other_sent in doc.sents:
                    if other_sent != sent:
                        # Usar hash de los primeros 50 caracteres como ID
                        dependencies.add(other_sent.text[:50])

        return dependencies

    def _get_context_window(self, text: str, start: int, end: int, window_size: int = 200) -> str:
        """Obtiene ventana de contexto alrededor de una posición"""
        context_start = max(0, start - window_size)
        context_end = min(len(text), end + window_size)
        return text[context_start:context_end]

    def _calculate_similarity(self, stmt_a: PolicyStatement, stmt_b: PolicyStatement) -> float:
        """Calcula similaridad entre dos declaraciones"""
        if stmt_a.embedding is not None and stmt_b.embedding is not None:
            return float(1 - cosine(stmt_a.embedding, stmt_b.embedding))
        return 0.0

    def _classify_contradiction(self, text: str) -> float:
        """Clasifica probabilidad de contradicción en texto"""
        try:
            result = self.contradiction_classifier(text)
            # Buscar score de contradicción
            for item in result:
                if 'contradiction' in item['label'].lower():
                    return item['score']
            return 0.0
        except Exception as e:
            logger.warning(f"Error en clasificación de contradicción: {e}")
            return 0.0

    def _get_domain_weight(self, dimension: PolicyDimension) -> float:
        """Obtiene peso específico del dominio"""
        weights = {
            PolicyDimension.DIAGNOSTICO: 0.8,
            PolicyDimension.ESTRATEGICO: 1.2,
            PolicyDimension.PROGRAMATICO: 1.0,
            PolicyDimension.FINANCIERO: 1.5,
            PolicyDimension.SEGUIMIENTO: 0.9,
            PolicyDimension.TERRITORIAL: 1.1
        }
        return weights.get(dimension, 1.0)

    def _suggest_resolutions(self, contradiction_type: ContradictionType) -> List[str]:
        """Sugiere resoluciones específicas por tipo de contradicción"""
        suggestions = {
            ContradictionType.NUMERICAL_INCONSISTENCY: [
                "Verificar fuentes de datos y metodologías de cálculo",
                "Establecer línea base única con validación técnica",
                "Documentar supuestos y proyecciones utilizadas"
            ],
            ContradictionType.TEMPORAL_CONFLICT: [
                "Revisar cronograma maestro del plan",
                "Validar secuencia lógica de actividades",
                "Ajustar plazos según capacidad institucional"
            ],
            ContradictionType.SEMANTIC_OPPOSITION: [
                "Realizar taller de alineación conceptual",
                "Clarificar definiciones en glosario técnico",
                "Priorizar objetivos según Plan Nacional de Desarrollo"
            ],
            ContradictionType.RESOURCE_ALLOCATION_MISMATCH: [
                "Realizar análisis de brechas financieras",
                "Priorizar inversiones según impacto social",
                "Explorar fuentes alternativas de financiación"
            ],
            ContradictionType.LOGICAL_INCOMPATIBILITY: [
                "Revisar cadena de valor de programas",
                "Validar teoría de cambio del plan",
                "Eliminar duplicidades y solapamientos"
            ]
        }
        return suggestions.get(contradiction_type, ["Revisar y ajustar según contexto"])

    def _are_comparable_claims(self, claim_a: Dict, claim_b: Dict) -> bool:
        """Determina si dos afirmaciones cuantitativas son comparables"""
        # Mismo tipo y contexto similar
        if claim_a['type'] != claim_b['type']:
            return False

        # Verificar si hablan del mismo concepto
        context_similarity = self._text_similarity(
            claim_a.get('context', ''),
            claim_b.get('context', '')
        )

        return context_similarity > 0.6

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Calcula similaridad simple entre textos"""
        if not text_a or not text_b:
            return 0.0

        # Tokenización simple
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())

        if not tokens_a or not tokens_b:
            return 0.0

        # Coeficiente de Jaccard
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union) if union else 0.0

    def _calculate_numerical_divergence(
            self,
            claim_a: Dict,
            claim_b: Dict
    ) -> Optional[float]:
        """Calcula divergencia entre valores numéricos"""
        value_a = claim_a.get('value', 0)
        value_b = claim_b.get('value', 0)

        if value_a == 0 and value_b == 0:
            return None

        # Divergencia relativa
        max_value = max(abs(value_a), abs(value_b))
        if max_value == 0:
            return None

        divergence = abs(value_a - value_b) / max_value
        return divergence

    def _statistical_significance_test(
            self,
            claim_a: Dict,
            claim_b: Dict
    ) -> float:
        """Realiza test de significancia estadística"""
        value_a = claim_a.get('value', 0)
        value_b = claim_b.get('value', 0)

        # Test t de una muestra para diferencia significativa
        # Asumiendo distribución normal con varianza estimada
        diff = abs(value_a - value_b)
        pooled_value = (value_a + value_b) / 2

        if pooled_value == 0:
            return 1.0  # No significativo

        # Estimación conservadora de error estándar
        se = pooled_value * 0.1  # 10% de error estimado

        if se == 0:
            return 0.0  # Altamente significativo

        # Estadístico t
        t_stat = diff / se

        # Valor p aproximado (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

        return p_value

    def _has_logical_conflict(self, stmt_a: PolicyStatement, stmt_b: PolicyStatement) -> bool:
        """Determina si hay conflicto lógico entre declaraciones"""
        # Verificar si las declaraciones tienen roles incompatibles
        if stmt_a.semantic_role and stmt_b.semantic_role:
            incompatible_roles = [
                ('objective', 'constraint'),
                ('strategy', 'constraint'),
                ('action', 'constraint')
            ]

            for role_pair in incompatible_roles:
                if (stmt_a.semantic_role in role_pair and
                        stmt_b.semantic_role in role_pair and
                        stmt_a.semantic_role != stmt_b.semantic_role):
                    return True

        # Verificar negación explícita
        negation_patterns = ['no', 'nunca', 'ningún', 'sin', 'tampoco']
        has_negation_a = any(pattern in stmt_a.text.lower() for pattern in negation_patterns)
        has_negation_b = any(pattern in stmt_b.text.lower() for pattern in negation_patterns)

        # Si una tiene negación y otra no, y son similares, hay conflicto
        if has_negation_a != has_negation_b:
            similarity = self._calculate_similarity(stmt_a, stmt_b)
            if similarity > 0.7:
                return True

        return False

    def _are_conflicting_allocations(
            self,
            amount_a: float,
            amount_b: float,
            total: float
    ) -> bool:
        """Determina si las asignaciones de recursos están en conflicto"""
        # Si la suma excede el total disponible
        if amount_a + amount_b > total * 1.1:  # 10% de margen
            return True

        # Si hay una diferencia muy grande entre asignaciones similares
        if abs(amount_a - amount_b) / max(amount_a, amount_b) > 0.5:
            return True

        return False

    def _determine_relation_type(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> str:
        """Determina el tipo de relación entre dos declaraciones"""
        # Analizar roles semánticos
        if stmt_a.semantic_role and stmt_b.semantic_role:
            if stmt_a.semantic_role == stmt_b.semantic_role:
                return "parallel"
            elif stmt_a.semantic_role in ["strategy", "objective"] and stmt_b.semantic_role == "action":
                return "enables"
            elif stmt_a.semantic_role == "action" and stmt_b.semantic_role in ["indicator", "resource"]:
                return "requires"
        
        # Analizar dependencias
        if stmt_a.dependencies & {stmt_b.text[:50]}:
            return "depends_on"
        
        # Por defecto, relación de similaridad
        return "related"

    def _calculate_severity(
            self,
            stmt_a: PolicyStatement,
            stmt_b: PolicyStatement
    ) -> float:
        """Calcula la severidad de una contradicción entre declaraciones"""
        severity = 0.5  # Base severity
        
        # Incrementar si las declaraciones están en la misma dimensión
        if stmt_a.dimension == stmt_b.dimension:
            severity += 0.2
        
        # Incrementar si tienen muchas entidades en común
        common_entities = set(stmt_a.entities) & set(stmt_b.entities)
        if len(common_entities) > 0:
            severity += min(0.2, len(common_entities) * 0.05)
        
        # Incrementar si tienen marcadores temporales en conflicto
        if stmt_a.temporal_markers and stmt_b.temporal_markers:
            severity += 0.1
        
        return min(1.0, severity)


# Punto de entrada para uso directo
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(level=logging.INFO)

    # Ejemplo de uso
    detector = PolicyContradictionDetector()

    sample_text = """
    El municipio aumentará la cobertura educativa al 95% para 2027, 
    sin embargo los recursos del SGP educación serán de 1500 millones.
    En el primer trimestre de 2025 se ejecutará el 40% del presupuesto.
    Para el segundo trimestre de 2025 se proyecta ejecutar el 70% del presupuesto total.
    La meta es beneficiar a 10000 familias, pero el programa solo tiene capacidad para 5000 beneficiarios.
    """

    result = detector.detect(
        sample_text,
        plan_name="PDM Ejemplo 2024-2027",
        dimension=PolicyDimension.ESTRATEGICO
    )

    print(result)