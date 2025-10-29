"""
Deterministic Seed Factory
Generates reproducible seeds for all stochastic operations
"""

import hashlib
import hmac
import random
from typing import Dict, Any, Optional

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None  # type: ignore


class SeedFactory:
    """
    Factory for generating deterministic seeds
    
    Ensures:
    - Reproducibility: Same inputs → same seed
    - Uniqueness: Different contexts → different seeds
    - Cryptographic quality: HMAC-SHA256 derivation
    """
    
    # Fixed salt for seed derivation (should be configured per deployment)
    DEFAULT_SALT = b"PDM_EVALUATOR_V2_DETERMINISTIC_SEED_2025"
    
    def __init__(self, fixed_salt: Optional[bytes] = None):
        self.salt = fixed_salt or self.DEFAULT_SALT
    
    def create_deterministic_seed(
        self,
        correlation_id: str,
        file_checksums: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Generate deterministic seed from correlation ID and context
        
        Args:
            correlation_id: Unique workflow instance identifier
            file_checksums: Dict of {filename: sha256_checksum}
            context: Additional context (question_id, policy_area, etc.)
        
        Returns:
            32-bit integer seed (0 to 2^32-1)
        
        Example:
            >>> factory = SeedFactory()
            >>> seed1 = factory.create_deterministic_seed("run-001", {"data.json": "abc123"})
            >>> seed2 = factory.create_deterministic_seed("run-001", {"data.json": "abc123"})
            >>> assert seed1 == seed2  # Reproducible
        """
        
        # Build deterministic input string
        components = [correlation_id]
        
        # Add file checksums (sorted for determinism)
        if file_checksums:
            sorted_checksums = sorted(file_checksums.items())
            for filename, checksum in sorted_checksums:
                components.append(f"{filename}:{checksum}")
        
        # Add context (sorted for determinism)
        if context:
            sorted_context = sorted(context.items())
            for key, value in sorted_context:
                components.append(f"{key}={value}")
        
        # Combine with deterministic separator
        seed_input = "|".join(components).encode('utf-8')
        
        # HMAC-SHA256 for cryptographic quality
        seed_hmac = hmac.new(
            key=self.salt,
            msg=seed_input,
            digestmod=hashlib.sha256
        )
        
        # Convert to 32-bit integer
        seed_bytes = seed_hmac.digest()[:4]  # First 4 bytes
        seed_int = int.from_bytes(seed_bytes, byteorder='big')
        
        return seed_int
    
    def configure_global_random_state(self, seed: int):
        """
        Configure all random number generators with seed
        
        Sets:
        - Python random module
        - NumPy random state
        - (Add torch, tensorflow if needed)
        
        Args:
            seed: Deterministic seed
        """
        
        # Python random module
        random.seed(seed)
        
        # NumPy
        if NUMPY_AVAILABLE and np is not None:
            np.random.seed(seed)
        
        # TODO: Add torch.manual_seed(seed) if PyTorch is used
        # TODO: Add tf.random.set_seed(seed) if TensorFlow is used


class DeterministicContext:
    """
    Context manager for deterministic execution
    
    Usage:
        with DeterministicContext(correlation_id="run-001") as seed:
            # All random operations are deterministic
            result = some_stochastic_function()
    """
    
    def __init__(
        self,
        correlation_id: str,
        file_checksums: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None,
        fixed_salt: Optional[bytes] = None
    ):
        self.correlation_id = correlation_id
        self.file_checksums = file_checksums
        self.context = context
        self.factory = SeedFactory(fixed_salt=fixed_salt)
        
        self.seed: Optional[int] = None
        self.previous_random_state = None
        self.previous_numpy_state = None
    
    def __enter__(self) -> int:
        """Enter deterministic context"""
        
        # Generate deterministic seed
        self.seed = self.factory.create_deterministic_seed(
            correlation_id=self.correlation_id,
            file_checksums=self.file_checksums,
            context=self.context
        )
        
        # Save current random states
        self.previous_random_state = random.getstate()
        if NUMPY_AVAILABLE and np is not None:
            self.previous_numpy_state = np.random.get_state()
        
        # Configure with deterministic seed
        self.factory.configure_global_random_state(self.seed)
        
        return self.seed
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit deterministic context and restore previous state"""
        
        # Restore previous random states
        if self.previous_random_state:
            random.setstate(self.previous_random_state)
        
        if NUMPY_AVAILABLE and np is not None and self.previous_numpy_state:
            np.random.set_state(self.previous_numpy_state)
        
        return False


def create_deterministic_seed(
    correlation_id: str,
    file_checksums: Optional[Dict[str, str]] = None,
    **context_kwargs
) -> int:
    """
    Convenience function for creating deterministic seed
    
    Args:
        correlation_id: Unique workflow instance ID
        file_checksums: Dict of file checksums
        **context_kwargs: Additional context as keyword arguments
    
    Returns:
        Deterministic 32-bit integer seed
    
    Example:
        >>> seed = create_deterministic_seed(
        ...     "run-001",
        ...     question_id="P1-D1-Q001",
        ...     policy_area="P1"
        ... )
    """
    factory = SeedFactory()
    return factory.create_deterministic_seed(
        correlation_id=correlation_id,
        file_checksums=file_checksums,
        context=context_kwargs if context_kwargs else None
    )
