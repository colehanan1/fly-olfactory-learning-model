"""
DoOR odor feature encoder for Stage 2.

Integrates door-python-toolkit to convert odor_name to fixed-length ORN response vectors.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
from door_toolkit import DoOREncoder


# Mapping from Stage 1 odor names to DoOR database names
ODOR_NAME_MAP = {
    "Hexanol": "1-hexanol",
    "Benzaldehyde": "benzaldehyde",
    "Ethyl Butyrate": "ethyl butyrate",
    "3-Octonol": "3-octanol",
    "Linalool": "(-)-linalool",
    "Citral": "citral",
    "Apple Cider Vinegar": "acetic acid",  # Main component of vinegar
    "AIR": None,  # No DoOR equivalent - will use fill_missing
}


class DoorOdorEncoder:
    """Encode odor names as DoOR ORN response vectors with caching."""
    
    def __init__(self, cache_dir: str = "door_cache", fill_missing: float = 0.0, cache_enabled: bool = True):
        """
        Initialize DoOR encoder.
        
        Args:
            cache_dir: Directory containing DoOR cache files  
            fill_missing: Value to use for missing ORN responses
            cache_enabled: Whether to cache per-odor vectors
        """
        self.fill_missing = fill_missing
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, np.ndarray] = {}
        # Initialize DoOREncoder with cache path
        self._encoder = DoOREncoder(cache_path=cache_dir, use_torch=False)
        self._feature_dim: int = 78  # DoOR has 78 receptors
        
    def encode_odor(self, odor_name: str) -> np.ndarray:
        """
        Encode a single odor name to ORN response vector.
        
        Args:
            odor_name: Odor name (e.g., "Benzaldehyde")
            
        Returns:
            1D array of ORN responses (78-dim)
        """
        # Check cache
        if self.cache_enabled and odor_name in self._cache:
            return self._cache[odor_name]
        
        # Map odor name to DoOR database name
        door_name = ODOR_NAME_MAP.get(odor_name, odor_name.lower())
        
        # Handle AIR or unmapped odors
        if door_name is None:
            response = np.full(self._feature_dim, self.fill_missing, dtype=np.float32)
        else:
            # Get DoOR response vector using encode() method
            try:
                response = self._encoder.encode(door_name)
                # Fill missing values (NaN)
                response = np.nan_to_num(response, nan=self.fill_missing)
            except (KeyError, ValueError, Exception) as e:
                # Fallback: if odor not found, return fill_missing vector
                print(f"Warning: DoOR lookup failed for '{odor_name}' (mapped to '{door_name}'): {e}")
                print(f"         Using fill_missing={self.fill_missing}")
                response = np.full(self._feature_dim, self.fill_missing, dtype=np.float32)
        
        # Ensure float32 type
        response = response.astype(np.float32)
        
        # Cache if enabled
        if self.cache_enabled:
            self._cache[odor_name] = response
        
        return response
    
    def encode_dataframe(self, df: pd.DataFrame, odor_column: str = "odor_name") -> np.ndarray:
        """
        Encode all odors in a DataFrame to feature matrix.
        
        Args:
            df: DataFrame with odor_column
            odor_column: Name of column containing odor names
            
        Returns:
            2D array (n_samples, n_orn_features)
        """
        odor_names = df[odor_column].values
        
        # Encode each odor
        features = []
        for odor_name in odor_names:
            features.append(self.encode_odor(odor_name))
        
        return np.array(features)
    
    @property
    def feature_dim(self) -> Optional[int]:
        """Get feature dimensionality (None if not yet determined)."""
        return self._feature_dim
    
    @property
    def cache_size(self) -> int:
        """Get number of cached odors."""
        return len(self._cache)
