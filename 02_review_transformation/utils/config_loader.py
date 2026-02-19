"""Configuration loader for review transformation experiments."""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default location.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_years(config: Dict[str, Any]) -> List[int]:
    """Extract years list from config."""
    return config.get('years', [])


def get_venues(config: Dict[str, Any]) -> List[str]:
    """Extract venues list from config (e.g. ICLR, NeurIPS)."""
    return config.get('venues', ['ICLR'])


def get_data_root(config: Dict[str, Any]) -> Optional[str]:
    """Extract data_root from config (relative to repo root)."""
    return config.get('data_root')


def get_papers_per_year(config: Dict[str, Any]) -> Optional[int]:
    """Extract papers_per_year from config."""
    papers_per_year = config.get('papers_per_year')
    if papers_per_year is None or papers_per_year == -1:
        return None
    return papers_per_year


def get_llms(config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract LLM configurations from config."""
    return config.get('llms', [])


def get_output_config(config: Dict[str, Any]) -> Dict[str, str]:
    """Extract output configuration from config."""
    return config.get('output', {})


def get_llm_reviews_dir(config: Dict[str, Any]) -> Optional[str]:
    """Extract optional llm_reviews_dir from config (relative to repo root). Used by hybrid."""
    return config.get('output', {}).get('llm_reviews_dir')


def get_api_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract API configuration from config."""
    return config.get('api', {})


def get_manuscript_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract manuscript configuration from config."""
    return config.get('manuscript', {})

