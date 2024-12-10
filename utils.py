import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load and perform basic preprocessing of the data."""
    df = pd.read_csv(file_path)
    return df

def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables using LabelEncoder."""
    df = df.copy()
    encoders = {}
    
    for col in ['surface', 'ab', 'a', 'b']:
        encoders[col] = LabelEncoder()
        df[f'{col}_encoded'] = encoders[col].fit_transform(df[col])
    
    return df, encoders

def create_chemical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create chemical-based features."""
    df = df.copy()
    
    # Asterisk features
    df['has_asterisk_a'] = df['a'].str.contains(r'\*', regex=True).astype(int)
    df['has_asterisk_b'] = df['b'].str.contains(r'\*', regex=True).astype(int)
    df['has_asterisk_ab'] = df['ab'].str.contains(r'\*', regex=True).astype(int)
    
    # Length features
    df['len_a'] = df['a'].str.len()
    df['len_b'] = df['b'].str.len()
    df['len_ab'] = df['ab'].str.len()
    
    # Element counts
    elements = ['C', 'O', 'H', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    for elem in elements:
        df[f'{elem}_count_a'] = df['a'].str.count(elem)
        df[f'{elem}_count_b'] = df['b'].str.count(elem)
        df[f'{elem}_count_ab'] = df['ab'].str.count(elem)
    
    # Molecular complexity scores
    df['complexity_a'] = sum(df[f'{elem}_count_a'] for elem in elements)
    df['complexity_b'] = sum(df[f'{elem}_count_b'] for elem in elements)
    df['complexity_ab'] = sum(df[f'{elem}_count_ab'] for elem in elements)
    
    return df

def create_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create surface-specific features."""
    df = df.copy()
    
    # Basic surface features
    df['surface_metal'] = df['surface'].str.extract(r'([A-Z][a-z]?)')[0]
    df['surface_metal_encoded'] = LabelEncoder().fit_transform(df['surface_metal'])
    df['surface_facet'] = df['surface'].str.extract(r'\((\d+)\)')[0]
    df['surface_facet'] = pd.to_numeric(df['surface_facet'])
    
    # Alloy detection and composition
    df['is_alloy'] = df['surface'].str.contains(r'[A-Z][a-z]?[A-Z]').astype(int)
    
    # Extract first and second metals for alloys
    df['first_metal'] = df['surface'].str.extract(r'([A-Z][a-z]?)[A-Z]')[0]
    df['second_metal'] = df['surface'].str.extract(r'[A-Z][a-z]?([A-Z][a-z]?)')[0]
    
    # Encode metals
    metal_encoder = LabelEncoder()
    all_metals = pd.concat([df['first_metal'].dropna(), df['second_metal'].dropna()]).unique()
    metal_encoder.fit(all_metals)
    
    df['first_metal_encoded'] = metal_encoder.transform(df['first_metal'].fillna(all_metals[0]))
    df['second_metal_encoded'] = metal_encoder.transform(df['second_metal'].fillna(all_metals[0]))
    
    # Add atomic properties (periodic table data)
    atomic_properties = {
        'Ag': {'electronegativity': 1.93, 'atomic_radius': 144},
        'Au': {'electronegativity': 2.54, 'atomic_radius': 144},
        'Cu': {'electronegativity': 1.90, 'atomic_radius': 128},
        'Pd': {'electronegativity': 2.20, 'atomic_radius': 137},
        'Pt': {'electronegativity': 2.28, 'atomic_radius': 139},
        'Ni': {'electronegativity': 1.91, 'atomic_radius': 124},
        'Fe': {'electronegativity': 1.83, 'atomic_radius': 126},
        'Co': {'electronegativity': 1.88, 'atomic_radius': 125},
        'Zn': {'electronegativity': 1.65, 'atomic_radius': 134},
        # Add more metals as needed
    }
    
    def get_property(metal: str, prop: str) -> float:
        if pd.isna(metal) or metal not in atomic_properties:
            return 0
        return atomic_properties[metal][prop]
    
    df['first_metal_electronegativity'] = df['first_metal'].apply(lambda x: get_property(x, 'electronegativity'))
    df['second_metal_electronegativity'] = df['second_metal'].apply(lambda x: get_property(x, 'electronegativity'))
    df['first_metal_radius'] = df['first_metal'].apply(lambda x: get_property(x, 'atomic_radius'))
    df['second_metal_radius'] = df['second_metal'].apply(lambda x: get_property(x, 'atomic_radius'))
    
    # Calculate property differences for alloys
    df['electronegativity_diff'] = abs(df['first_metal_electronegativity'] - df['second_metal_electronegativity'])
    df['atomic_radius_diff'] = abs(df['first_metal_radius'] - df['second_metal_radius'])
    
    return df

def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features between surface and reactants."""
    df = df.copy()
    
    # Basic interactions
    df['surface_a_interaction'] = df['surface_encoded'] * df['a_encoded']
    df['surface_b_interaction'] = df['surface_encoded'] * df['b_encoded']
    df['a_b_interaction'] = df['a_encoded'] * df['b_encoded']
    df['metal_facet_interaction'] = df['surface_metal_encoded'] * df['surface_facet']
    
    # Advanced interactions for alloys
    df['alloy_reactant_interaction'] = df['is_alloy'] * (df['complexity_a'] + df['complexity_b'])
    df['metal_difference_effect'] = df['electronegativity_diff'] * df['atomic_radius_diff']
    
    return df

def get_reaction_type(row: pd.Series) -> str:
    """Determine the type of reaction based on reactants and products."""
    # Check for adsorption reactions
    if row['b'] == '*' and not row['a'].endswith('*'):
        return 'adsorption'
    
    # Check for surface reactions (both reactants are adsorbed species)
    elif row['a'].endswith('*') and row['b'].endswith('*'):
        return 'surface_reaction'
    
    # Check for desorption reactions
    elif row['a'].endswith('*') and row['b'] == '*':
        return 'desorption'
    
    # Default to other
    return 'other'

def split_by_reaction_type(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split the dataset by reaction type."""
    df['reaction_type'] = df.apply(get_reaction_type, axis=1)
    return {reaction_type: group for reaction_type, group in df.groupby('reaction_type')}

def get_feature_columns() -> List[str]:
    """Return the list of feature columns to use for modeling."""
    return [
        'surface_encoded', 'ab_encoded', 'a_encoded', 'b_encoded',
        'has_asterisk_a', 'has_asterisk_b', 'has_asterisk_ab',
        'len_a', 'len_b', 'len_ab',
        'C_count_a', 'C_count_b', 'C_count_ab',
        'O_count_a', 'O_count_b', 'O_count_ab',
        'H_count_a', 'H_count_b', 'H_count_ab',
        'N_count_a', 'N_count_b', 'N_count_ab',
        'S_count_a', 'S_count_b', 'S_count_ab',
        'P_count_a', 'P_count_b', 'P_count_ab',
        'F_count_a', 'F_count_b', 'F_count_ab',
        'Cl_count_a', 'Cl_count_b', 'Cl_count_ab',
        'Br_count_a', 'Br_count_b', 'Br_count_ab',
        'I_count_a', 'I_count_b', 'I_count_ab',
        'surface_metal_encoded', 'surface_facet',
        'is_alloy', 'first_metal_encoded', 'second_metal_encoded',
        'first_metal_electronegativity', 'second_metal_electronegativity',
        'first_metal_radius', 'second_metal_radius',
        'electronegativity_diff', 'atomic_radius_diff',
        'surface_a_interaction', 'surface_b_interaction',
        'a_b_interaction', 'metal_facet_interaction',
        'alloy_reactant_interaction', 'metal_difference_effect',
        'complexity_a', 'complexity_b', 'complexity_ab'
    ] 