# Machine Learning Models for Chemical Reaction Energy Prediction

## Project Overview
This project develops specialized machine learning models to predict reaction energies for heterogeneous catalysis, focusing on surface-mediated reactions. The models are designed to handle different types of reactions separately, recognizing that different reaction mechanisms may require different prediction approaches.

## Dataset Description
The dataset contains 3,269 chemical reactions from the CatApp database, including:
- Reaction energies
- Surface information (metal type, facet)
- Reactant and product species
- Reference information

### Data Distribution
- **Adsorption Reactions**: 1,794 reactions (55%)
  - Simple adsorption (e.g., CO → CO*)
  - Dissociative adsorption (e.g., H2 → 2H*)
- **Surface Reactions**: 760 reactions (23%)
  - Reactions between adsorbed species
  - Surface-mediated transformations
- **Other Reactions**: 715 reactions (22%)
  - Complex mechanisms
  - Multiple step reactions

## Model Architecture

### 1. Adsorption Model
- **Algorithm**: Gradient Boosting Regressor
- **Performance**:
  - R² Score: 0.67
  - RMSE: 1.10 eV
  - Predictions within ±0.5 eV: 63.7%
  - Predictions within ±1.0 eV: 86.3%
- **Key Features**:
  - Surface electronic properties
  - Molecular complexity descriptors
  - Adsorption site information

### 2. Surface Reaction Model
- **Algorithm**: Random Forest Regressor
- **Performance**:
  - R² Score: 0.85
  - RMSE: 0.69 eV
  - Predictions within ±0.5 eV: 54.4%
  - Predictions within ±1.0 eV: 88.6%
- **Key Features**:
  - Reactant-surface interactions
  - Bond formation/breaking descriptors
  - Surface geometry information

### 3. Other Reactions Model
- **Algorithm**: Random Forest Regressor
- **Performance**:
  - R² Score: 0.22
  - RMSE: 0.94 eV
  - Predictions within ±0.5 eV: 66.7%
  - Predictions within ±1.0 eV: 87.0%
- **Key Features**:
  - Complex reaction descriptors
  - Multiple step indicators
  - Combined interaction terms

## Feature Engineering

### Chemical Features
- Element-wise composition analysis
- Molecular complexity scores
- Surface binding indicators (*)
- Molecular size descriptors

### Surface Features
- Metal type encoding
- Facet information
- Alloy composition analysis
- Electronic properties:
  - Electronegativity
  - Atomic radius
  - Surface energy (where available)

### Interaction Features
- Surface-adsorbate interactions
- Reactant-reactant coupling
- Metal-facet correlations
- Alloy-specific descriptors

## Key Findings and Insights

### 1. Reaction Type Dependencies
- **Surface Reactions** show highest predictability (R² = 0.85)
  - Well-defined reaction mechanisms
  - Consistent surface interactions
  - Clear structure-property relationships

- **Adsorption Reactions** show good reliability (R² = 0.67)
  - Simple mechanisms are highly predictable
  - Alloy surfaces introduce complexity
  - Surface structure effects are significant

- **Other Reactions** need improvement (R² = 0.22)
  - Complex mechanisms reduce predictability
  - Multiple steps increase uncertainty
  - Need for more specialized descriptors

### 2. Surface Effects
- **Simple Metal Surfaces**:
  - High prediction accuracy
  - Consistent behavior
  - Well-understood mechanisms

- **Alloy Surfaces**:
  - Increased prediction errors
  - Complex electronic effects
  - Composition-dependent behavior

- **Facet Effects**:
  - (211) facets show higher variability
  - Structure sensitivity varies by reaction
  - Surface reconstruction effects

### 3. Challenging Cases

#### Adsorption Reactions
- CO on Pd3Sb(111): 13.79 eV error
  - Possible electronic structure complexity
  - Surface reconstruction effects
  - Limited training data for similar systems

#### Surface Reactions
- N2 formation on Re surfaces: ~2.46 eV error
  - Complex electronic structure
  - Strong correlation effects
  - Multiple reaction pathways

#### Other Reactions
- OH* + H2 → H2O on Pd alloys: ~6.13 eV error
  - Complex reaction mechanism
  - Multiple interaction sites
  - Electronic structure effects

## Recommendations for Improvement

### 1. Model Enhancements
- Implement hierarchical modeling approaches
- Develop metal-specific sub-models
- Add ensemble methods for robust predictions
- Include uncertainty quantification

### 2. Feature Development
- Add electronic structure descriptors
- Develop reaction mechanism indicators
- Include surface reconstruction effects
- Add thermodynamic descriptors

### 3. Data Improvements
- Gather more data for challenging cases
- Balance dataset across reaction types
- Include more diverse surface structures
- Add experimental validation data

## Applications and Impact

### 1. Catalyst Design
- Rapid screening of new catalysts
- Property prediction for novel materials
- Optimization of surface composition

### 2. Process Development
- Reaction condition optimization
- Catalyst stability prediction
- Process efficiency improvement

### 3. Research Applications
- Mechanism investigation
- Structure-property relationships
- Design principle development

## Technical Details

### Dependencies
- Python 3.8+
- scikit-learn
- pandas
- numpy

### Model Files
- `models.py`: Model implementations
- `utils.py`: Utility functions
- `train.py`: Training pipeline
- `model_base.py`: Base model class

## Future Work
1. Implement deep learning approaches
2. Add molecular fingerprinting
3. Develop automated feature selection
4. Include temperature and pressure effects
5. Add reaction path analysis

## References
- CatApp database
- Relevant publications
- Method documentation # CatalystML
