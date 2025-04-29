# Spatiotemporal Unemployment Forecasting in FEMA Region 9: A Comparative Analysis of Classical and Deep Learning Models

## Introduction
Predicting unemployment accurately presents significant challenges due to the unique economic behaviors of different regions. This study compares classical statistical models (SARIMA, AR) with advanced deep learning approaches (LSTM, GAT-LSTM, GCN hybrids) across 90 counties in FEMA Region 9 using 412 months of historical data. Our analysis reveals distinct performance patterns based on economic stability and crisis history, providing insights for policymakers and economists.

## Models Applied
### Classical Models
- **AR (AutoRegressive)**  
  Baseline model capturing linear temporal patterns
- **SARIMA (Seasonal ARIMA)**  
  Handles seasonality and non-stationary trends

### Deep Learning Models
- **LSTM**  
  Vanilla sequence model for temporal patterns
- **GAT-LSTM**  
  Graph Attention Networks with LSTM for spatial-temporal relationships
- **GCN-LSTM**  
  Graph Convolutional Networks with LSTM
- **GCN-TCN**  
  Hybrid model combining Graph Convolutions with Temporal Convolutional Networks

## Results
### Model Performance Comparison
| Model      | Average RMSE |
|------------|--------------|
| AR         | 0.557        |
| SARIMA     | 0.503        |
| LSTM       | 0.769        |
| GAT-LSTM   | 1.898        |
| GCN-LSTM   | 1.947        |
| GCN-TCN    | 2.019        |

**Key Findings:**
- SARIMA demonstrated superior performance in stable economic regions
- Graph-based models (GAT-LSTM/GCN-LSTM) excelled in counties with historical economic volatility
- Hybrid architectures showed particular strength in capturing sudden economic shocks

## Repository Structure
'''
C:.
└───Project
    ├───data
    ├───demo
    │   └───fig
    ├───graphs
    │   ├───ar_plots
    │   ├───gat_lstm_plots
    │   ├───gcn_tcn_plots
    │   ├───lstm_plots
    │   ├───results
    │   │   └───plots
    │   ├───rolling_stats_plots
    │   ├───sarima_plots
    │   ├───test_plots
    │   └───timeseries_plots
    ├───presentation
    ├───reports
    │   └───Research references
    ├───research_paper
    │   ├───Latex
    │   │   └───Fig
    │   └───Word
    └───src
        ├───component
        │   ├───deep learning models
        │   │   └───lstm_models
        │   └───shallow_models
        │       ├───AR
        │       └───SARIMA
        └───tests
'''


## Requirements
**Core Dependencies**
'''pip install pandas numpy matplotlib scikit-learn'''
**Deep Learning **
'''
pip install torch torchvision torchaudio
pip install torch-geometric
'''


## Contributing
This project welcomes contributions through:
- Issue tracking
- Model improvements
- Additional visualization features
- Documentation enhancements

Please create a new branch for proposed changes and submit pull requests for review.

## Authors
-Khush Shah
-Dr. Amir Jafari
-Dr. Michael Mann