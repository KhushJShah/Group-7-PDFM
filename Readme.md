# Capstone Proposal: Migration Trends of Low-Income Families in FEMA Region 9

## **Objective**
The primary goal of this project is to analyze and predict migration trends of low-income families in FEMA Region 9 (California, Arizona, Nevada) in response to climate change impacts. The focus is on identifying spatial and temporal patterns to inform equitable relocation strategies. Specifically, the project involves:
- Modeling the spatial and temporal dynamics of low-income groups.
- Incorporating complex relationships between regions, such as migration flows and resource availability.

---

## **Dataset**
The project will utilize three primary datasets:
1. **Health Analytics**: Data on health indicators like chronic illnesses, physical activity, and healthcare access.
2. **Unemployment**: County-level unemployment rates over time.
3. **Poverty**: Socioeconomic data related to income levels and poverty rates.

Additional datasets may be incorporated as needed, such as transportation networks or climate data, to enrich the analysis.

---

## **Rationale**
Graph Neural Networks (GNNs) are particularly suited for this problem because they:
- Leverage spatial features to model relationships between counties (e.g., proximity, shared resources).
- Incorporate temporal dynamics when combined with time-series models like LSTMs or GRUs.
- Provide insights into complex socioeconomic patterns that traditional models may overlook.

By using GNNs, we can better understand how climate change impacts migration trends and provide actionable insights for policymakers.

---

## **Approach**
The project will proceed in several phases:

### **Phase 1: Data Collection and Preprocessing**
- Collect geospatial data (e.g., county-level maps, housing costs, employment rates) and demographic data (e.g., income levels, population density).
- Represent data as both tabular datasets (for traditional models) and graph structures (for GNNs):
  - **Nodes**: Represent counties or regions with features like unemployment rates, health indicators, and poverty levels.
  - **Edges**: Represent relationships such as proximity, migration flows, or transportation networks.
- Preprocess data to ensure consistency across formats.

### **Phase 2: Modeling Approaches**
1. **Traditional Models**:
   - Use regression-based models (e.g., linear regression, random forests) as a baseline for predicting trends.
2. **Graph Neural Networks**:
   - Experiment with architectures like Graph Convolutional Networks (GCNs) for static spatial relationships.
   - Extend to Spatio-Temporal GNNs (ST-GNNs) for dynamic modeling of migration patterns over time.

### **Phase 3: Validation**
- Evaluate model performance using metrics like RMSE and MAE.
- Compare GNN-based models against traditional approaches to assess improvements in predictive accuracy.

### **Phase 4: Usability Evaluation**
- Collaborate with stakeholders (e.g., FEMA officials) to ensure the model's outputs are interpretable and actionable for policy-making.

### **Phase 5: Reporting**
- Document findings and provide recommendations for equitable relocation strategies.

---

## **Timeline**

| Week(s)       | Task                                                                 |
|---------------|----------------------------------------------------------------------|
| Weeks 1–2     | Familiarization with project requirements and open-source tools.    |
| Weeks 3–8     | Design, develop, and build a working proof of concept for one region.|
| Weeks 9–12    | Scale the model to other regions in FEMA Region 9.                  |
| Weeks 13–16   | Extrapolate missing regions and refine predictions.                 |
| Weeks 17–18   | Final reporting, documentation, and presentation of outcomes.       |

---

## **Expected Number of Students**
This project is suitable for a team of two students due to its interdisciplinary nature and scope.

---

## **Possible Issues**
Potential challenges include:
1. **Limited Training Data**:
   - Finding appropriate datasets for spatial analysis may be a limiting factor.
   - Additional data sources may be required to improve model accuracy.
2. **Interdisciplinary Collaboration**:
   - The project spans multiple domains (e.g., geography, data science), requiring effective communication across fields.

---

## **Proposed by**
- Dr. Michael Mann  
- Khush Shah  
- Email: [khushjayant.shah@gwu.edu](mailto:khushjayant.shah@gwu.edu)

---

## **Instructor**
- Amir Jafari  
- Email: [ajafari@gmail.com](mailto:ajafari@gmail.com)

---

## **GitHub Repository**
[Capstone Project Repository](https://github.com/amir-jafari/Capstone)
