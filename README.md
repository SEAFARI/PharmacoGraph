# PharmacoGraph-3D: Interpretable Drug-Target Affinity Prediction

<div align="center">
  
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://pharmacograph-suxvsetk8axdzq5farzkwo.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-Davis%20Benchmark-blue?style=for-the-badge)](https://tdcommons.ai/multi_pred_tasks/dti/)

<br><br>

<img src="https://github.com/user-attachments/assets/8448cb5d-897a-4489-a567-9860190bfa41" alt="PharmacoGraph-3D Heatmap Visualization" />
<br>
<em>(Caption: GNNExplainer highlights the functional hotspots of Imatinib driving target binding)</em>

</div>

### Overview
PharmacoGraph-3D is an interactive computational biology tool that predicts the binding affinity of small molecule drugs to protein targets. Rather than functioning as a "black box," this project utilizes **Explainable AI (XAI)** to identify and visualize the specific atomic substructures (pharmacophores) driving the model's predictions in an interactive 3D environment.

### The Architecture
The pipeline is built entirely in PyTorch and deployed via Streamlit, bridging complex graph mathematics with a user-friendly biochemical interface.

1.  **Data Engineering:** Chemical SMILES strings are parsed into topological graph tensors (nodes = atoms, edges = bonds) using `PyTorch Geometric` and `RDKit`.
2.  **Model Backbone:** A 2-layer Graph Convolutional Network (GCN) with global mean pooling learns molecular embeddings and predicts continuous binding affinity.
3.  **Explainability:** `GNNExplainer` interrogates the trained network, selectively masking atoms to extract node-level importance scores based on their contribution to the final prediction.
4.  **3D Visualization:** The 2D graphs are mapped to optimized 3D physical coordinates where the AI importance scores dictate the radius and color of glowing spatial heatmaps using `py3Dmol`.

### Repository Structure
*   `automated_pharmacophore_identification.ipynb`: The master notebook containing the complete pipeline (data ingestion, model definition, training loop, and visual extraction).
*   `app.py`: The frontend application code for the interactive Streamlit deployment.
*   `drug_affinity_gnn.pth`: The pre-trained model weights.
*   `requirements.txt`: Environment dependencies for Streamlit Community Cloud.

### Quick Start (Local Deployment)
To run the interactive web application on your local machine:

```bash
# 1. Clone the repository
git clone [https://github.com/yourusername/PharmacoGraph-3D.git](https://github.com/yourusername/PharmacoGraph-3D.git)
cd PharmacoGraph-3D
```
```
# 2. Install requirements
pip install -r requirements.txt
```
# 3. Launch the dashboard
```
streamlit run app.py
```
