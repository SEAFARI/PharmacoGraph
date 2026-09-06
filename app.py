import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils.smiles import from_smiles
from torch_geometric.explain import Explainer, GNNExplainer
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import matplotlib as mpl
import matplotlib.colors as mcolors

# --- Page Setup ---
st.set_page_config(page_title="PharmacoGraph-3D", layout="wide")
st.title("Interpretable GNN for Drug-Target Affinity Prediction")
st.markdown("Visualizing atomic feature importance with **Graph Neural Networks** and **GNNExplainer**.")

# --- Define Architecture & Load Weights ---
class DrugAffinityGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super(DrugAffinityGNN, self).__init__()
        self.conv1 = GCNConv(9, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)

@st.cache_resource
def load_trained_model():
    model = DrugAffinityGNN(hidden_channels=64)
    # Load weights on CPU for Streamlit Cloud compatibility
    model.load_state_dict(torch.load('drug_affinity_gnn.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    return model

model = load_trained_model()

# --- Sidebar Inputs ---
st.sidebar.header("Molecule Selection")
preset_drugs = {
    "Imatinib (Leukemia - BCR-ABL)": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CC=CC=C5",
    "Gefitinib (Lung Cancer - EGFR)": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
    "Erlotinib (Pancreatic/Lung - EGFR)": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"
}

selected_drug = st.sidebar.selectbox("Choose a benchmark drug or paste SMILES:", list(preset_drugs.keys()))
smiles_input = st.sidebar.text_area("SMILES String", preset_drugs[selected_drug])

# --- Inference & Visualization ---
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("3D Pharmacophore Heatmap")
    try:
        # Convert SMILES to PyG Graph
        pyg_graph = from_smiles(smiles_input)
        x_float = pyg_graph.x.float()
        dummy_batch = torch.zeros(x_float.shape[0], dtype=torch.int64)

        # Run Prediction
        with torch.no_grad():
            predicted_affinity = model(x_float, pyg_graph.edge_index, dummy_batch).item()

        # Run GNNExplainer
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=100),
            explanation_type='model',
            node_mask_type='object',
            model_config=dict(mode='regression', task_level='graph', return_type='raw'),
        )
        explanation = explainer(x_float, pyg_graph.edge_index, batch=dummy_batch)
        scores = explanation.node_mask.detach().cpu().numpy()
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Build 3D Conformer with RDKit
        mol = Chem.MolFromSmiles(smiles_input)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.MMFFOptimizeMolecule(mol)
        mol_block = Chem.MolToMolBlock(mol)

        # Render with py3Dmol
        color_map = mpl.colormaps['plasma']
        view = py3Dmol.view(width=750, height=500)
        view.addModel(mol_block, 'mol')
        
        # Bright silver backbone
        view.setStyle({'model': -1}, {'stick': {'color': '#999999', 'radius': 0.15}})

        for i, score in enumerate(scores):
            if score > 0.4:
                pos = mol.GetConformer().GetAtomPosition(i)
                hex_color = mcolors.to_hex(color_map(float(score)))
                view.addSphere({
                    'center': {'x': pos.x, 'y': pos.y, 'z': pos.z},
                    'radius': float(score) * 1.05,
                    'color': hex_color,
                    'alpha': 0.92
                })

        view.setBackgroundColor('black')
        view.zoomTo()

        # Display directly in Streamlit via raw HTML
        components.html(view._make_html(), height=520, width=770)

    except Exception as e:
        st.error(f"Error parsing molecule. Ensure you entered a valid SMILES string. Error: {e}")

with col2:
    st.subheader("Model Output & Interpretation")
    st.metric(label="Predicted Binding Affinity ($pK_d$)", value=f"{predicted_affinity:.2f}")
    
    st.info(
        """
        **How to read this visualization:**
        - **Light Silver Wireframe:** Low-importance atomic backbone.
        - **Glowing Yellow/Pink Spheres:** High-importance atomic subgraphs identified by `GNNExplainer`.
        - These regions correspond to predicted **pharmacophore motifs** driving the target binding interaction.
        """
    )
    st.markdown("---")
    st.markdown("**Dataset:** Davis Benchmark")
    st.markdown("**Backbone:** 2-Layer Graph Convolutional Network (GCN)")
