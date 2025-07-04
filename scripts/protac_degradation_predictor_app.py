import protac_degradation_predictor as pdp

from typing import Dict, List, Literal
import difflib

import torch
import numpy as np
from rdkit import Chem
import gradio as gr


def gradio_app(
        protac_smiles: str | List[str],
        e3_ligase: str | List[str],
        target_uniprot: str | List[str],
        cell_line: str | List[str],
        use_models_from_cv: bool = False,
) -> Dict[str, np.ndarray]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avail_uniprots = pdp.avail_uniprots()
    avail_cells = pdp.avail_cell_lines()

    if target_uniprot not in avail_uniprots:
        suggestions = difflib.get_close_matches(target_uniprot, avail_uniprots, n=3, cutoff=0.5)
        suggestion_text = "Did you mean:" + ", ".join(suggestions) + "?" if suggestions else "No close matches found."
        raise gr.Error(f"Invalid Uniprot ID. {suggestion_text}", duration=None)

    if cell_line not in avail_cells:
        suggestions = difflib.get_close_matches(cell_line, avail_cells, n=3, cutoff=0.5)
        suggestion_text = "Did you mean:" + ", ".join(suggestions) + "?" if suggestions else "No close matches found."
        raise gr.Error(f"Invalid Cell Line. {suggestion_text}", duration=None)

    prediction = pdp.get_protac_active_proba(
        protac_smiles,
        e3_ligase,
        target_uniprot,
        cell_line,
        device=device,
        use_models_from_cv=use_models_from_cv,
    )
    mean_pred = {"Active": prediction['mean'], "Inactive": 1 - prediction['mean']}
    majvote_pred = "Active" if prediction['majority_vote'] else "Inactive"
    return mean_pred, majvote_pred

demo = gr.Interface(
    fn=gradio_app,
    inputs=[
        gr.Textbox(placeholder="PROTAC SMILES", label="PROTAC SMILES"),
        gr.Dropdown(pdp.avail_e3_ligases(), label="E3 ligase"),
        gr.Textbox(placeholder="E.g., Q92769", label="Target Uniprot"),
        gr.Textbox(placeholder="E.g., HeLa", label="Cell line"),
        gr.Checkbox(label="Use models trained during cross-validation"),
    ],
    outputs=[gr.Label(label="Average probability (confidence)"), gr.Label(label="Majority vote prediction")],
    title="PROTAC Degradation Predictor",
    examples=[        [
            "Cc1ncsc1-c1ccc(CNC(=O)[C@@H]2C[C@@H](O)CN2C(=O)[C@@H](NC(=O)COCCCCCCCCCOCC(=O)Nc2ccc(C(=O)Nc3ccc(F)cc3N)cc2)C(C)(C)C)cc1",
            "VHL",
            "Q92769",
            "HeLa",
        ],
    ],
    description="Predict whether a PROTAC is active or inactive.",
)

demo.launch()
