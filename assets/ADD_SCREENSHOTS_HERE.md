# Screenshots needed

This pipeline needs a GPU and downloads CLIP + FLAN-T5-Large (~1.1GB) plus a
sample video, so these renders couldn't be reproduced outside Colab. Run the
notebook in Colab (T4 GPU) and drop in screenshots of:

1. The Gradio dashboard header + summary output (`Dashboard_Overview.png`)
2. The CLIP confidence timeline with scene-cut markers (`Confidence_Timeline.png`)
3. The scene segment Gantt-style map (`Scene_Segment_Map.png`)
4. The label frequency bar chart (`Label_Frequency.png`)
5. The PCA embedding scatter plot (`PCA_Embeddings.png`)

Once added, reference them in README.md the same way GhostHunt's assets are
referenced — delete this file afterward.
