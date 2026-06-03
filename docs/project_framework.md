# Tokenizer Evaluation Framework

This project is organized around three evaluation axes for pretrained audio tokenizers and representation models.

1. Reconstruction

   Encode and decode audio, then compare the decoded waveform against the source with waveform, spectral, perceptual, and optional listening-test artifacts.

2. Semantic Representation

   Extract latent embeddings and evaluate whether semantic labels are linearly or locally separable. This includes visualization, probing, retrieval, and clustering-style diagnostics.

3. Downstream Generation

   Use tokenizer latents as conditioning or discrete/continuous tokens for generation tasks, then evaluate generated audio for fidelity, diversity, controllability, and prompt/task alignment.

The current implementation focuses on one semantic-representation benchmark: SAME and MERT embeddings on NSynth instrument family labels.
