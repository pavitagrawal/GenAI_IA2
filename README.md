# Autoregressive Japanese Fairy-Tale Text Generation

Foundations of Generative AI (ICT4403) - Assignment 2

This project compares Simple RNN, Bidirectional RNN, LSTM, and GRU next-word models under a matched experimental protocol. Each architecture learns from the same leakage-safe chronological split and generates from the same seeds at temperatures 0.5, 0.8, and 1.2.

## Dataset

The project uses *Japanese Fairy Tales* by Yei Theodora Ozaki, Project Gutenberg eBook #4018:

- Dataset page: https://www.gutenberg.org/ebooks/4018
- UTF-8 text: https://www.gutenberg.org/cache/epub/4018/pg4018.txt

The collection contains 22 thematically related Japanese folk narratives in English. It is an uncommon alternative to heavily reused Shakespeare and news corpora, while its recurring narrative language, dialogue, characters, and story arcs suit next-word generation. Project Gutenberg marks it public domain in the USA; users elsewhere should check local law.

The exact UTF-8 source is bundled at `data/japanese_fairy_tales_pg4018.txt` (SHA-256: `a17eec00fcaaba3d8e3496945486a3be8a984cbb694d4f6db0f4616e782af115`). The notebook uses this local copy first and downloads it only when the file is missing. It verifies the digest, removes Gutenberg boilerplate and front matter, normalizes layout artifacts, and tokenizes words plus punctuation.

## Models

- Simple RNN: 128 recurrent units
- Bidirectional RNN: 64 SimpleRNN units per direction, concatenated to width 128
- LSTM: 128 recurrent units
- GRU: 128 recurrent units

All models use the same 128-dimensional embedding, context length, split, optimizer, batch size, stopping rule, and generation settings. Parameter counts are reported because gate structures differ.

## Run

1. Use Python 3.10+ (Google Colab is suitable).
2. Install dependencies: `pip install -r requirements.txt`.
3. Open `FGAI_Assignment2_Autoregressive_Text_Generation.ipynb`.
4. Run all cells from top to bottom.
5. Keep `QUICK_MODE = False` for final results. Set it to `True` only for a short smoke test.

A GPU is recommended but not required. Internet access is not needed when the bundled `data/` directory remains beside the notebook; the download URL is retained as a fallback and for source attribution.

## Evaluation and outputs

The notebook reports test loss, perplexity, exact next-word accuracy, training time, parameter count, learning curves, and generated samples. It also measures distinct-1/2 diversity, repeated-trigram rate, training-bigram coverage as a local-coherence proxy, and seed content overlap as a context-consistency proxy.

Generated artifacts are written automatically:

- `models/*.keras`: best checkpoint for each architecture
- `results/training_summary.csv`: time, epochs, and validation loss
- `results/model_evaluation.csv`: held-out loss, perplexity, and accuracy
- `results/generated_samples.csv`: every model/seed/temperature sample
- `results/generation_metrics.csv`: qualitative proxy metrics
- `results/final_model_ranking.csv`: transparent weighted ranking
- `results/*.png`: learning, comparison, and temperature plots

## Model comparison and conclusion

Run the notebook before filling the final report: actual rankings depend on the hardware, TensorFlow version, and trained weights. The notebook automatically identifies a preliminary winner using a declared weighted score, then provides a 1-5 human rubric for coherence, diversity, contextual consistency, and repetition. The submitted conclusion should combine both and cite concrete sample excerpts.

## Reproducibility notes

- Random seeds are fixed at 42 and deterministic TensorFlow operations are requested where supported.
- Train/validation/test splits are chronological and are created before sliding windows, preventing cross-boundary window leakage.
- The vocabulary is fitted only on the training partition.
- Top-k remains fixed at 50, so temperature is the controlled generation variable.
- Wall-clock timings are hardware-dependent; record the device information printed by the notebook.

