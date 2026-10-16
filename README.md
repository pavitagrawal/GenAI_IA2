# Autoregressive Japanese Fairy-Tale Text Generation

**Course:** Foundations of Generative AI (ICT4403) - Assignment 2  
**Task:** Next-word text generation using Simple RNN, Bidirectional RNN, LSTM, and GRU  
**Implementation:** `FGAI_Assignment2_Autoregressive_Text_Generation.ipynb`

## 1. Problem statement and objective

The task is to develop an autoregressive text-generation system that predicts the next word from a fixed-length sequence of preceding words. Four recurrent neural-network architectures must be implemented and trained under comparable experimental conditions:

1. Simple Recurrent Neural Network (Simple RNN)
2. Bidirectional Recurrent Neural Network (Bidirectional RNN)
3. Long Short-Term Memory network (LSTM)
4. Gated Recurrent Unit network (GRU)

The principal objectives are to:

- select and describe a suitable textual dataset;
- clean and tokenize the corpus;
- construct input-target sequences for next-word prediction;
- train all four models using the same data partitions and training protocol;
- compare loss, perplexity, prediction accuracy, parameter count, and training time;
- generate text from identical seed sequences at several sampling temperatures;
- examine coherence, diversity, contextual consistency, and repetition; and
- identify the model that provides the most effective autoregressive generation.

The experiment uses a chronological 80% training, 10% validation, and 10% test split. Splitting is performed before sliding windows are constructed, preventing almost-identical overlapping windows from crossing partition boundaries and producing misleading evaluation scores.

## 2. Dataset description and source link

### Selected dataset

The project uses *Japanese Fairy Tales* by Yei Theodora Ozaki, Project Gutenberg eBook #4018.

- Dataset information page: https://www.gutenberg.org/ebooks/4018
- Direct UTF-8 text: https://www.gutenberg.org/cache/epub/4018/pg4018.txt
- Bundled local copy: `data/japanese_fairy_tales_pg4018.txt`
- SHA-256: `a17eec00fcaaba3d8e3496945486a3be8a984cbb694d4f6db0f4616e782af115`

Project Gutenberg identifies the work as public domain in the United States. Users outside the United States should verify the copyright rules applicable in their country.

### Dataset characteristics

The book is a collection of 22 thematically related Japanese folk narratives translated and retold in English. It contains recurring narrative patterns, dialogue, fantasy vocabulary, named characters, descriptive passages, and long-range story structure. These properties make it well suited to comparing the sequence-modelling behavior of recurrent architectures.

The corpus is less commonly used in student language-modelling assignments than Shakespeare, *Alice's Adventures in Wonderland*, or generic news datasets. It therefore provides a more original experiment without sacrificing linguistic coherence.

Verified source-file and extracted-corpus characteristics are:

| Characteristic | Value |
|---|---:|
| Downloaded source size | 403,288 bytes |
| Usable corpus characters | 368,820 |
| Approximate modelling tokens | 78,622 |
| Unique tokens before vocabulary restriction | 5,042 |
| Language | English |
| Content | 22 Japanese folk narratives |

The notebook removes the Project Gutenberg header, licence footer, table of contents, dedication, and preface. It normalizes line endings, repairs line-break hyphenation, converts typographic quotation marks, lowercases the corpus, and tokenizes words, internal apostrophes, numbers, and sentence punctuation.

The tokenizer is fitted only on the training partition. Unknown validation or test words are represented by an `<OOV>` token. The model vocabulary is limited to a maximum of 8,000 entries.

## 3. Models implemented

All four models receive sequences of 30 tokens and predict the token immediately following each sequence. Every architecture uses a 128-dimensional trainable embedding, 20% dropout, and a dense vocabulary-sized output layer that produces unnormalized logits.

### Simple RNN

The Simple RNN maintains a recurrent hidden state and processes the context from left to right. It is computationally straightforward but can suffer from vanishing gradients, making long-range relationships difficult to retain.

Configuration: 128 recurrent units.

### Bidirectional RNN

The Bidirectional RNN processes the observed context in forward and reverse directions and concatenates the two representations. Each direction receives only tokens from the input context; the next-word target is outside that context, so the target is not leaked to the model.

Configuration: 64 SimpleRNN units in each direction, giving a concatenated output width of 128.

### LSTM

The LSTM uses input, forget, and output gates together with a persistent cell state. These mechanisms help preserve useful information across longer sequences and reduce the vanishing-gradient problem.

Configuration: 128 LSTM units.

### GRU

The GRU uses update and reset gates to control its hidden state. It provides gated memory similar to an LSTM but uses fewer gates and normally requires fewer parameters.

Configuration: 128 GRU units.

### Comparable experimental conditions

The following settings remain identical across all models:

- sequence length: 30 tokens;
- embedding dimension: 128;
- effective recurrent output width: 128;
- batch size: 128;
- optimizer: Adam;
- initial learning rate: 0.001;
- gradient clipping norm: 1.0;
- maximum epochs: 15;
- early-stopping patience: 3 epochs;
- validation-loss checkpointing;
- ReduceLROnPlateau learning-rate schedule;
- training-window stride: 2;
- random seed: 42; and
- the same training, validation, and test partitions.

Parameter counts are reported rather than assumed equal because the recurrent cells contain different numbers of gates.

## 4. Execution steps and required dependencies

### Recommended environment

- Python 3.10 or newer
- Jupyter Notebook, JupyterLab, or Google Colab
- GPU recommended for final training, although CPU execution is supported

### Dependencies

The required packages are listed in `requirements.txt`:

- TensorFlow 2.15-2.18
- NumPy
- pandas
- Matplotlib
- seaborn
- requests
- Jupyter

### Local execution

1. Open a terminal in this project directory.
2. Optionally create and activate a virtual environment.
3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Jupyter:

   ```bash
   jupyter notebook
   ```

5. Open `FGAI_Assignment2_Autoregressive_Text_Generation.ipynb`.
6. Ensure that `QUICK_MODE = False` for the final experiment.
7. Select **Kernel -> Restart Kernel and Run All Cells**.
8. Allow all four models to finish training and verify that every cell completes without an exception.
9. Commit the notebook together with the populated `results/` and `models/` directories.

### Google Colab execution

1. Upload the entire project folder to Google Drive, including the `data` directory.
2. Open the notebook in Google Colab.
3. Select **Runtime -> Change runtime type -> GPU**.
4. Make the project folder the working directory.
5. Keep `QUICK_MODE = False` and select **Runtime -> Run all**.

The bundled dataset is used first. If it is missing, the notebook downloads the same UTF-8 source from Project Gutenberg and caches it locally.

`QUICK_MODE = True` reduces the number of epochs, generation length, and training-window density. It is intended only for debugging and must not be used for the final reported results.

## 5. Results obtained

The notebook automatically writes the following reproducible outputs after a complete run:

- `results/training_summary.csv`: parameter count, epochs, best epoch, validation loss, and training time;
- `results/model_evaluation.csv`: held-out loss, perplexity, and accuracy;
- `results/generated_samples.csv`: generated text for every model, seed, and temperature;
- `results/generation_metrics.csv`: diversity, repetition, coherence, and context-consistency proxies;
- `results/final_model_ranking.csv`: the final weighted model comparison;
- `results/learning_curves.png`: training and validation curves;
- `results/model_comparison.png`: perplexity, accuracy, and training-time comparison;
- `results/temperature_analysis.png`: temperature-effect plots; and
- `models/*.keras`: the best validation checkpoint for each model.

### Final quantitative results table

The table below must contain values from `results/model_evaluation.csv` after running the notebook in final mode. Results are intentionally not fabricated in this repository before model training.

| Model | Parameters | Best epoch | Test loss | Test perplexity | Test accuracy | Training time (s) |
|---|---:|---:|---:|---:|---:|---:|
| Simple RNN | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook |
| Bidirectional RNN | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook |
| LSTM | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook |
| GRU | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook | Generated by notebook |

The notebook displays these values and exports them automatically. Before submission, replace the four rows above with the produced numbers or commit the generated CSV file and add its values here.

## 6. Model comparison and evaluation

### Quantitative measures

- **Cross-entropy loss:** measures the error in the predicted next-token probability distribution. Lower is better.
- **Perplexity:** calculated as `exp(loss)`. It represents the model's effective average uncertainty over the next token. Lower is better.
- **Exact next-word accuracy:** measures how often the most probable predicted token exactly equals the target token. Higher is better. This is a strict measure for a vocabulary containing thousands of possible outputs.
- **Training time:** measures wall-clock training duration. Lower is faster, but the value depends on hardware and TensorFlow configuration.
- **Parameter count:** provides context for the accuracy-speed trade-off.

### Qualitative measures

Every model generates from the same three seeds at the same temperatures. The continuations are assessed using:

- **coherence:** whether phrases and sentences form understandable local and global progressions;
- **diversity:** whether the continuation uses varied vocabulary and events;
- **contextual consistency:** whether the continuation maintains the topic, entities, and direction implied by the seed; and
- **repetition:** whether words, phrases, or events become trapped in loops.

The notebook supplements human reading with reproducible proxy metrics:

| Metric | Interpretation | Preferred direction |
|---|---|---|
| Distinct-1 | Unique words divided by generated words | Higher |
| Distinct-2 | Unique bigrams divided by generated bigrams | Higher |
| Trigram repetition | Proportion of repeated generated trigrams | Lower |
| Seen-bigram ratio | Generated bigrams also present in training data | Higher, within reason |
| Seed-content overlap | Seed content words reused in the continuation | Moderate to high |

Seen-bigram ratio is treated as a local-coherence proxy, not proof of overall narrative coherence. Seed overlap is treated as a context-consistency proxy, but an extremely high value may also indicate repetition. Generated samples must therefore be inspected manually using the 1-5 rubric provided in the notebook.

### Overall model-ranking rule

The final notebook ranking uses the following declared weights:

- 35% inverse test perplexity;
- 20% test accuracy;
- 20% seen-bigram ratio;
- 15% distinct-2 diversity; and
- 10% inverse trigram repetition.

Training time is reported separately and is not allowed to override generation quality. This prevents a fast but incoherent model from being selected as the best generator.

## 7. Sampling-temperature analysis

Text is generated at temperatures `0.5`, `0.8`, and `1.2`. A fixed top-k value of 50 is used in every condition, so temperature remains the controlled independent variable.

### Temperature 0.5

Dividing logits by 0.5 sharpens the probability distribution. High-probability words become much more likely, normally improving grammatical safety and local coherence. The disadvantages are conservative wording, reduced lexical diversity, and a greater risk of repeated phrases.

### Temperature 0.8

Temperature 0.8 provides a balance between confident predictions and exploration. It is expected to retain reasonable narrative structure while introducing more varied vocabulary than temperature 0.5. It is therefore a strong default for comparing the perceptual quality of the four models.

### Temperature 1.2

Temperature 1.2 flattens the probability distribution. Less probable words are sampled more frequently, normally increasing distinct-1 and distinct-2 diversity. However, the continuation may contain weaker transitions, inconsistent entities, unusual word combinations, and grammatical errors.

### Required evidence from the completed run

The final temperature discussion should use `results/temperature_analysis.png` and `results/generation_metrics.csv` to answer:

1. Does distinct-2 increase as temperature rises?
2. Does the seen-bigram coherence proxy decrease at high temperature?
3. Is repetition highest at low temperature?
4. Which temperature produces the best human-rated balance of coherence and diversity?
5. Do all architectures respond to temperature in the same way?

The expected general pattern is that lower temperatures improve predictability but reduce variety, whereas higher temperatures improve variety at the cost of coherence. The submitted discussion must confirm or reject this expectation using the actual generated samples.

## 8. Conclusion and best-performing model

The best-performing model is the architecture ranked first in `results/final_model_ranking.csv` after a complete final-mode run and confirmation through the human qualitative rubric. The notebook prints this model automatically in its final **Data-driven conclusion** cell.

At present, no model is labelled as the measured winner in this README because the final four-model training run has not been executed in the current environment. Naming LSTM, GRU, or another architecture without the produced metrics would fabricate an experimental conclusion.

After executing the notebook, replace the paragraph below with the values printed by the final cell:

> **Best-performing model:** `[MODEL NAME]`. It achieved a test perplexity of `[VALUE]`, an exact next-word accuracy of `[VALUE]`, and an overall weighted score of `[VALUE]`. It was selected because `[brief quantitative comparison]`. At temperature `[VALUE]`, its samples showed `[brief observation about coherence, diversity, contextual consistency, and repetition]`. Therefore, `[MODEL NAME]` provided the most effective autoregressive generation for the selected dataset.

A theoretically expected outcome is that LSTM or GRU will outperform the ungated Simple RNN on perplexity and longer-range coherence, while GRU may train faster than LSTM. This is a hypothesis, not the reported result; the final conclusion must follow the generated evidence.

## 9. Limitations

- The corpus is relatively small compared with modern language-model datasets.
- The historical English may contain archaic or culturally dated language.
- Next-word accuracy does not measure every valid alternative word.
- Automated qualitative proxies cannot fully measure narrative quality.
- Training times cannot be compared across different hardware environments.
- A bidirectional encoder can use both directions inside a fixed observed context, but it is less appropriate than a forward recurrent model for stateful token-by-token streaming deployment.

## 10. Reproducibility

- Random seeds are fixed at 42.
- Deterministic TensorFlow operations are requested where supported.
- Dataset source and SHA-256 are recorded.
- The vocabulary is fitted only on training data.
- Dataset partitions are constructed before sliding windows.
- Identical seeds and temperatures are used for all architectures.
- Model checkpoints, result tables, plots, and generated samples are saved automatically.
