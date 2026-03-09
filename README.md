# GBmooer-on-NLP-label-noise
This is source code from paper GBmooer, against to label noise in NLP yield.
## Project Structure
GBmooer/
├── bert-base-uncased/        # English pretrained model (BERT-Base-Uncased)
├── chinese-bert-wwm-ext/     # Chinese pretrained model
├── checkpoints/              # Saved model checkpoints
├── data/                     # Experimental datasets
├── logs/                     # Training logs and results
When running the code, you should first create and ensure the existence of these directories. In the data directory, you need to prepare your dataset in CSV format, where each sample is organized as label, text, and noisy label.
Please ensure that all required libraries for running the code are installed. Then, simply run:
```bash
bash run1.sh
