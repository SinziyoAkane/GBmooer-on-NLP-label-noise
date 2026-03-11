# GBmooer-on-NLP-label-noise
This is source code from paper GBmooer, against to label noise in NLP yield.
## Project Structure
```
GBmooer/
├── bert-base-uncased/        # English pretrained model (BERT-Base-Uncased)
├── chinese-bert-wwm-ext/     # Chinese pretrained model
├── checkpoints/              # Saved model checkpoints
├── data/                     # Experimental datasets
├── logs/                     # Training logs and results
```

## Data Preparation

When running the code, you should first create and ensure the existence of these directories. In the data directory, you need to prepare your dataset in CSV format, where each sample is organized as label, text, and noisy label.
- `label` : the ground truth class label  
- `text` : the input text  
- `noisy_label` : the potentially noisy label

Please ensure that all required libraries for running the code are installed. Then, simply run:
```bash
bash run1.sh
```

## Supported Datasets

The following datasets are supported in GBmooer:

- **TREC** – Question classification dataset ([Download](https://cogcomp.seas.upenn.edu/Data/QA/QC/))
- **AG News** – News topic classification dataset ([Download](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz))
- **IMDB** – Movie review sentiment classification dataset ([Download](https://ai.stanford.edu/~amaas/data/sentiment/))
- **Chnsenticorp** – Chinese sentiment analysis dataset ([Download](https://github.com/ymcui/Chinese-BERT-wwm/tree/master/data/chnsenticorp))
