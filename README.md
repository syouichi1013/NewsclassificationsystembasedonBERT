# News-Insight: BERT-based Classification & Interpretability

## Introduction
A deep learning-based news classification system featuring real-time inference and Model Interpretability. It utilizes a fine-tuned BERT model to classify Chinese news. The system not only predicts categories but also provides an Attention Map to visualize the model's learning/decision process (i.e., which specific words or tokens the model focused on to make its judgment).

## Data & Model
The system is built on the THUCNews dataset and uses a pre-trained BERT-Base-Chinese model. 

## File Structure
```text
./
├── bert_pretrain/          # Pre-trained BERT model files
├── THUCNews/
│   ├── data/               # Dataset (Class list)
│   └── saved_dict/         # Fine-tuned model checkpoint (bert.ckpt)
├── app.py                  # FastAPI Backend (Inference & Attention extraction)
├── index.html              # Frontend (Visualization & Attention Map)
├── bert.py                 # Model architecture
└── utils.py                # Text preprocessing and tokenization
```

## Training
Run the training script to fine-tune the BERT model on the THUCNews dataset:
```bash
python run.py --model bert
```

## Running the System
Start the Backend: Execute the FastAPI script to load the model:

```bash
python app.py
Launch the Interface: Open index.html in your browser.
```

Analyze: Input news text and click "Run Inference." The system will output the predicted label and the Attention Map showing exactly which characters influenced the decision.
