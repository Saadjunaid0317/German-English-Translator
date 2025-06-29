# German-to-English Neural Machine Translation 🇩🇪 -> 🇬🇧

This repository contains the complete code for a German-to-English Neural Machine Translation (NMT) system. The model is built with TensorFlow and uses an Encoder-Decoder architecture with Bahdanau Attention. The project includes scripts for training the model and an interactive web application built with Streamlit for easy translation.

## Features
- **Encoder-Decoder Architecture** with GRU layers.
- **Bahdanau Attention Mechanism** for improved translation quality.
- **Full Training Pipeline** (`main.py`) to train the model from scratch.
- **Interactive Web UI** (`app.py`) built with Streamlit for easy inference.

## How to Run This Project

### 1. Prerequisites
- Python 3.8+
- Git

### 2. Setup
First, clone the repository to your local machine:
```bash
git clone [https://github.com/YourUsername/German-English-Translator.git](https://github.com/YourUsername/German-English-Translator.git)
cd German-English-Translator
```

Next, it is recommended to create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

Install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Download Data
The training data is from the Tatoeba project. You can download the `deu-eng.zip` file from [manythings.org/anki](http://www.manythings.org/anki/) and place it in the project root folder.

### 4. Train the Model
Run the training script to create the model checkpoints and tokenizer files. This is a long process and can take several hours.
```bash
python main.py
```
This will create the `training_checkpoints` folder and the `.pickle` files (which are ignored by Git).

### 5. Launch the Web App
Once training is complete, run the Streamlit application to start translating!
```bash
streamlit run app.py
```