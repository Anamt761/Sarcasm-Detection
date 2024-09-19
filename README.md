# Sarcasm-Detection

This project focuses on detecting sarcasm in text, particularly in news headlines, using various machine learning and deep learning approaches. By leveraging natural language processing (NLP) techniques, the model aims to distinguish between sarcastic and non-sarcastic content to improve human-computer interaction and textual understanding in social media and other domains.

**Project Overview**

The main goal of this project is to detect sarcasm in text-based news headlines, which is challenging due to the nuanced nature of sarcastic expressions. We aim to develop a model capable of accurately identifying sarcasm to enhance various applications, such as sentiment analysis and content moderation.

**Available Link of Dataset**

The dataset used for this project can be found on Kaggle: https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

**Dataset Description**

The News Headlines Dataset for Sarcasm Detection was collected from two distinct news websites: TheOnion and HuffPost. The Onion specializes in producing sarcastic versions of current events, and we sourced all headlines from the News in Brief and News in categories (which are sarcastic). In contrast, we collected non-sarcastic news headlines from HuffPost, providing a balanced dataset with clear distinctions between sarcastic and non-sarcastic content.

Each record in the dataset contains three attributes:

**is_sarcastic:** A binary label indicating whether the headline is sarcastic (1) or not (0).

**headline:** The news article's headline.

**article_link:** A link to the original news article, useful for collecting supplementary information.

This dataset is ideal for training models to detect sarcasm in a structured, professional context, enabling more accurate natural language understanding (NLU) applications.

**Background**

With the rise of social media and online platforms, text communication has dramatically increased. As users generate vast amounts of content, distinguishing sarcasm becomes crucial for understanding user sentiment, detecting misleading content, and improving human-computer interaction.

**Focus**

This project focuses on:

- Detecting sarcastic expressions in text.
- Enhancing the accuracy of sarcasm detection using machine learning (ML) and deep learning (DL) models.
- Exploring various feature engineering techniques such as word embeddings and natural language understanding (NLU).

**Setup Instructions**

To set up the project on your local machine, follow these steps:

Clone the repository:

git clone https://github.com/Anamt761/Sarcasm-Detection.git

cd Sarcasm-Detection

**Key Libraries**

The following key libraries are used in the project:

- NumPy: For numerical computations.
- Pandas: For data manipulation and analysis.
- scikit-learn: For machine learning model implementation and evaluation.
- TensorFlow/Keras: For deep learning model implementation.
- NLTK/Spacy: For text preprocessing and NLP tasks.
- Matplotlib/Seaborn: For data visualization.


a. Open Google Colab at colab.research.google.com.

b. Upload the Sarcasm_Detection_ML+DL+EDA.ipynb notebook or use the Colab link from the repository.

c. Install the required dependencies by adding this to a code cell at the start:

- pip install numpy
- pip install spacy
- pip install tqdm
- pip install xgboost
- pip install Lightgbm
- pip install nltk
- pip install scikit-learn
- pip install transformers
- pip install tensorflow
- pip install pandas
- pip install gensim
- pip install sentence-transformers

**Brief Analysis of How to Run the Scripts or Notebooks**

This section provides a guide on how to run the scripts or Jupyter notebooks included in this project, covering data preprocessing, model training, and evaluation.

**Preprocessing**

1. **Tokenization**: Split text into individual tokens (words or phrases).
2. **Lemmatization**: Convert words to their base forms to standardize vocabulary.
3. **Stemming**: Reduce words to their root forms to handle inflections.
4. **Removal of Links**: Eliminate hyperlinks from the text to focus on the core content.
5. **Stop Words Removal**: Remove common words (e.g., "and", "the") that do not contribute significant meaning.
6. **Digit Removal**: Remove numerical values to focus solely on textual content.
7. **Text Normalization**: Convert text to lowercase and handle special characters to maintain consistency.

**Feature Extraction**

- **Textual Features**:
- **POS (Part-of-Speech)**: Extract grammatical structures (nouns, verbs, etc.) to capture syntactic features.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Generate word importance scores to emphasize distinguishing terms.
- **N-grams**: Capture word sequences to understand context and patterns within the text.
- **Word Embeddings**:
  - **Word2Vec**: Learn vector representations of words to capture semantic meaning.
  - **GloVe**: Utilize global co-occurrence statistics to create word embeddings.
  - **FastText**: Extend word embeddings by learning representations at subword levels.
  - **Sentence Transformer**: Create embeddings at the sentence level for enhanced contextual understanding.

**Model Training**

- **Shallow Machine Learning Models**:
  
  - **Support Vector Machine (SVM)**: Find hyperplanes that separate classes.
  - **K-Nearest Neighbors (KNN)**: Classify based on the nearest training examples in feature space.
  - **Decision Tree (DT)**: Use a tree structure to model decisions and consequences.
  - **Logistic Regression (LR)**: Perform binary classification based on a linear combination of features.
  
- **Ensemble Models**:
  
  - **Gradient Boosting (GB)**: Build models iteratively, correcting errors of previous models.
  - **XGBoost (XGB)**: A faster version of Gradient Boosting.
  - **Random Forest (RF)**: Construct multiple decision trees for improved accuracy.
  - **AdaBoost**: Combine weak classifiers to form a strong classifier.
  - **LightGBM**: A fast and efficient gradient-boosting framework.

- **Deep Learning Models**:

  - **LSTM (Long Short-Term Memory)**: Capture sequential dependencies in text data.
  - **Bi-LSTM (Bidirectional LSTM)**: Enhance LSTM by processing data in both directions.
  - **CNN (Convolutional Neural Networks)**: Identify spatial patterns and features in text data.
  - **RNN (Recurrent Neural Networks)**: Handle sequential data but with simpler mechanisms compared to LSTM.

**Model Evaluation**

- **Accuracy**: Measure the overall correctness of predictions.
- **Precision**: Calculate the percentage of correctly predicted positive results out of all predicted positives.
- **Recall**: Compute the percentage of actual positives correctly identified by the model.
- **F1-Score**: The harmonic mean of precision and recall, balancing both metrics.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Evaluate the model's ability to distinguish between classes across all decision thresholds.

Follow these steps to run the code, ensuring that you preprocess the data, extract features, train models, and evaluate performance using the metrics mentioned above.

