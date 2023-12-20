# LingoRank the development of model for english speakers that predicts the difficulty of a French written text
  ![Logo](/Logo.png)
## Repository Index
---
- [training_data.csv](/training_data.csv) Data for training provided by Kaggle competition
- [unlabelled_test_data.csv](/unlabelled_test_data.csv) The unlablled data for final prediction
- [frenchbacktranslated.csv](/frenchbacktranslated.csv) French back translate for data augmentation
- [trainingandgenfixed.csv](/trainingandgenfixed.csv) Data for training with additional generated 1634 rows 
- [streamlit.py](/streamlit.py) The Python file for presenting the user interface
- [xgboost.pkl](/xgboost.pkl) Model with the (one of...) best performance for classification, used in streamlit


## Video
--- Link to video --

## 1 Project overview

### 1.1 Problem description

Language learners often struggle to find texts that match their proficiency level. A beginner (A1) might be overwhelmed by advanced (B2) texts, impeding learning progress. The ideal learning material should contain mostly familiar words, with a few new terms for gradual improvement.

### 1.2 Overall objective

This project aims to develop a predictive model for English speakers learning French, designed to assess the difficulty level of French written texts. The model will categorize texts according to the Common European Framework of Reference for Languages (CEFR) levels, ranging from A1 (beginner) to C2 (advanced).

### 1.3 Thinking as a linguist 

In our initial efforts, we primarily used simple algorithms such as Logistic Regression, Decision Trees, and KNN, which only provided moderately successful results. Subsequently, we explored various online tools believed to be powered by machine learning, including Cathoven’s CEFR Checker (https://www.cathoven.com/en/cefr-checker/) and RoadToGrammar’s Text Analyzer (http://www.roadtogrammar.com/textanalysis/). These platforms revealed innovative text analysis techniques we had not previously considered. Initially, we assumed that the length of sentences and the total word count would significantly influence the CEFR level of a text. However, these two tools demonstrated the substantial impact these factors truly have. 

Through our analysis of these tools, we realized the necessity of transitioning to a more sophisticated model, that is less susceptible to uncontrollable variables. For instance, if our test data contains longer sentences typical of C1 level, but our training data is based on longer sentences from C2 level, there's a risk of biased assessments and inaccurate difficulty gradings. Although this issue can arise with any model, advanced models are better equipped to identify and evaluate recurring words and patterns, something beyond the capability of basic algorithms.


## 2 Data preparation

In response to the limited size of the original training set, comprising 4799 rows, we undertook an initiative to augment this dataset by incorporating an additional 1634 rows (with relatively equal amount of sentences for each difficulty level), synthesized through precise and strategic prompts with ChatGPT. This augmentation aimed at enhancing the overall dataset to a comprehensive total of 6433 rows. The purpose of this expansion was to refine the accuracy and effectiveness of our predictive model. However, upon implementation of this augmented dataset, we observed that the anticipated significant improvements in model performance did not happen. Consequently, in the later stages of the competition, we made a decision to stop our pursuit of data augmentation as a method for enhancing the model's accuracy. We have ultimately came to a realisation, that it is not the quantity of mimicked data that counts, but its quality.

### 3.3 (A) Processing

--- 

#### 3.3.1 Tokenization and Text Feature Extraction (Bert)

---

#### 3.3.2 Select Classification Model

 --- 
 
#### 3.3.3 Prediction on the test set

--- 

### 3.3 (B) Processing

#### 3.3.1 Embedding of Text Using OpenAI API
In this section, the method for converting text data into embeddings using OpenAI's API is described. The provided function get_embeddings_batch takes a batch of texts and returns their embeddings using the specified OpenAI model:

This function is crucial for transforming raw text into a format that the machine learning model can interpret and analyze.

#### 3.3.2 Selection of Classification Model
In our study, following the extraction of feature representations via OpenAI's API, we focused on training these features using a neural network model implemented with Keras, using ReLU  activation functions. 

To conduct this evaluation, the dataset was systematically divided into two distinct subsets: a training set and a test set. Specifically, the dataset was split such that X_train and y_train comprised the training set, encompassing the feature vectors and their corresponding difficulty labels. Conversely, X_test and y_test formed the test set, reserved for validating the model's performance. This split was configured with a test size parameter of 0.2, ensuring that 20% of the data was allocated for testing, while the remaining 80% was used for training. 

The model's architecture, consists of multiple dense layers with ReLU activation functions. However, the key to enhance its predictive accuracy was in fine-tuning its hyperparameters, especially the learning rate.
 
#### 3.3.3 Prediction and Evaluation on the Test Set

We managed to effectively extract predictions from our model by applying it to the test dataset. This process involved using the model's predict method on the embeddings derived from the test set sentences. The model outputted probabilities for each difficulty level, from which we extracted the highest probability class as the final prediction:

After obtaining these predictions, we appended them to the test dataset, providing a clear and accessible way to review the model's output.

In a Kaggle competition, we submitted our predictions to verify the model's performance against real-world standards. Over the course of the competition, we uploaded our submissions a total of 71 times. The highest score was 0.555, using the OpenAI text embeddings
![Снимок экрана 2023-12-20 в 16.36.06](/Image/Снимок экрана 2023-12-20 в 16.36.06.png)
