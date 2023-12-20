# LingoRank the development of model for english speakers that predicts the difficulty of a French written text.
  
## Repository Index
---
---
---

## Video
--- Link to video --

## 1 Project overview

### 1.1 Problem description

Language learners often struggle to find texts that match their proficiency level. A beginner (A1) might be overwhelmed by advanced (B2) texts, impeding learning progress. The ideal learning material should contain mostly familiar words, with a few new terms for gradual improvement.

### 1.2 Overall objective

This project aims to develop a predictive model for English speakers learning French, designed to assess the difficulty level of French written texts. The model will categorize texts according to the Common European Framework of Reference for Languages (CEFR) levels, ranging from A1 (beginner) to C2 (advanced).

## 2 Data preparation

In response to the limited size of the original training set, comprising 4799 rows, we undertook an initiative to augment this dataset by incorporating an additional 1634 rows (with relatively equal amount of sentences for each difficulty level), synthesized through precise and strategic prompts with ChatGPT. This augmentation aimed at enhancing the overall dataset to a comprehensive total of 6433 rows. The purpose of this expansion was to refine the accuracy and effectiveness of our predictive model. However, upon implementation of this augmented dataset, we observed that the anticipated significant improvements in model performance did not happen. Consequently, in the later stages of the competition, we made a decision to stop our pursuit of data augmentation as a method for enhancing the model's accuracy.

### 3.3 Processiong

--- 

#### 3.3.1 Tokenization and Text Feature Extraction (Bert)

---

#### 3.3.2 Select Classification Model

 --- 
 
#### 3.3.3 Prediction on the test set

--- 

