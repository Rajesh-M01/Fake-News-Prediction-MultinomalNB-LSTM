# Fake-News-Prediction-MultinomalNB-LSTM
A Naive Bayes(Multinomial NB)-LSTM-based hybrid model for fake news classification, leveraging probability scores from Naive Bayes as input features for an LSTM network to enhance accuracy. This project applies NLP techniques for text preprocessing and deep learning for improved fake news detection.

Dataset :
Used Fake News Dataset dataset(FakeNewsDataset.csv).
Data sourced from Kaggle.
Contains features of URL, Headline, Body and Label

Model Overview :
Preprocessing:
Tokenization & Lemmatization using NLTK  
Removing stopwords & special characters  
Vectorization using TF-IDF 

Architecture:
Naive Bayes Layer: Generates probability scores for fake/real news  
LSTM Layer: Captures deeper semantic patterns in text  
Dense Layers: Outputs final classification  
Loss Function: Categorical Crossentropy  
Optimizer: Adam 

Results:
Naïve Bayes Accuracy: 92.9825
LSTM Accuracy: 98.2456
Stacked Model Accuracy: 98.24561403508771

            
![image](https://github.com/user-attachments/assets/cde874ff-0ac3-4d36-9df6-8f6cc7a61882)


Sample Outputs - > 
1. news = "Economists Predict Steady Growth for the Global Economy in the Next Quarter"
print(prediction_input_processing(news))
output - > No, It is not fake

2. news = "Scientists Reveal: Gravity Is Just a Simulation Designed by Elon Musk!"
print(prediction_input_processing(news))
output -> Yes, It is fake

3. news = "Indian Government Announces New Infrastructure Plan to Boost National Economy"
print(prediction_input_processing(news))
output - > No, It is not fake

4. news = "Breaking: New study reveals that the Earth is and has always been flat and NASA and ISRO have been lying about space for decades!"
print(prediction_input_processing(news))
output -> Yes, It is fake




How to Run:
Clone the repository:
git clone https://github.com/Rajesh-M01/Fake-News-Prediction-MultinomalNB-LSTM.git

cd Fake-News-Prediction-MultinomalNB-LSTM
pip install tensorflow pandas numpy scikit-learn matplotlib  

Run the Jupyter Notebook (Fake_News_Detection_Hybrid_MultinomialNB_LSTM.ipynb) in Google Colab or locally.

Libraries Used :
TensorFlow / Keras
Scikit-Learn
NLTK
NumPy
Pandas
Matplotlib

Key Learnings
How to combine Naïve Bayes and LSTM for fake news classification using stacking.
Importance of text preprocessing (tokenization, stemming, and lemmatization) to improve model performance in NLP tasks.
How to handle imbalanced datasets and the significance of using evaluation metrics like accuracy, precision, recall, and F1-score. 
The role of hybrid models (Naive Bayes + LSTM) in improving accuracy over individual models (in this case the traditional Naive Bayes).


🚀 Connect with Me
If you liked this project, feel free to ⭐ the repository and connect with me on https://www.linkedin.com/in/rajesh-m-a42539317/ ! 🚀



