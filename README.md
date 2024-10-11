# ChildInternetUse_DNN_TeacherStudentModel
The goal of this project is to develop a predictive model that analyzes children's physical activity and fitness data to identify early signs of problematic internet use. Identifying these patterns can help trigger interventions to encourage healthier digital habits.

## Introduction

This project is motivated by a deep interest in the relationship between physical fitness and mental health in children, particularly as it relates to problematic internet usage. Recent statistics show a concerning rise in child obesity rates and unhealthy internet habits in the United States. By leveraging data-driven approaches, we can better understand the interplay between these variables and potentially inform strategies to improve child well-being.

This project is built around a dataset from a Kaggle competition focused on analyzing children’s mental health and internet use. The competition provided data from the Healthy Brain Network, a research initiative focused on identifying patterns that contribute to child and adolescent development. Through the Kaggle platform, the dataset became a valuable resource for machine learning practitioners and data scientists to explore and propose models that can predict problematic internet usage levels based on various demographic, physiological, and behavioral factors.
Dataset

The dataset used in this project is sourced from the Healthy Brain Network and covers a comprehensive range of variables, including demographic data, physical health measures, and internet usage patterns. The original dataset has over 4,000 participants aged 5-22 years and includes information gathered from various sources, such as:

    Basic Demographic Information: Age, sex, and enrollment season.
    CGAS (Children’s Global Assessment Scale): A numeric score to evaluate the functioning of children in different settings.
    Physical Health Measures: BMI, height, weight, heart rate, blood pressure, and waist circumference.
    Fitness and Endurance Measures: Max stage reached during fitness tests, endurance time (in minutes and seconds), and total endurance score.
    Bio-electrical Impedance Analysis (BIA): Various measures related to body composition, including fat-free mass, total body water, intracellular and extracellular water, and more.
    Physical Activity Questionnaire (PAQ): Total PAQ scores capturing the participant’s physical activity level.
    Internet Use Variables: Hours of computer or internet use per day, evaluated using responses from the Pre-Internet Education History survey.
    Problematic Internet Use Severity (Target Variable): The sii column captures the severity of problematic internet use with classes ranging from 0 to 3:
        0: No problematic use.
        1: Mild problematic use.
        2: Moderate problematic use.
        3: Severe problematic use.

The dataset exhibits class imbalance, which required the application of Synthetic Minority Over-sampling Technique (SMOTE) to ensure balanced representation across all severity classes. After preprocessing and feature engineering, the final dataset used for training included interaction terms, polynomial features, and multiple imputation techniques to handle missing values effectively.
Model Summary

This project employs a Student-Teacher Model Architecture to make predictions. The student-teacher model approach allows the student model to mimic the outputs of a larger, more complex teacher model. This technique is particularly useful when the student model needs to perform well under constraints, such as limited feature availability or computational resources.

## Teacher Model

The teacher model is a deep neural network (DNN) trained using all available features, including demographic, fitness, physical health, and internet use variables. The teacher model was designed to capture complex patterns in the data and generate high-quality probabilistic predictions (soft labels) for the student model to learn from. The teacher model’s architecture includes:

    Layers: Multiple fully connected layers with ReLU activation functions and dropout for regularization.
    Loss Function: Standard Categorical Cross-Entropy with Quadratic Weighted Kappa as an additional evaluation metric.
    Optimizer: Adam with a learning rate scheduler to dynamically adjust the learning rate based on validation performance.

## Student Model

The student model is a distilled version of the teacher model, trained using a subset of the original features due to constraints in the testing dataset. The selected features include demographic data, some physical health measures, and internet usage patterns. The student model is trained to not only optimize for the true labels but also to match the output distribution of the teacher model through a custom loss function that combines Categorical Cross-Entropy and KL Divergence.

    Layers: Multiple dense layers with LeakyReLU and ELU activation functions to improve the model’s capacity to capture non-linear relationships.
    Loss Function: Custom loss combining Categorical Cross-Entropy and KL Divergence, weighted equally.
    Optimizer: Adam with an initial learning rate of 0.001.
    Regularization: Dropout and L2 regularization to prevent overfitting.

Training and Validation

The teacher model was trained using the full feature set, while the student model used only a subset of 10 features. The training was performed using the following configurations:

    Batch Size: 64
    Number of Epochs: 50
    Early Stopping: Enabled with a patience of 6 epochs.

Results
Teacher Model Performance

    Accuracy: ~71%
    F1-Score: ~0.69 (weighted average)
    Quadratic Weighted Kappa (QWK): 0.78

Student Model Performance

The student model, despite being trained with a limited feature set, achieved comparable performance to the teacher model:

    Accuracy: ~62.45%
    F1-Score: 0.6067 (weighted average)
    Quadratic Weighted Kappa (QWK): 0.7166

The student model shows high precision and recall for the "No Problematic Use" class (class 0) and reasonably balanced performance across the other classes. The model's high QWK value indicates that it effectively captures the ordinal nature of the problematic internet usage levels.
How to Reproduce the Results

## Clone the Repository:

bash

git clone https://github.com/your-username/ChildInternetUse_DNN_TeacherStudentModel.git
cd ChildInternetUse_DNN_TeacherStudentModel

Install Dependencies: Install the required libraries by running:

bash

pip install -r requirements.txt

Data Preparation: Download the dataset from the Kaggle competition page Healthy Brain Network Kaggle Dataset and place it in the data folder.

Training: Run the following command to train the teacher and student models:

bash

python train_models.py

Evaluation: To evaluate the student model on the test dataset, use:

bash

    python evaluate_student_model.py

    Submission: The predicted outputs for the test dataset will be saved as sample_submission.csv. Use this file to submit to the Kaggle competition.

## Conclusion

This project demonstrates the effectiveness of a student-teacher model architecture in scenarios where feature constraints exist. Despite being trained on a limited set of features, the student model managed to retain much of the performance of the teacher model, highlighting the power of knowledge distillation and model compression.

For any questions or discussions, feel free to reach out through GitHub Issues or Fork this repository to contribute further!
