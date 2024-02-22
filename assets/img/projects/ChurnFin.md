<h1 style="text-align:center; color:grey; font-size:40px;"> Demystifying Customer Fallout: A Comparative Evaluation of Algorithmic Predictors (Vanilla, Ensemble, and Neural Network Models) </h1>

<h4 style="color:grey; text-align:center;">Vaishnavi Shankar Devadig, 2023</h4>

# Logistic Regression:

<p style="text-align:justify">It is a linear classification algorithm that models the probability of an event (churn, in our case) occurring based on independent variables (customer attributes). It utilizes a logistic function to estimate the probability between 0 and 1, making it well-suited for binary classification problems like churn prediction.</p>

<p style="text-align:justify">Well, do not underestimate the power of its simplicity! While not the flashiest tool in the shed, logistic regression's interpretability and reliability make it a handy algorithm for data scientists. Let's say it is basically he swiss army knife of classification algorithms!</p>
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

![output_1_1.png](7f0d251a-43a0-4a69-84d9-d279086cee48.png)

![image.png](3c8831dd-d127-4220-a4e3-412c2254565f.png)

<p style="text-align:justify">Precision, recall, and F1-score are reported for both classes (0 and 1), with an overall accuracy of 0.77. 
 Precision for class 0 (non-churned) is 0.86, indicating a high rate of correctly predicting non-churned customers. 
 Class 1 (churned) has a precision of 0.69. </p>
 
<p style="text-align:justify"> The recall for class 0 is 0.90, and for class 1 is 0.60, indicating the model's ability to find all positive samples.</p>

<p style="text-align:justify">However, the model does not do a great job picking possible churners, which is the main goal of the task.</p>

#### Calibration Plot

![output_10_0.png](2b26afef-9e8b-420a-8cda-110ab5f6d57c.png)
brier_score_lr = brier_score_loss(y_test, y_pred)
print(brier_score_lr)
0.17885024840312277

# Random Forest:

<p style="text-align:justify">
Random forest is an ensemble learning method that combines multiple decision trees trained on randomly sampled data subsets. Each tree independently predicts churn, and the final prediction is made by aggregating the majority vote from all trees. This randomness helps prevent overfitting and improves model robustness. </p>

<p style="text-align:justify">Imagine a room full of detectives, each with their own belief about customer churn based on different clues. Random forest is like bringing them all together, letting them share their findings, and voting on the most likely culprit. However, we need to keep in mind that too many cooks can spoil the broth, so we have to carefully tune the number of trees to avoid overcomplexity and overfitting.</p>

![output_14_0.png](2db2b0ef-c82c-4f29-9bb5-a650aad8534f.png)

#### Decision Tree with Random Forest

![output_14_1.png](5da8fd37-b84d-4f2f-a3cf-3c0bee99da3d.png)
y_pred_rf = model_rf.predict(X_test)
report1 = classification_report(y_test, y_pred_rf)
conf_matrix1 = confusion_matrix(y_test, y_pred_rf)
print(report1)
![image.png](a02b88c4-7c65-466d-a2ac-de31ad015ef4.png)

<p style="text-align:justify">ROC Curve and AUC: The Random Forest model achieved an AUC of approximately 0.81, indicating a strong ability to distinguish between churned and retained customers.</p>

<p style="text-align:justify">Classification Report: The model's precision for classifying non-churned customers (class 0) was 0.82, and for churned customers (class 1), it was 0.63. The overall accuracy of the model was 0.78, with a recall of 0.91 for non-churned and 0.44 for churned customers, suggesting that while the model is quite good at identifying non-churned customers, it could improve in correctly identifying churned customers.</p>

#### Calibration Plot

![output_22_0.png](1ee9b6b6-d89e-4614-ba7b-d4c96dc69081.png)

#### Calibrating the model with Isotonic Regression and Cross Validation

![output_24_1.png](5a00d661-63df-458e-a689-65e67efd5ed2.png)

Brier score before calibration: 0.1499402767920511

Brier score after calibration: 0.13912276125211268

Brier Score reduced very mildly after calibration, indicating that the model did not change much.

#### Random Forest after Hyperparameter Tuning:

Best parameters found by grid search: {'classifier__max_depth': 8, 'classifier__max_features': 'auto', 'classifier__min_samples_leaf': 1, 'classifier__min_samples_split': 10, 'classifier__n_estimators': 100}
Best score: 0.8024493263209989

![Screenshot 2024-02-20 at 3.21.44 PM.png](36b2e1f8-c6de-4567-b170-7387c72449a6.png)

![output_38_0.png](c03bb87b-7548-4ed0-a964-0fa49d96ad26.png)

Brier Score: 0.12878265899294405

The Brier Score has reduced even more to 0.129 now, indicating that after hyperparameter tuning, the model has gotten comparatively better than it was earlier.

# Gradient Boosting Machine

<p style="text-align:justify">Gradient Boosting is an ensemble method building iteratively by learning from the mistakes of previous models in the sequence. Each new model focuses on areas where previous models incorrectly predicted churn, leading to a progressively more accurate ensemble. This approach is mostly meant for complex prediction problems.</p>

![output_43_0.png](0e2ed3c6-ca1a-4fde-8244-34343b60c2a9.png)

![image.png](fb7b9d12-ab6d-4cad-92e8-2528a0755988.png)

<p style="text-align:justify">The Gradient Boosting model achieved an AUC of approximately 0.86, thus a strong ability to distinguish between churned and retained customers.</p>

<p style="text-align:justify">Classification Report: The model's precision for classifying non-churned customers (class 0) was 0.84, and for churned customers (class 1), it was 0.67. The overall accuracy of the model was 0.81, with a recall of 0.91 for non-churned and 0.53 for churned customers, suggesting that while the model is quite good at identifying non-churned customers, it could improve in correctly identifying churned customers.</p>

#### Calibration Plot

![output_48_0.png](7b92b733-b68b-44f1-b284-be8e4a255e4e.png)

Brier Score: 0.12878756966393143

This is one of the lower Brier Scores in comparison, indicating this could be a good choice for the modeling.

# Support Vector Machine

<p style="text-align:justify">SVM is a discriminative classifier that aims to find a hyperplane that maximizes the margin between churned and loyal customers in a high-dimensional space. This margin represents the confidence in the separation, making SVM robust to outliers and noise in the data.</p>

<p style="text-align:justify">While effective, we have to remember that a rigid general might struggle to adapt to unexpected changes in the data, so its use is limited for highly dynamic data.</p>


![output_51_0.png](0be62c2d-110e-4b6a-ba5b-0b0e439818b6.png)

![image.png](9ee2cddc-b401-4bcf-ad38-6d53300bd1c6.png)

The results of support vector machine are not as decent as the other models. This is understood, given the fact that it can separate datapoints by means of only a place.

#### Calibration Plot

![output_55_0.png](56a9165f-5e38-4edc-b673-c620770ee11f.png)

Brier Score: 0.19020581973030518

The Brier Score is pretty high compared to the rest, eliminating the model for this use case.

# Multi Layer Perceptron Classifier

<p style="text-align:justify">This algorithm utilizes an artificial neural network architecture with multiple interconnected layers of nodes. Each layer transforms the data representation, extracting increasingly complex features relevant to churn prediction. This allows for highly non-linear relationships between features and churn, making it suitable for complex data. However, it is computationally expensive.</p>

![output_60_0.png](0468bc0f-e404-47cd-b46c-2bd22076839e.png)

The training stopped after 19 iterations.

![image.png](bea046a6-511b-4054-87dd-707941c5f748.png)

#### Calibration Plot

![output_61_1.png](98da035a-b957-442c-8dd2-5751e0a97a2f.png)

#### Learning Curve:

![output_62_0.png](427a9606-eb33-4707-8e1e-7867c0a66c89.png)

### Final Results

To determine the best model for predicting customer churn, let us compare the models based on the results, focusing on several key metrics:

- <p style="text-align:justify">AUC: This measures the model's ability to discriminate between the positive (churners) and negative (non-churners) classes. Higher values indicate better model performance.</p>

- <p style="text-align:justify">Precision, Recall, and F1-Score: Precision measures the accuracy of positive predictions. Recall (sensitivity) measures the ability of the model to find all the positive instances. F1-score provides a balance between precision and recall. In the context of churn prediction, precision would reflect how accurately the model identifies churners, while recall indicates the model's ability to capture the majority of actual churners.</p>

- <p style="text-align:justify">Accuracy: This reflects the overall correctness of the model across both classes.</p>
- <p style="text-align:justify">Brier Score: It quantifies the probability predictions' proximity to the actual outcomes. Lower values indicate better calibration.</p>

- <p style="text-align:justify">Business Objective: The primary goal is to identify potential churners accurately to take preventative actions. Therefore, a model that offers a good balance between recall (to catch as many churners as possible) and precision (to avoid misclassifying non-churners as churners, which could waste resources) would be ideal.</p>



### Model Comparison

<p style="text-align:justify"><b>Logistic Regression</b> and <b>MLPClassifier</b> have an AUC of 0.86, an accuracy of 0.82, and similar precision, recall, and F1-scores for identifying churners. Their performance metrics are closely matched, but the Brier Score for the MLPClassifier (0.1294) is better than that for Logistic Regression (0.1788), indicating the MLPClassifier's probabilistic predictions are more accurate.</p>

<p style="text-align:justify"><b>Random Forest</b> initially showed lower performance metrics across the board than Logistic Regression and MLPClassifier. However, after calibration and hyperparameter tuning, it achieved a Brier Score of 0.1288, which is slightly better than MLPClassifier and significantly better than its uncalibrated version. Its best accuracy post-tuning is 0.81, slightly lower than Logistic Regression and MLPClassifier.</p>

<p style="text-align:justify"><b>Gradient Boosting Model</b> also has an AUC of 0.86, with accuracy, precision, recall, and F1 scores similar to MLPClassifier's. Its Brier Score (0.1288) is comparable to the calibrated Random Forest model, suggesting <b>good probability prediction accuracy</b>.</p>

<p style="text-align:justify"><b>Support Vector Machine</b> has the lowest Brier Score (0.1902) performance than the other models, despite having a decent AUC of 0.81 and accuracy similar to the other models.</p>

### Best Model Selection:

<p style="text-align:justify">Given the business objective of accurately identifying potential churners to take preventative actions, the MLPClassifier or Gradient Boosting Model appears to be the best model for several reasons:</p>

- <p style="text-align:justify">Balanced Performance: They offer a good balance between precision and recall. This balance is crucial for accurately identifying churners (high recall) while minimizing the risk of false alarms (high precision).</p>

- <p style="text-align:justify">Brier Score: Their Brier Score is among the lowest, suggesting their probability predictions are more accurate. This is important for making informed decisions about customers to target with retention strategies.</p>

- <p style="text-align:justify">AUC: Their AUC is tied for the highest among the models, indicating strong discriminative power between churners and non-churners.</p>

<p style="text-align:justify">Choosing the best model could also depend on other factors not covered by the metrics alone, such as model interpretability, computational efficiency, and ease of deployment. Hence, for smaller-scale applications, the Gradient Boosting Machine would be a better choice.</p>


```python

```
