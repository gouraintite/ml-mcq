"use client";
import { useState, useEffect, useRef } from "react";

const allQuestions = [
  // ═══════════════════════════════════════════════════
  // CHAPTER 1: LINEAR REGRESSION (25 questions)
  // ═══════════════════════════════════════════════════
  {id:1,ch:"Ch1: Linear Regression",q:"In linear regression, the dependent variable (Y) is also called:",opts:["Predictor","Feature","Response / Outcome","Independent variable"],ans:2,exp:"Y is the response/outcome we want to predict. X variables are the predictors/features."},
  {id:2,ch:"Ch1: Linear Regression",q:"What is the equation of Simple Linear Regression?",opts:["Y = β₀ + β₁X₁ + β₂X₂","Y = β₀ + β₁X","Y = β₀ × β₁X","Y = β₁X"],ans:1,exp:"Simple LR has one predictor: Y = β₀ (intercept) + β₁ (slope) × X."},
  {id:3,ch:"Ch1: Linear Regression",q:"Multiple Linear Regression differs from Simple LR because it has:",opts:["A non-linear function","≥ 2 independent variables","No intercept term","A logistic loss function"],ans:1,exp:"Multiple LR: Y = β₀ + β₁X₁ + β₂X₂ + ... + βₚXₚ with p ≥ 2 predictors."},
  {id:4,ch:"Ch1: Linear Regression",q:"The MSE cost function J(β) = (1/n)Σ(Yᵢ − Ŷᵢ)² measures:",opts:["Sum of all predictions","Avg squared difference between actual & predicted","Correlation between X and Y","Number of features"],ans:1,exp:"MSE averages squared residuals — quantifies how far predictions are from actuals."},
  {id:5,ch:"Ch1: Linear Regression",q:"In the gradient descent update rule βⱼ = βⱼ − α(∂J/∂βⱼ), α represents:",opts:["The cost function","The number of features","The learning rate (step size)","The intercept"],ans:2,exp:"α controls step size. Too small → slow; too large → divergence."},
  {id:6,ch:"Ch1: Linear Regression",q:"The OLS closed-form solution β = (XᵀX)⁻¹XᵀY fails when:",opts:["The dataset is small","XᵀX is not invertible (multicollinearity)","Y has outliers","There is only one feature"],ans:1,exp:"OLS needs XᵀX invertible. Multicollinearity makes it singular."},
  {id:7,ch:"Ch1: Linear Regression",q:"R² = 1 − RSS/TSS. If R² = 0.85 this means:",opts:["85% of variance in Y is explained by X","The model is 85% accurate","15% of variance is explained","MSE = 0.85"],ans:0,exp:"R² measures proportion of variance explained. 0.85 = 85% explained."},
  {id:8,ch:"Ch1: Linear Regression",q:"Adjusted R² is preferred over R² because it:",opts:["Is always higher than R²","Penalizes adding irrelevant predictors","Is faster to compute","Works only for simple regression"],ans:1,exp:"R² never decreases when adding features. Adjusted R² penalizes useless predictors."},
  {id:9,ch:"Ch1: Linear Regression",q:"Which is NOT an assumption of Multiple Linear Regression?",opts:["Linear relationship","Independence of errors","Normality of the input features X","Homoscedasticity"],ans:2,exp:"Normality applies to RESIDUALS, not input features X."},
  {id:10,ch:"Ch1: Linear Regression",q:"Homoscedasticity means:",opts:["Errors follow a normal distribution","Residuals have constant variance","Features are uncorrelated","The model is linear"],ans:1,exp:"Constant variance of residuals across all levels of predicted values."},
  {id:11,ch:"Ch1: Linear Regression",q:"Given: Salary = −10 + 4×YearsExp + 1.5×Age. Predict salary for 5 years exp, age 30:",opts:["45","55","35","25"],ans:1,exp:"−10 + 4(5) + 1.5(30) = −10 + 20 + 45 = 55."},
  {id:12,ch:"Ch1: Linear Regression",q:"If correlation between two predictors is r = 0.99, what is the VIF?",opts:["1.01","10.05","≈ 50.25","99"],ans:2,exp:"R² = 0.99² = 0.9801. VIF = 1/(1−0.9801) = 1/0.0199 ≈ 50.25."},
  {id:13,ch:"Ch1: Linear Regression",q:"A model has RSS = 200, TSS = 1000. What is R²?",opts:["0.20","0.80","0.50","5.00"],ans:1,exp:"R² = 1 − 200/1000 = 0.80."},
  {id:14,ch:"Ch1: Linear Regression",q:"If learning rate α is too large during gradient descent:",opts:["Very slow convergence","Cost function diverges (overshoots minimum)","Model underfits","OLS is used instead"],ans:1,exp:"Too large α → updates overshoot → J(β) increases → divergence."},
  {id:15,ch:"Ch1: Linear Regression",q:"VIF = 1 for a predictor means:",opts:["Perfect collinearity","Severe multicollinearity","No collinearity with other predictors","Predictor should be removed"],ans:2,exp:"VIF=1 → R²ⱼ=0 in auxiliary regression → no linear dependence with others."},
  {id:16,ch:"Ch1: Linear Regression",q:"Which method gives an EXACT (analytic) solution for linear regression?",opts:["Gradient Descent","OLS (Ordinary Least Squares)","SGD","Adam optimizer"],ans:1,exp:"OLS: β = (XᵀX)⁻¹XᵀY is exact. GD is iterative/approximate."},
  {id:17,ch:"Ch1: Linear Regression",q:"For n > 10⁶ samples, which method is more practical?",opts:["OLS","Gradient Descent","Both equally fast","Neither works"],ans:1,exp:"OLS needs matrix inversion (slow for large n). GD is scalable."},
  {id:18,ch:"Ch1: Linear Regression",q:"Which test checks independence of residuals?",opts:["Shapiro-Wilk","Durbin-Watson","VIF","QQ plot"],ans:1,exp:"Durbin-Watson tests autocorrelation. Shapiro-Wilk → normality. VIF → multicollinearity."},
  {id:19,ch:"Ch1: Linear Regression",q:"VIF > 10 suggests:",opts:["Predictor is very important","Strong/severe multicollinearity","Model is overfitting","Residuals not normal"],ans:1,exp:"VIF > 10 = severe multicollinearity. Solution: remove predictor, PCA, or combine variables."},
  {id:20,ch:"Ch1: Linear Regression",q:"In sklearn, the correct import for linear regression is:",opts:["from sklearn.linear_model import LinearRegression","from sklearn.regression import LinearModel","from sklearn.models import LinReg","from sklearn.svm import LinearRegression"],ans:0,exp:"sklearn.linear_model contains LinearRegression, LogisticRegression, etc."},
  {id:21,ch:"Ch1: Linear Regression",q:"train_test_split(X, y, test_size=0.2) returns:",opts:["Two arrays: X_train, X_test","Four arrays: X_train, X_test, y_train, y_test","A single shuffled dataset","The model score"],ans:1,exp:"Returns 4 arrays with 80/20 split."},
  {id:22,ch:"Ch1: Linear Regression",q:"To standardize features (mean=0, std=1), use:",opts:["Normalizer","MinMaxScaler","StandardScaler","LabelEncoder"],ans:2,exp:"StandardScaler from sklearn.preprocessing transforms to mean=0, std=1."},
  {id:23,ch:"Ch1: Linear Regression",q:"After reg.fit(X_train, y_train), predictions on new data use:",opts:["reg.transform(X_test)","reg.predict(X_test)","reg.score(X_test)","reg.classify(X_test)"],ans:1,exp:".predict() gives predictions. .score() returns R²."},
  {id:24,ch:"Ch1: Linear Regression",q:"The partial derivative ∂J/∂βⱼ = −(2/n)Σ Xᵢⱼ(Yᵢ − Ŷᵢ) is used in:",opts:["OLS closed-form solution","Gradient Descent weight update","R² calculation","VIF computation"],ans:1,exp:"This partial derivative is used in GD: βⱼ = βⱼ − α × (∂J/∂βⱼ)."},
  {id:25,ch:"Ch1: Linear Regression",q:"RSE = √[(1/(n−2))Σ(yᵢ−ŷᵢ)²]. A smaller RSE means:",opts:["Worse fit","Better fit","More features needed","Higher bias"],ans:1,exp:"Smaller RSE → predictions closer to actual values → better fit."},

  // ═══════════════════════════════════════════════════
  // CHAPTER 2: LOGISTIC REGRESSION & KNN (30 questions)
  // ═══════════════════════════════════════════════════
  {id:26,ch:"Ch2: Logistic Regression & KNN",q:"Logistic regression is used for:",opts:["Predicting continuous values","Binary/multi-class classification","Clustering","Dimensionality reduction"],ans:1,exp:"Logistic regression predicts the probability that an input belongs to a class."},
  {id:27,ch:"Ch2: Logistic Regression & KNN",q:"The sigmoid function σ(z) = 1/(1+e⁻ᶻ) outputs values in the range:",opts:["(−∞, +∞)","[0, 1]","(0, 1)","[−1, 1]"],ans:2,exp:"Sigmoid maps any real number to (0, 1), never exactly 0 or 1."},
  {id:28,ch:"Ch2: Logistic Regression & KNN",q:"At z = 0, the sigmoid function outputs:",opts:["0","1","0.5","−1"],ans:2,exp:"σ(0) = 1/(1+e⁰) = 1/(1+1) = 0.5 → maximum uncertainty."},
  {id:29,ch:"Ch2: Logistic Regression & KNN",q:"The decision boundary in logistic regression is drawn at:",opts:["ŷ = 0","ŷ = 1","ŷ = 0.5","ŷ = 0.7"],ans:2,exp:"If ŷ > 0.5 → class 1, otherwise class 0. The boundary is linear in feature space."},
  {id:30,ch:"Ch2: Logistic Regression & KNN",q:"The loss function for binary logistic regression is:",opts:["MSE","Binary Cross-Entropy (BCE)","Hinge Loss","Gini Index"],ans:1,exp:"BCE: Loss = −(1/n)Σ[yᵢlog(ŷᵢ) + (1−yᵢ)log(1−ŷᵢ)]"},
  {id:31,ch:"Ch2: Logistic Regression & KNN",q:"In BCE loss, when y=1 and ŷ is close to 1, the loss is:",opts:["Very high","Close to 0","Exactly 1","Undefined"],ans:1,exp:"−log(ŷ) ≈ −log(1) = 0 when ŷ ≈ 1 → low loss (correct confident prediction)."},
  {id:32,ch:"Ch2: Logistic Regression & KNN",q:"The logistic regression algorithm steps in order are:",opts:["Init params → Forward pass (sigmoid) → Compute loss → Backward pass → Update → Repeat","Forward pass → Init → Loss → Update","Loss → Forward → Backward → Init","Init → Loss → Sigmoid → Predict"],ans:0,exp:"Initialize β, apply sigmoid, compute BCE loss, compute gradients, update with GD, repeat until convergence."},
  {id:33,ch:"Ch2: Logistic Regression & KNN",q:"The gradient of the BCE loss w.r.t. βⱼ is: (1/n)Σ(ŷᵢ − yᵢ)xᵢⱼ. This is used for:",opts:["Prediction","Weight update via gradient descent","Feature selection","Data normalization"],ans:1,exp:"βⱼ := βⱼ − α × (1/n)Σ(ŷᵢ − yᵢ)xᵢⱼ."},
  {id:34,ch:"Ch2: Logistic Regression & KNN",q:"In a confusion matrix, a False Positive (FP) means:",opts:["Predicted positive, actually positive","Predicted positive, actually negative","Predicted negative, actually positive","Predicted negative, actually negative"],ans:1,exp:"FP: model predicted positive but ground truth is negative (Type I error)."},
  {id:35,ch:"Ch2: Logistic Regression & KNN",q:"Precision is calculated as:",opts:["TP / (TP + FN)","TP / (TP + FP)","(TP + TN) / n","TN / (FP + TN)"],ans:1,exp:"Precision = TP/(TP+FP) — of all positive predictions, how many were correct?"},
  {id:36,ch:"Ch2: Logistic Regression & KNN",q:"Recall (Sensitivity) is calculated as:",opts:["TP / (TP + FP)","TP / (TP + FN)","TN / (FP + TN)","(TP + TN) / n"],ans:1,exp:"Recall = TP/(TP+FN) — of all actual positives, how many did we catch?"},
  {id:37,ch:"Ch2: Logistic Regression & KNN",q:"Specificity is:",opts:["TP / (TP + FN)","TN / (FP + TN)","TP / (TP + FP)","FP / (FP + TN)"],ans:1,exp:"Specificity = TN/(FP+TN) — true negative rate."},
  {id:38,ch:"Ch2: Logistic Regression & KNN",q:"A confusion matrix has TP=40, TN=50, FP=5, FN=5. What is accuracy?",opts:["80%","90%","95%","85%"],ans:1,exp:"Accuracy = (40+50)/(40+50+5+5) = 90/100 = 90%."},
  {id:39,ch:"Ch2: Logistic Regression & KNN",q:"With TP=40, FP=5, FN=5: what is Precision?",opts:["40/45 ≈ 0.889","40/50 = 0.80","45/50 = 0.90","40/100 = 0.40"],ans:0,exp:"Precision = 40/(40+5) = 40/45 ≈ 0.889."},
  {id:40,ch:"Ch2: Logistic Regression & KNN",q:"With TP=40, FP=5, FN=5: what is Recall?",opts:["40/45 ≈ 0.889","40/50 = 0.80","45/50 = 0.90","5/45 = 0.111"],ans:0,exp:"Recall = 40/(40+5) = 40/45 ≈ 0.889."},
  {id:41,ch:"Ch2: Logistic Regression & KNN",q:"For a 5-class One-vs-All (OvA) classification, how many binary classifiers are trained?",opts:["5","10","25","1"],ans:0,exp:"One-vs-All: N classes → N binary classifiers."},
  {id:42,ch:"Ch2: Logistic Regression & KNN",q:"For 5-class One-vs-One classification, how many binary classifiers?",opts:["5","10","25","15"],ans:1,exp:"One-vs-One: N(N−1)/2 = 5×4/2 = 10 classifiers."},
  {id:43,ch:"Ch2: Logistic Regression & KNN",q:"In sklearn, the correct code to train logistic regression is:",opts:["LogisticRegression().fit(X_train, y_train)","LogisticRegressor().train(X, y)","LogReg().learn(X_train, y_train)","LinearRegression().fit(X_train, y_train)"],ans:0,exp:"from sklearn.linear_model import LogisticRegression; logreg.fit(X_train, y_train)."},
  {id:44,ch:"Ch2: Logistic Regression & KNN",q:"For multinomial logistic regression in sklearn, which parameter must be set?",opts:["multi_class='multinomial'","type='multi'","classes=3","mode='softmax'"],ans:0,exp:"LogisticRegression(multi_class='multinomial', solver='lbfgs')."},
  {id:45,ch:"Ch2: Logistic Regression & KNN",q:"stratify=y in train_test_split ensures:",opts:["Random shuffling","Same class proportions in train and test sets","No duplicates","Alphabetical sorting"],ans:1,exp:"Stratification preserves class distribution — critical for imbalanced datasets."},
  {id:46,ch:"Ch2: Logistic Regression & KNN",q:"To get predicted probabilities (not just labels) in sklearn:",opts:["model.predict(X)","model.predict_proba(X)","model.score(X)","model.probability(X)"],ans:1,exp:".predict_proba() returns probability estimates for each class."},
  {id:47,ch:"Ch2: Logistic Regression & KNN",q:"Lowering the classification threshold from 0.5 to 0.3 will generally:",opts:["Increase precision, decrease recall","Increase recall, decrease precision","Not change anything","Increase both precision and recall"],ans:1,exp:"Lower threshold → more positives → catch more TP (higher recall) but also more FP (lower precision)."},
  {id:48,ch:"Ch2: Logistic Regression & KNN",q:"KNN is a:",opts:["Parametric method","Non-parametric method","Unsupervised method","Generative model"],ans:1,exp:"KNN makes no assumptions about the form of f — it's instance-based/non-parametric."},
  {id:49,ch:"Ch2: Logistic Regression & KNN",q:"In KNN classification, the predicted class is determined by:",opts:["Weighted average of neighbors","Majority vote of k nearest neighbors","Gradient descent","Decision boundary"],ans:1,exp:"For classification: assign the most frequent class among k nearest neighbors."},
  {id:50,ch:"Ch2: Logistic Regression & KNN",q:"A disadvantage of KNN is:",opts:["Too simple to implement","Heavy calculations for large n or p","Cannot handle classification","Requires gradient descent"],ans:1,exp:"KNN computes distances to ALL training samples → O(n×p) per prediction."},
  {id:51,ch:"Ch2: Logistic Regression & KNN",q:"Why is feature scaling important for KNN?",opts:["It speeds up gradient descent","Distance metrics are sensitive to feature scales","It reduces the number of neighbors","It removes outliers"],ans:1,exp:"KNN uses distances. Unscaled features with larger ranges dominate the distance calculation."},
  {id:52,ch:"Ch2: Logistic Regression & KNN",q:"Which sklearn class is used for KNN classification?",opts:["KNeighborsClassifier","KNNClassifier","NearestNeighbors","KMeans"],ans:0,exp:"from sklearn.neighbors import KNeighborsClassifier."},
  {id:53,ch:"Ch2: Logistic Regression & KNN",q:"An advantage of logistic regression over KNN is:",opts:["No hyperparameters to tune","Works only with images","Requires more computation","Cannot provide probabilities"],ans:0,exp:"Logistic regression is simple, interpretable, and has no major hyperparameters (vs k in KNN)."},
  {id:54,ch:"Ch2: Logistic Regression & KNN",q:"To import classification metrics in sklearn:",opts:["from sklearn.metrics import accuracy_score, confusion_matrix","from sklearn.model import metrics","import sklearn.classification","from sklearn import accuracy"],ans:0,exp:"sklearn.metrics contains accuracy_score, precision_score, recall_score, confusion_matrix, etc."},
  {id:55,ch:"Ch2: Logistic Regression & KNN",q:"A logistic regression prediction of 0.95 means:",opts:["95% accuracy","Model is 95% confident input belongs to class 1","95 data points classified","Error rate is 95%"],ans:1,exp:"The sigmoid output represents probabilistic confidence for class 1."},

  // ═══════════════════════════════════════════════════
  // CHAPTER 3: DECISION TREES & RANDOM FORESTS (35 questions)
  // ═══════════════════════════════════════════════════
  {id:56,ch:"Ch3: Decision Trees & Random Forests",q:"Decision trees segment the predictor space into:",opts:["Continuous functions","A number of simple regions","A single hyperplane","Probability distributions"],ans:1,exp:"Trees stratify the predictor space into regions R₁, R₂, ..., Rⱼ."},
  {id:57,ch:"Ch3: Decision Trees & Random Forests",q:"Decision trees can be used for:",opts:["Regression only","Classification only","Both regression and classification","Clustering only"],ans:2,exp:"Trees work for both: regression trees predict means, classification trees predict classes."},
  {id:58,ch:"Ch3: Decision Trees & Random Forests",q:"In a classification tree, the Gini Index formula is:",opts:["G = Σ p̂ⱼₖ(1 − p̂ⱼₖ)","G = −Σ p̂ⱼₖ log(p̂ⱼₖ)","G = Σ(yᵢ − ŷᵢ)²","G = TP / (TP + FP)"],ans:0,exp:"Gini measures impurity: G = Σₖ p̂ⱼₖ(1 − p̂ⱼₖ). Small when p̂ⱼₖ close to 0 or 1 (pure node)."},
  {id:59,ch:"Ch3: Decision Trees & Random Forests",q:"Cross-Entropy for a node is: D = −Σ p̂ⱼₖ log(p̂ⱼₖ). It is near zero when:",opts:["All classes are equally represented","The node is pure (one class dominates)","The tree is deep","Features are normalized"],ans:1,exp:"Like Gini, entropy → 0 when p̂ⱼₖ ≈ 0 or 1 (pure node)."},
  {id:60,ch:"Ch3: Decision Trees & Random Forests",q:"A node with 50% class A and 50% class B has a Gini index of:",opts:["0","0.25","0.50","1.0"],ans:2,exp:"Gini = 0.5×(1−0.5) + 0.5×(1−0.5) = 0.25 + 0.25 = 0.50 (maximum impurity for 2 classes)."},
  {id:61,ch:"Ch3: Decision Trees & Random Forests",q:"A pure node (100% one class) has Gini index:",opts:["0","0.5","1.0","Undefined"],ans:0,exp:"Gini = 1×(1−1) = 0. Zero impurity = pure node."},
  {id:62,ch:"Ch3: Decision Trees & Random Forests",q:"For binary entropy: if p = 0.5, entropy = −0.5 log₂(0.5) − 0.5 log₂(0.5) =",opts:["0","0.5","1.0","2.0"],ans:2,exp:"−0.5×(−1) − 0.5×(−1) = 0.5 + 0.5 = 1.0 bit (maximum entropy for binary)."},
  {id:63,ch:"Ch3: Decision Trees & Random Forests",q:"When making a split in a classification tree, we want to:",opts:["Maximize the Gini index","Minimize the Gini index or cross-entropy","Maximize the entropy","Minimize the number of features"],ans:1,exp:"A good split reduces impurity → minimize Gini or entropy in child nodes."},
  {id:64,ch:"Ch3: Decision Trees & Random Forests",q:"In decision tree terminology, 'terminal nodes' are also called:",opts:["Roots","Internal nodes","Leaves","Branches"],ans:2,exp:"Terminal nodes = leaves. They contain the final predictions."},
  {id:65,ch:"Ch3: Decision Trees & Random Forests",q:"For a classification tree, the prediction in a leaf node is:",opts:["The mean of Y values","The most common class (majority vote)","The median","A weighted sum"],ans:1,exp:"Classification: majority vote. Regression: mean of Y in that region."},
  {id:66,ch:"Ch3: Decision Trees & Random Forests",q:"Pruning a decision tree is done to:",opts:["Make the tree deeper","Prevent overfitting","Add more features","Increase training accuracy"],ans:1,exp:"Pruning removes branches to get a simpler tree → reduces overfitting."},
  {id:67,ch:"Ch3: Decision Trees & Random Forests",q:"Cost complexity pruning uses a parameter α that controls:",opts:["Learning rate","Trade-off between tree complexity and fit","Number of features","Batch size"],ans:1,exp:"α balances subtree size vs. training error. Chosen via cross-validation."},
  {id:68,ch:"Ch3: Decision Trees & Random Forests",q:"A key advantage of decision trees is:",opts:["High prediction accuracy","Easy interpretability","No overfitting risk","Built-in feature scaling"],ans:1,exp:"Trees are intuitive and easy to explain (even to non-experts)."},
  {id:69,ch:"Ch3: Decision Trees & Random Forests",q:"A key disadvantage of single decision trees is:",opts:["Cannot handle categorical features","High variance (sensitive to small data changes)","Too slow to train","Cannot do classification"],ans:1,exp:"Small data changes → very different tree. High variance is the main weakness."},
  {id:70,ch:"Ch3: Decision Trees & Random Forests",q:"Bagging (Bootstrap Aggregation) reduces:",opts:["Bias","Variance","Both bias and variance","Neither"],ans:1,exp:"Averaging B trees from bootstrapped samples reduces variance: Var(X̄) = σ²/B."},
  {id:71,ch:"Ch3: Decision Trees & Random Forests",q:"Bootstrapping means:",opts:["Removing outliers","Sampling WITH replacement from the dataset","Splitting into train/test","Feature engineering"],ans:1,exp:"Draw n observations with replacement from the original sample → bootstrap sample."},
  {id:72,ch:"Ch3: Decision Trees & Random Forests",q:"In bagging, the final classification prediction uses:",opts:["Average of predictions","Majority vote across all B trees","The best single tree","Weighted probabilities only"],ans:1,exp:"For classification: majority vote. For regression: average of predictions."},
  {id:73,ch:"Ch3: Decision Trees & Random Forests",q:"Random Forests improve on bagging by:",opts:["Using fewer trees","Selecting a random subset of m predictors at each split","Using deeper trees","Removing all pruning"],ans:1,exp:"At each split, only m random predictors considered → decorrelates trees → lower variance."},
  {id:74,ch:"Ch3: Decision Trees & Random Forests",q:"In Random Forests for classification, the typical value of m (features per split) is:",opts:["m = p (all features)","m ≈ √p","m = p/3","m = 1"],ans:1,exp:"Classification: m ≈ √p. Regression: m = p/3."},
  {id:75,ch:"Ch3: Decision Trees & Random Forests",q:"Why does Random Forest decorrelate the trees?",opts:["By using different hyperparameters per tree","By randomly selecting feature subsets at each split","By using different loss functions","By pruning randomly"],ans:1,exp:"When a strong predictor exists, not all trees use it at the top → trees become less correlated."},
  {id:76,ch:"Ch3: Decision Trees & Random Forests",q:"The Out-of-Bag (OOB) score in Random Forest is:",opts:["Training accuracy","An estimate of test error using unsampled data","The number of trees","A hyperparameter"],ans:1,exp:"~1/3 of data not used in each bootstrap sample → used to estimate generalization error."},
  {id:77,ch:"Ch3: Decision Trees & Random Forests",q:"In sklearn, to train a Decision Tree classifier:",opts:["from sklearn.tree import DecisionTreeClassifier","from sklearn.forest import TreeClassifier","from sklearn.dt import DTree","from sklearn.ensemble import DecisionTree"],ans:0,exp:"DecisionTreeClassifier is in sklearn.tree module."},
  {id:78,ch:"Ch3: Decision Trees & Random Forests",q:"Which parameter in DecisionTreeClassifier controls the splitting criterion?",opts:["max_depth","criterion (='gini' or 'entropy')","n_estimators","min_samples_split"],ans:1,exp:"criterion='gini' (default) or criterion='entropy'."},
  {id:79,ch:"Ch3: Decision Trees & Random Forests",q:"In sklearn, RandomForestClassifier is in which module?",opts:["sklearn.tree","sklearn.ensemble","sklearn.forest","sklearn.linear_model"],ans:1,exp:"from sklearn.ensemble import RandomForestClassifier."},
  {id:80,ch:"Ch3: Decision Trees & Random Forests",q:"The n_estimators parameter in RandomForestClassifier controls:",opts:["Number of features","Number of trees in the forest","Maximum depth","Minimum samples per leaf"],ans:1,exp:"n_estimators = number of decision trees grown in the ensemble."},
  {id:81,ch:"Ch3: Decision Trees & Random Forests",q:"To access feature importance scores after training:",opts:["model.coef_","model.feature_importances_","model.importances()","model.weights_"],ans:1,exp:"feature_importances_ attribute measures how much each feature reduces impurity."},
  {id:82,ch:"Ch3: Decision Trees & Random Forests",q:"In the Wisconsin Breast Cancer lab, the dataset has how many features?",opts:["10","20","30","54"],ans:2,exp:"30 features (radius, texture, perimeter, area, smoothness, etc.) + target."},
  {id:83,ch:"Ch3: Decision Trees & Random Forests",q:"What is the first step when working with a new dataset?",opts:["Train the model immediately","Exploratory Data Analysis (EDA) — statistics, distributions, correlations","Apply PCA","Run gradient descent"],ans:1,exp:"Always start with EDA: check distributions, missing values, outliers, correlations, class balance."},
  {id:84,ch:"Ch3: Decision Trees & Random Forests",q:"ydata_profiling is used for:",opts:["Training models","Generating automated dataset profile reports","Feature scaling","Hyperparameter tuning"],ans:1,exp:"ProfileReport generates comprehensive analysis: missing values, distributions, correlations."},
  {id:85,ch:"Ch3: Decision Trees & Random Forests",q:"Node with 26 samples: 24 class A, 2 class B. Gini = ?",opts:["≈ 0.14","0","0.50","≈ 0.89"],ans:0,exp:"Gini = (24/26)(2/26) + (2/26)(24/26) = 2×(24/26)(2/26) ≈ 2×0.923×0.077 ≈ 0.142."},
  {id:86,ch:"Ch3: Decision Trees & Random Forests",q:"The bias-variance tradeoff in decision trees: a very deep tree has:",opts:["High bias, low variance","Low bias, high variance","Low bias, low variance","High bias, high variance"],ans:1,exp:"Deep tree → fits training data closely (low bias) but overfits (high variance)."},
  {id:87,ch:"Ch3: Decision Trees & Random Forests",q:"Boosting builds trees:",opts:["Independently in parallel","Sequentially, each learning from previous errors","Randomly","Using a single deep tree"],ans:1,exp:"Boosting: each new tree is built on the residuals of the previous model → sequential."},
  {id:88,ch:"Ch3: Decision Trees & Random Forests",q:"To visualize a decision tree in sklearn, you can use:",opts:["matplotlib.plot_tree()","sklearn.tree.plot_tree() or graphviz","seaborn.treeplot()","pandas.visualize()"],ans:1,exp:"sklearn.tree.plot_tree(model) or export_graphviz for visualization."},
  {id:89,ch:"Ch3: Decision Trees & Random Forests",q:"max_depth in DecisionTreeClassifier controls:",opts:["Number of features used","Maximum depth of the tree","Minimum impurity decrease","Number of leaf nodes"],ans:1,exp:"Limiting max_depth prevents overfitting by restricting tree growth."},
  {id:90,ch:"Ch3: Decision Trees & Random Forests",q:"When comparing Decision Tree vs Random Forest, Random Forest generally:",opts:["Is more interpretable","Has lower test accuracy","Generalizes better to unseen data","Trains faster"],ans:2,exp:"RF reduces variance via averaging → better generalization, but less interpretable."},

  // ═══════════════════════════════════════════════════
  // CHAPTER 4: NEURAL NETWORKS (30 questions)
  // ═══════════════════════════════════════════════════
  {id:91,ch:"Ch4: Neural Networks",q:"A Neural Network is inspired by:",opts:["Database systems","The human brain","Linear algebra only","Decision trees"],ans:1,exp:"NNs are computational models inspired by biological neural networks."},
  {id:92,ch:"Ch4: Neural Networks",q:"A single neuron (perceptron) computes:",opts:["z = Σ(wᵢxᵢ) + b, then applies activation f(z)","z = max(x₁, x₂, ...)","z = x₁ × x₂ × ... × xₙ","z = median of inputs"],ans:0,exp:"Weighted sum of inputs plus bias → activation function → output."},
  {id:93,ch:"Ch4: Neural Networks",q:"In a perceptron, the bias term b serves to:",opts:["Normalize the weights","Shift the decision boundary","Remove outliers","Reduce dimensionality"],ans:1,exp:"Bias allows the neuron's activation to shift, enabling better fitting."},
  {id:94,ch:"Ch4: Neural Networks",q:"ReLU activation function is defined as:",opts:["f(x) = 1/(1+e⁻ˣ)","f(x) = max(0, x)","f(x) = tanh(x)","f(x) = x²"],ans:1,exp:"ReLU: output is x if x>0, else 0. Simple, non-linear, widely used in deep learning."},
  {id:95,ch:"Ch4: Neural Networks",q:"If the input to ReLU is −3, the output is:",opts:["−3","3","0","1"],ans:2,exp:"ReLU = max(0, −3) = 0."},
  {id:96,ch:"Ch4: Neural Networks",q:"If the input to ReLU is 5, the output is:",opts:["0","1","5","0.5"],ans:2,exp:"ReLU = max(0, 5) = 5."},
  {id:97,ch:"Ch4: Neural Networks",q:"An MLP (Multilayer Perceptron) is a feedforward network with:",opts:["Only input and output layers","At least one hidden layer between input and output","Recurrent connections","No activation functions"],ans:1,exp:"MLP = input layer + ≥1 hidden layers + output layer, fully connected, feedforward."},
  {id:98,ch:"Ch4: Neural Networks",q:"In a 'fully connected' layer, each neuron is connected to:",opts:["One neuron in the next layer","Every neuron in the next layer","Random neurons","No neurons"],ans:1,exp:"Fully connected (dense): every neuron connects to every neuron in the next layer."},
  {id:99,ch:"Ch4: Neural Networks",q:"Forward propagation is:",opts:["Computing gradients backward","Passing inputs through the network to generate predictions","Updating weights","Initializing biases"],ans:1,exp:"Forward prop: input → hidden layers → output = prediction."},
  {id:100,ch:"Ch4: Neural Networks",q:"Backpropagation is used to:",opts:["Generate predictions","Compute gradients of the loss w.r.t. weights for updating","Initialize the network","Choose architecture"],ans:1,exp:"Backprop computes ∂Loss/∂w for each weight → used in gradient descent update."},
  {id:101,ch:"Ch4: Neural Networks",q:"The training loop for a neural network is:",opts:["Forward prop → Loss → Backprop → Update weights → Repeat","Backprop → Forward → Loss → Predict","Loss → Forward → Update → Backprop","Update → Predict → Loss"],ans:0,exp:"Forward prop → compute loss → backprop gradients → update weights → repeat."},
  {id:102,ch:"Ch4: Neural Networks",q:"The Universal Approximation Theorem states that MLPs can:",opts:["Only learn linear functions","Approximate any continuous function (with enough neurons)","Only classify images","Replace all other algorithms"],ans:1,exp:"A single hidden layer MLP with enough neurons can approximate any continuous function."},
  {id:103,ch:"Ch4: Neural Networks",q:"Why are activation functions needed in neural networks?",opts:["To speed up computation","To introduce non-linearity (learn complex patterns)","To reduce the number of parameters","To normalize inputs"],ans:1,exp:"Without activation, stacked layers = single linear function. Non-linearity enables learning complex patterns."},
  {id:104,ch:"Ch4: Neural Networks",q:"For multi-class classification, the output layer typically uses:",opts:["ReLU","Sigmoid","Softmax","Tanh"],ans:2,exp:"Softmax outputs probabilities across K classes that sum to 1."},
  {id:105,ch:"Ch4: Neural Networks",q:"For binary classification, the output layer typically uses:",opts:["ReLU","Sigmoid","Linear","Tanh"],ans:1,exp:"Sigmoid maps output to (0,1) — interpreted as P(class=1)."},
  {id:106,ch:"Ch4: Neural Networks",q:"In sklearn, MLPClassifier is imported from:",opts:["sklearn.linear_model","sklearn.neural_network","sklearn.ensemble","sklearn.tree"],ans:1,exp:"from sklearn.neural_network import MLPClassifier."},
  {id:107,ch:"Ch4: Neural Networks",q:"MLPClassifier(hidden_layer_sizes=(128, 64)) creates a network with:",opts:["1 hidden layer of 192 neurons","2 hidden layers: 128 neurons then 64 neurons","128 layers of 64 neurons each","64 layers of 128 neurons each"],ans:1,exp:"Tuple (128, 64) = first hidden layer has 128 neurons, second has 64."},
  {id:108,ch:"Ch4: Neural Networks",q:"In the MLP lab, which optimizer is commonly used?",opts:["SGD only","Adam","OLS","Bagging"],ans:1,exp:"solver='adam' — adaptive moment estimation, popular for neural networks."},
  {id:109,ch:"Ch4: Neural Networks",q:"The parameter alpha in MLPClassifier controls:",opts:["Learning rate","L2 regularization strength","Number of layers","Batch size"],ans:1,exp:"alpha is the L2 penalty parameter that prevents overfitting."},
  {id:110,ch:"Ch4: Neural Networks",q:"early_stopping=True in MLPClassifier:",opts:["Stops after 1 epoch","Stops training when validation loss stops improving","Removes the first layer","Disables regularization"],ans:1,exp:"Early stopping prevents overfitting by monitoring validation performance."},
  {id:111,ch:"Ch4: Neural Networks",q:"learning_rate_init in MLPClassifier sets:",opts:["The initial learning rate for weight updates","The number of iterations","The regularization strength","The batch size"],ans:0,exp:"learning_rate_init controls the initial step size for gradient updates."},
  {id:112,ch:"Ch4: Neural Networks",q:"A neural network with weights initialized to zero everywhere will:",opts:["Converge quickly","Fail to learn (symmetry problem)","Achieve perfect accuracy","Need no training"],ans:1,exp:"All neurons compute the same output → same gradients → never break symmetry. Random init is needed."},
  {id:113,ch:"Ch4: Neural Networks",q:"Types of neural networks include:",opts:["MLP, CNN, RNN, GNN","Only MLP","Only CNN","Only Perceptron"],ans:0,exp:"Perceptron, MLP, CNN (images), RNN (sequences), GNN (graphs), etc."},
  {id:114,ch:"Ch4: Neural Networks",q:"Compared to Logistic Regression, an MLP can model:",opts:["Only linear decision boundaries","Non-linear decision boundaries","Fewer patterns","Only regression tasks"],ans:1,exp:"Hidden layers with non-linear activations allow complex, non-linear boundaries."},
  {id:115,ch:"Ch4: Neural Networks",q:"In TensorFlow/Keras, layers.Dense(10, activation='tanh') creates:",opts:["10 layers with tanh","A fully connected layer with 10 neurons using tanh activation","10 inputs","A convolutional layer"],ans:1,exp:"Dense = fully connected. 10 neurons. activation='tanh'."},
  {id:116,ch:"Ch4: Neural Networks",q:"model.compile(optimizer='adam', loss='mean_squared_error') is used for:",opts:["Classification only","Regression tasks","Clustering","Data loading"],ans:1,exp:"MSE loss → regression. For classification, use 'categorical_crossentropy'."},
  {id:117,ch:"Ch4: Neural Networks",q:"Cross-validation (e.g., 5-fold) in sklearn for MLP uses:",opts:["train_test_split only","cross_val_score(mlp, X, y, cv=5)","mlp.crossvalidate()","KFold.predict()"],ans:1,exp:"from sklearn.model_selection import cross_val_score; cross_val_score(model, X, y, cv=5)."},
  {id:118,ch:"Ch4: Neural Networks",q:"An MLP with 3 input features, hidden layer of 4 neurons, and 2 output neurons has how many weight parameters (excluding biases)?",opts:["12 + 8 = 20","3×4 + 4×2 = 20","3 + 4 + 2 = 9","3×4×2 = 24"],ans:1,exp:"Input→Hidden: 3×4=12 weights. Hidden→Output: 4×2=8 weights. Total = 20."},
  {id:119,ch:"Ch4: Neural Networks",q:"The correct ML pipeline order for a new dataset is:",opts:["Train → Load → Split → Evaluate","Load → EDA/Profile → Preprocess/Scale → Split → Train → Evaluate","Split → Load → Train → EDA","Evaluate → Train → Load → Split"],ans:1,exp:"Load data → EDA/profiling → preprocess (clean, scale) → split → train → evaluate."},
  {id:120,ch:"Ch4: Neural Networks",q:"Which library is used for dataset profiling in the labs?",opts:["scikit-learn","ydata_profiling (formerly pandas-profiling)","TensorFlow","matplotlib"],ans:1,exp:"ydata_profiling generates comprehensive reports: missing values, distributions, correlations."},
];

const chapterColors = {
  "Ch1: Linear Regression": {bg: "#fef3c7", accent: "#d97706", badge: "#92400e", light:"#fffbeb"},
  "Ch2: Logistic Regression & KNN": {bg: "#dbeafe", accent: "#2563eb", badge: "#1e3a8a", light:"#eff6ff"},
  "Ch3: Decision Trees & Random Forests": {bg: "#d1fae5", accent: "#059669", badge: "#065f46", light:"#ecfdf5"},
  "Ch4: Neural Networks": {bg: "#ede9fe", accent: "#7c3aed", badge: "#4c1d95", light:"#f5f3ff"},
};

export default function MLQuizApp() {
  const [answers, setAnswers] = useState({});
  const [revealed, setRevealed] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [filter, setFilter] = useState("All");
  const [showScrollTop, setShowScrollTop] = useState(false);
  const topRef = useRef(null);
  const resultsRef = useRef(null);

  useEffect(() => {
    const h = () => setShowScrollTop(window.scrollY > 600);
    window.addEventListener("scroll", h);
    return () => window.removeEventListener("scroll", h);
  }, []);

  const chapters = ["All", ...new Set(allQuestions.map(q => q.ch))];
  const filtered = filter === "All" ? allQuestions : allQuestions.filter(q => q.ch === filter);

  const handleSelect = (qid, idx) => {
    if (submitted) return;
    setAnswers(a => ({...a, [qid]: idx}));
  };

  const toggleReveal = (qid) => {
    setRevealed(r => ({...r, [qid]: !r[qid]}));
  };

  const score = allQuestions.reduce((s, q) => s + (answers[q.id] === q.ans ? 1 : 0), 0);
  const answered = Object.keys(answers).length;

  const handleSubmit = () => {
    setSubmitted(true);
    setFilter("All");
    setTimeout(() => resultsRef.current?.scrollIntoView({behavior:"smooth"}), 100);
  };

  const handleReset = () => {
    setAnswers({});
    setRevealed({});
    setSubmitted(false);
    setFilter("All");
    topRef.current?.scrollIntoView({behavior:"smooth"});
  };

  const pct = allQuestions.length > 0 ? Math.round((score / allQuestions.length) * 100) : 0;

  return (
    <div ref={topRef} style={{fontFamily:"'Crimson Pro', 'Georgia', serif", background:"#0f0f14", minHeight:"100vh", color:"#e2e2e8"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; margin:0; padding:0; }
        ::selection { background: #7c3aed; color: #fff; }
        .quiz-card { transition: all 0.2s ease; }
        .quiz-card:hover { transform: translateY(-1px); }
        .opt-btn { transition: all 0.15s ease; cursor: pointer; border: 2px solid #2a2a35; }
        .opt-btn:hover:not(.locked) { border-color: #555; background: #1e1e28 !important; }
        .opt-btn.selected { border-color: #7c3aed !important; background: #1a1530 !important; }
        .opt-btn.correct { border-color: #10b981 !important; background: #0a2e1f !important; }
        .opt-btn.wrong { border-color: #ef4444 !important; background: #2e0a0a !important; }
        .opt-btn.locked { cursor: default; }
        .reveal-btn { cursor:pointer; transition: all 0.15s; }
        .reveal-btn:hover { opacity: 0.8; }
        .ch-filter { cursor:pointer; padding:8px 16px; border-radius:20px; border:1.5px solid #333; font-size:13px; transition:all 0.15s; font-family:'JetBrains Mono',monospace; }
        .ch-filter:hover { border-color: #666; }
        .ch-filter.active { background:#7c3aed; border-color:#7c3aed; color:#fff; }
        .scroll-top { position:fixed; bottom:24px; right:24px; width:48px; height:48px; border-radius:50%; background:#7c3aed; color:#fff; display:flex; align-items:center; justify-content:center; cursor:pointer; font-size:20px; border:none; box-shadow:0 4px 20px rgba(124,58,237,0.4); transition:all 0.2s; z-index:99; }
        .scroll-top:hover { transform:scale(1.1); }
        .progress-fill { transition: width 0.5s ease; }
        @keyframes fadeIn { from {opacity:0;transform:translateY(8px)} to {opacity:1;transform:translateY(0)} }
        .fade-in { animation: fadeIn 0.3s ease forwards; }
      `}</style>

      {/* Header */}
      <div style={{background:"linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)", padding:"40px 24px 32px", borderBottom:"1px solid #222"}}>
        <div style={{maxWidth:900, margin:"0 auto"}}>
          <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:8}}>
            <span style={{fontSize:32}}>🎓</span>
            <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:11,color:"#7c3aed",letterSpacing:2,textTransform:"uppercase"}}>EFREI Paris — Machine Learning</span>
          </div>
          <h1 style={{fontSize:36,fontWeight:700,lineHeight:1.2,marginBottom:8,background:"linear-gradient(90deg,#e2e2e8,#a78bfa)",WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>
            Final Exam Review — 120 MCQs
          </h1>
          <p style={{color:"#888",fontSize:15,maxWidth:600}}>
            Covers all 4 chapters: Linear Regression, Logistic Regression & KNN, Decision Trees & Random Forests, Neural Networks. Formulas, calculations, code & concepts.
          </p>
        </div>
      </div>

      {/* Score Bar */}
      <div style={{position:"sticky",top:0,zIndex:50,background:"#131318",borderBottom:"1px solid #222",padding:"12px 24px"}}>
        <div style={{maxWidth:900,margin:"0 auto",display:"flex",alignItems:"center",gap:16,flexWrap:"wrap"}}>
          <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:13,color:"#aaa"}}>
            <span style={{color:"#7c3aed",fontWeight:600}}>{answered}</span>/{allQuestions.length} answered
          </div>
          <div style={{flex:1,height:6,background:"#222",borderRadius:3,minWidth:120}}>
            <div className="progress-fill" style={{height:6,borderRadius:3,background:"linear-gradient(90deg,#7c3aed,#a78bfa)",width:`${(answered/allQuestions.length)*100}%`}}/>
          </div>
          {!submitted ? (
            <button onClick={handleSubmit} disabled={answered===0} style={{padding:"8px 24px",borderRadius:8,border:"none",background:answered>0?"#7c3aed":"#333",color:answered>0?"#fff":"#666",fontWeight:600,cursor:answered>0?"pointer":"default",fontFamily:"'JetBrains Mono',monospace",fontSize:12}}>
              Submit & See Score
            </button>
          ) : (
            <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:14}}>
              Score: <span style={{color:pct>=60?"#10b981":"#ef4444",fontWeight:700,fontSize:18}}>{score}/{allQuestions.length}</span> <span style={{color:"#666"}}>({pct}%)</span>
            </div>
          )}
        </div>
      </div>

      {/* Chapter Filters */}
      <div style={{maxWidth:900,margin:"20px auto 0",padding:"0 24px",display:"flex",gap:8,flexWrap:"wrap"}}>
        {chapters.map(ch => (
          <button key={ch} className={`ch-filter ${filter===ch?"active":""}`} onClick={()=>setFilter(ch)} style={filter===ch?{}:{background:"transparent",color:"#888"}}>
            {ch === "All" ? `All (${allQuestions.length})` : `${ch.split(":")[0]} (${allQuestions.filter(q=>q.ch===ch).length})`}
          </button>
        ))}
      </div>

      {/* Questions */}
      <div style={{maxWidth:900,margin:"24px auto",padding:"0 24px"}}>
        {filtered.map((q, qi) => {
          const colors = chapterColors[q.ch] || {bg:"#222",accent:"#888",badge:"#444",light:"#1a1a1a"};
          const userAns = answers[q.id];
          const isRevealed = revealed[q.id];
          const isCorrect = userAns === q.ans;

          return (
            <div key={q.id} className="quiz-card fade-in" style={{background:"#18181f",border:"1px solid #252530",borderRadius:12,padding:"24px",marginBottom:16,borderLeft:`4px solid ${colors.accent}`}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",marginBottom:12,gap:8,flexWrap:"wrap"}}>
                <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:10,background:colors.badge,color:"#fff",padding:"3px 10px",borderRadius:10,letterSpacing:0.5}}>
                  {q.ch}
                </span>
                <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:11,color:"#555"}}>Q{q.id}</span>
              </div>
              <p style={{fontSize:16,fontWeight:600,lineHeight:1.5,marginBottom:16,color:"#e2e2e8"}}>{q.q}</p>
              <div style={{display:"flex",flexDirection:"column",gap:8}}>
                {q.opts.map((opt, oi) => {
                  let cls = "opt-btn";
                  if (submitted) cls += " locked";
                  if (userAns === oi && !submitted) cls += " selected";
                  if (submitted && oi === q.ans) cls += " correct";
                  if (submitted && userAns === oi && oi !== q.ans) cls += " wrong";

                  return (
                    <div key={oi} className={cls} onClick={()=>handleSelect(q.id,oi)} style={{padding:"10px 14px",borderRadius:8,display:"flex",alignItems:"flex-start",gap:10,background:"#13131a"}}>
                      <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:12,color:submitted&&oi===q.ans?"#10b981":submitted&&userAns===oi&&oi!==q.ans?"#ef4444":userAns===oi?"#a78bfa":"#555",fontWeight:600,minWidth:20}}>
                        {String.fromCharCode(65+oi)}.
                      </span>
                      <span style={{fontSize:14,lineHeight:1.4}}>{opt}</span>
                      {submitted && oi === q.ans && <span style={{marginLeft:"auto",fontSize:14}}>✓</span>}
                      {submitted && userAns===oi && oi!==q.ans && <span style={{marginLeft:"auto",fontSize:14}}>✗</span>}
                    </div>
                  );
                })}
              </div>
              {/* Reveal answer button */}
              {!submitted && (
                <div style={{marginTop:12,display:"flex",alignItems:"center",gap:8}}>
                  <button className="reveal-btn" onClick={()=>toggleReveal(q.id)} style={{fontFamily:"'JetBrains Mono',monospace",fontSize:11,background:"transparent",border:"1px solid #333",color:"#888",padding:"4px 12px",borderRadius:6}}>
                    {isRevealed ? "Hide answer" : "👁 Show answer"}
                  </button>
                </div>
              )}
              {(isRevealed || submitted) && (
                <div style={{marginTop:12,padding:"12px 14px",background:"#111116",borderRadius:8,borderLeft:`3px solid ${submitted?(isCorrect?"#10b981":"#ef4444"):colors.accent}`}}>
                  <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:11,color:submitted?(isCorrect?"#10b981":"#ef4444"):colors.accent,marginBottom:4}}>
                    {submitted ? (isCorrect ? "✓ Correct" : `✗ Wrong — Correct: ${String.fromCharCode(65+q.ans)}`) : `Answer: ${String.fromCharCode(65+q.ans)}`}
                  </div>
                  <p style={{fontSize:13,color:"#aaa",lineHeight:1.5}}>{q.exp}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Final Results */}
      {submitted && (
        <div ref={resultsRef} style={{maxWidth:900,margin:"0 auto 40px",padding:"0 24px"}}>
          <div style={{background:"linear-gradient(135deg,#1a1a2e,#16213e)",border:"1px solid #252530",borderRadius:16,padding:"32px",textAlign:"center"}}>
            <h2 style={{fontSize:28,fontWeight:700,marginBottom:8}}>Final Score</h2>
            <div style={{fontSize:64,fontWeight:700,color:pct>=60?"#10b981":"#ef4444",fontFamily:"'JetBrains Mono',monospace"}}>{pct}%</div>
            <p style={{fontSize:18,color:"#aaa",marginBottom:4}}>{score} / {allQuestions.length} correct</p>
            <p style={{fontSize:14,color:"#666",marginBottom:24}}>
              {pct>=90?"Outstanding! 🏆":pct>=75?"Great job! 🎉":pct>=60?"Good — review weak areas 📖":"Keep studying — you'll get there! 💪"}
            </p>

            {/* Per-chapter breakdown */}
            <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fit,minmax(180px,1fr))",gap:12,marginBottom:24}}>
              {Object.keys(chapterColors).map(ch => {
                const qs = allQuestions.filter(q => q.ch === ch);
                const correct = qs.filter(q => answers[q.id] === q.ans).length;
                const chPct = Math.round((correct/qs.length)*100);
                const c = chapterColors[ch];
                return (
                  <div key={ch} style={{background:"#13131a",borderRadius:10,padding:"16px",borderTop:`3px solid ${c.accent}`}}>
                    <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:10,color:c.accent,marginBottom:6}}>{ch.split(":")[0]}</div>
                    <div style={{fontSize:24,fontWeight:700,color:chPct>=60?"#10b981":"#ef4444",fontFamily:"'JetBrains Mono',monospace"}}>{correct}/{qs.length}</div>
                    <div style={{fontSize:12,color:"#666"}}>{chPct}%</div>
                  </div>
                );
              })}
            </div>

            <button onClick={handleReset} style={{padding:"12px 32px",borderRadius:10,border:"none",background:"#7c3aed",color:"#fff",fontWeight:600,cursor:"pointer",fontSize:14,fontFamily:"'JetBrains Mono',monospace"}}>
              🔄 Retry Quiz
            </button>
          </div>
        </div>
      )}

      {/* Scroll to top */}
      {showScrollTop && (
        <button className="scroll-top" onClick={()=>topRef.current?.scrollIntoView({behavior:"smooth"})}>↑</button>
      )}
    </div>
  );
}
