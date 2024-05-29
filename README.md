### **![Screen Shot 2024-02-11 at 5.30.35 PM.png](https://ncf.instructure.com/courses/8127/files/755958/preview)Applied Machine Learning**

Syllabus: [ML_Spring_Syllabus](https://ncf.instructure.com/courses/8127/files/752828?wrap=1)

---

#### **Class Diary:**

**Lecture 1** (Jan 30):

- **Types of Machine Learning**: We differentiated between supervised, unsupervised, and reinforcement learning as the main approaches in the field.
- **Classification and** **Regression**: Within supervised learning these are the two tasks, where our target is categorical or continuous respectively.
- **Supervised Learning with the IRIS Dataset**: We demonstrated the k-nearest neighbors algorithm using the IRIS dataset to introduce classification tasks.
- **Importance of Data Splitting**: We emphasized the need for splitting data into training and testing sets to prevent misleading accuracy scores.
- **Model Evaluation Techniques**: We introduced train-test split, validation sets, and cross-validation methods for assessing model performance.
- **Cross-Validation**: We explained how cross-validation helps in estimating model performance more reliably by using different data subsets for training and testing.

---

### **Supervised Learning:**

#### **Regression:**

**Lecture 2** (Feb 1): We introduced the concept of Linear Regression, and used it as an example to illustrate other topics including

- **Loss or Objective Functions**: The class explored the significance of loss functions, such as the mean squared error, in defining how well a model is performing.
- **Gradient Descent**: We discussed the gradient descent optimization algorithm, emphasizing its role in minimizing the loss function by iteratively adjusting model parameters.
- **Analytical vs. Gradient Descent Solutions**: The differences between analytical solutions, like the normal equation, and iterative solutions, like gradient descent, were highlighted, including when and why to use one over the other.
- **Feature Engineering and Polynomial Regression**: The session covered feature engineering, specifically the creation of polynomial features, and how they enhance linear regression models to fit non-linear data.
- **Regularization Techniques**: We introduced LASSO and Ridge regression as regularization techniques to prevent overfitting by penalizing large coefficients.
  - LASSO (L1) Regularization usually results in some coefficients being set to zero (this is called sparsity)
  - Ridge (L2) Regularization discourages coefficients from becoming too big but doesn't necessarily result in sparsity
- **Bias-Variance Tradeoff**: We discussed "bias-variance tradeoff" between model complexity and generalization ability.

**Lecture 3** (Feb 6): We continued to look at Linear Regression, this time focusing on:

- **Loss Functions**: How they quantify model errors and guide the learning process. We looked at some interactive examples.
- **Gradient Descent**: How we minimize loss functions whose minima are too hard to calculate explicitly.
![LaTeX: \vec\theta \rightarrow \vec\theta - \alpha \vec\nabla \mathscr{L}](https://ncf.instructure.com/equation_images/%255Cvec%255Ctheta%2520%255Crightarrow%2520%255Cvec%255Ctheta%2520-%2520%255Calpha%2520%255Cvec%255Cnabla%2520%255Cmathscr%257BL%257D?scale=1)
- **Learning Rate**: The importance of choosing the right learning rate for convergence.
- **Convexity**: Its significance in ensuring we find the global optimum.
- **Manual Derivation**: Calculating derivatives to understand gradients and looking at basic python implementation.
- **Momentum**: Introduced to accelerate convergence and navigate complex loss landscapes more effectively.
- **Regularization Techniques**: Briefly discussed LASSO and Ridge regression to reduce overfitting and, in the case of LASSO, make the model sparse.

![google-deepmind-kUmcSBJcFPg-unsplash.jpg](https://ncf.instructure.com/courses/8127/files/755959/preview)

**Lecture 4** (Feb 8):

- **Linear Regression** (wrap-up):
  - **Feature Scaling**: Discussed the importance of scaling features to prevent issues with model convergence and numerical stability.
  - **Model Interpretation**: Examined the coefficients of a linear regression model to understand the impact of each feature.
  - **Feature Engineering**: Revisited one-hot encoding of categorical variables, demonstrating its application in our model analysis.
- **Decision Trees**: Introduced decision trees, clarifying the role of supervised learning and how it applies to model training and deployment.
  - We illustrating classification with decision trees on the board, and discussed how a decision tree might go about learning a decision boundary.
  - We compared the tree structure to the implications for

 decision boundaries

![image.png](https://ncf.instructure.com/courses/8127/files/755955/preview)

#### **Classification:**

**Lecture 5** (Feb 13):

- Classification Problems and Decision Boundaries
  - There are infinitely many ways to draw a general boundary
  - If we restrict ourselves to rectilinear (boxy) boundaries there are still too many
  - Decision Trees greedily choose the best binary split at each point and recursively partition the space
- Decision Trees
  - How they're trained
    - What do we mean by "best split"?
      - For Classification this could be:
        - Entropy
        - Gini Impurity
        - Misclassification Rate
      - Using them for Regression
        - Typically use MSE
    - Early Stopping
      - Max depth
      - Min samples at leaf (or at decision node)
      - Max tree size (total number of leaves)
    - Pruning
      - Cost complexity pruning
  - Hyperparameters
  - Pros
    - Interpretability - Feature Importance
    - Fast to train and query
    - Insensitive to feature scaling
  - Cons
    - Tends to overfit
    - Non-robust
    - Bad at extrapolating out of training distribution
- Ensemble Methods for Decision Trees
  - Bagging (Bootstrap-Aggregating)
  - Random Forests are an example of Bagging where we also introduce randomized axis choice.
  - Intro to Boosting
    - Training on the residuals of previous models
    - Shrinkage parameter ![LaTeX: \lambda](https://ncf.instructure.com/equation_images/%255Clambda?scale=1)

**Lecture 6** (Feb 15):

- Reiterated concepts from last class including:
  - How trees are structured
  - Finding the best decision to reduce loss
  - Random Forests as Bootstrap Aggregating + Randomized Feature Selection
- Gini Impurity: meaning and calculation
- Looked at concrete example with Titanic dataset:
  - Explored issue leading to alleged 100% accuracy (including the target in the features)
  - Explored why the first question used the "gender" feature
  - Ran though the process of predicting unseen data

**Lecture 7** (Feb 20):

- Node Importance and Feature Importance
  - Definitions in terms of Loss function
    - ![LaTeX: \Delta \mathscr{L} = \mathscr{L}_\text{parent} - \sum_i p_i \mathscr{L}_\text{children}](https://ncf.instructure.com/equation_images/%255CDelta%2520%255Cmathscr%257BL%257D%2520%253D%2520%255Cmathscr%257BL%257D_%255Ctext%257Bparent%257D%2520-%2520%255Csum_i%2520p_i%2520%255Cmathscr%257BL%257D_%255Ctext%257Bchildren%257D?scale=1) where ![LaTeX: p_i](https://ncf.instructure.com/equation_images/p_i?scale=1) is the proportion of samples going to each child node
    - This quantity is called the **Information Gain** if the the loss function is Entropy.
  - Examples: Titanic Dataset with Decision Tree and then with Random Forest, 8x8 Digits
- Introduced 8x8 digits classification:
  - Looked at the dependence of the 'test_loss' upon number of trees
  - Looked at the dependence of the 'test_loss' upon max depth of the trees
  - Looked at the gap between the 'test_loss' and 'train_loss' which (when large) is a case overfitting
- Gradient Boosting
  - Successively adds models trained to reduce the residuals. This acts like taking steps in the space of targets in the direction of ![LaTeX: \vec y_\text{true}](https://ncf.instructure.com/equation_images/%255Cvec%2520y_%255Ctext%257Btrue%257D?scale=1). That direction is the eponymous "gradient"
- AdaBoost
  - Adds weights to both the samples and to the estimators:
    - The weight of a sample is increased if the ensemble prediction error for that sample is bad
    - The weight of each estimator is determined by its performance, anti-predictors can be given negative weight
- XGBoost: The model with all the bells and whistles, the best overall but with lots of hyperparameters to tune. Can underperform simpler models on smaller datasets.
- CatBoost: Specialized in handling categorical variables.
- LightGBM: Lightweight model which works well on big datasets

 or when efficiency is needed (for example: in mobile applications)

**Lecture 8** (Feb 22):

- Boosting Methods Summary
- **Quiz 1**
- Review of quiz questions
- Support Vector Machines
  - Decision boundary is a straight line, plane, or hyperplane depending on if you're in 2, 3, or 4+ dimensions respectively
  - Specifically we pick the boundary with the "maximum margin" (the widest street)

- Digression into Linear Algebra: Dot Products
  - How to calculate them
  - Geometric interpretation as length of projection of one vector onto another
  - Orthogonality (Right angles) ![LaTeX: \Leftrightarrow](https://ncf.instructure.com/equation_images/%255CLeftrightarrow?scale=1) zero dot product
  - Aside: Relationship to angles (![LaTeX: \vec a \cdot \vec b = \|a\|\|b\| \cos \theta](https://ncf.instructure.com/equation_images/%255Cvec%2520a%2520%255Ccdot%2520%255Cvec%2520b%2520%253D%2520%255C%257Ca%255C%257C%255C%257Cb%255C%257C%2520%255Ccos%2520%255Ctheta?scale=1)) and notion of "Cosine similarity"

**Lecture 9** (Feb 27):

- Getting non-linear boundaries with feature maps
  - [https://www.youtube.com/watch?v=OdlNM96sHio](https://www.youtube.com/watch?v=OdlNM96sHio)

- SVM decision rule and constraints ![LaTeX: y_i(\vec w \cdot \vec x_i + b) - 1 \geq 0](https://ncf.instructure.com/equation_images/y_i(%255Cvec%2520w%2520%255Ccdot%2520%255Cvec%2520x_i%2520%252B%2520b)%2520-%25201%2520%255Cgeq%25200?scale=1)
- Calculating the width of the street in terms of ![LaTeX: \vec w](https://ncf.instructure.com/equation_images/%255Cvec%2520w?scale=1)
- "Loss function"
- Dual form of loss function
  - Dual form of decision rule
  - Convexity
  - Depends only upon dot products of ![LaTeX: \vec x_i](https://ncf.instructure.com/equation_images/%255Cvec%2520x_i?scale=1)'s

- The "Kernel Trick"
  - Feature maps revisited (reducing computational complexity)
  - Polynomial Kernel
  - Gaussian Kernel (aka. Radial Basis Functions (RBF))

![Screen Shot 2024-02-27 at 3.39.03 PM.png](https://ncf.instructure.com/courses/8127/files/761078/preview)

**Lecture 10** (Feb 29):

- Interactive SVM fitting example
- Hard vs Soft margin SVMs
  - Slack parameter
  - "Hinge Loss" formulation

- Gaussian Kernel
  - Meaning of ![LaTeX: \sigma](https://ncf.instructure.com/equation_images/%255Csigma?scale=1) and ![LaTeX: \gamma := \frac{1}{2\sigma^2}](https://ncf.instructure.com/equation_images/%255Cgamma%2520%253A%253D%2520%255Cfrac%257B1%257D%257B2%255Csigma%255E2%257D?scale=1), and implications for overfitting/underfitting

- Face Classification Example

**Lecture 11** (Mar 5):

- Example: Binary Classification of College Acceptance based upon SAT Score
  - Looking at p(Acceptance | SAT Score) we see that the curve is s-shaped.
  - We want to model it in a principled way
  - We look at the odds (p(Accepted)/p(Rejected)) and want to view it on a log scale.
  - Looking at the log-odds we see that it is linear!
  - This gives us the idea to model the log-odds with a linear function. This corresponds to modeling p(Acceptance | SAT Score) with a "sigmoid" function

- Logistic Regression
  - Model p(y|x) as ![LaTeX: p(y|x) = \frac{e^{wx+b}}{1+e^{wx+b}}](https://ncf.instructure.com/equation_images/p(y%257Cx)%252

0%253D%2520%255Cfrac%257Be%255E%257Bwx%252Bb%257D%257D%257B1%252Be%255E%257Bwx%252Bb%257D%257D?scale=1)
  - Train by **maximizing** the "Likelihood" of the data according to the model
  ![LaTeX: \prod_{I=1}^{N} y_{\text{pred}, i}^{y_i} + (1-y_{\text{pred}, i})^{(1-y_i)}](https://ncf.instructure.com/equation_images/%255Cprod_%257BI%253D1%257D%255E%257BN%257D%2520y_%257B%255Ctext%257Bpred%257D%252C%2520i%257D%255E%257By_i%257D%2520%252B%2520(1-y_%257B%255Ctext%257Bpred%257D%252C%2520i%257D)%255E%257B(1-y_i)%257D?scale=1)In practice we do this by **minimizing** the negative log of the Likelihood (nll)
  ![LaTeX: -\frac{1}{N} \sum_{i=1}^N y_i \log(y_{\text{pred},i}) + (1-y_i) \log(1-\hat y_{\text{pred},i})](https://ncf.instructure.com/equation_images/-%255Cfrac%257B1%257D%257BN%257D%2520%255Csum_%257Bi%253D1%257D%255EN%2520y_i%2520%255Clog(y_%257B%255Ctext%257Bpred%257D%252Ci%257D)%2520%252B%2520(1-y_i)%2520%255Clog(1-%255Chat%2520y_%257B%255Ctext%257Bpred%257D%252Ci%257D)?scale=1)This is the "famous" Binary Cross Entropy (BCE) loss

**Lecture 12** (Mar 7):

- Model Evaluation and Selection
  - Confusion Matrix
  - **Precision, Recall, F1 score,**
  - **Specificity and Sensitivity**

- Example with unbalanced data

- ROC curves, and AUC (Self study: Didn't fit into class)

**Lecture 13** (Mar 12):

- Followup on Last Class: ROC curves:
  - Selecting thresholds with ROC curves
  - Evaluating binary classifiers with Area Under Curve (AUC)
    - Diagonal line (AUC=0.5) is a random classifier which is totally uninformative
    - Perfect classifier (AUC=1.0)
  - Interactive example

- Techniques for handling with unbalanced data:
  - Using appropriate model evaluation metrics
    - **Precision, Recall, F1, and AUC**
    - **Choosing a** threshold appropriately
  - Undersampling
    - Random Undersampling
    - Prototyping Methods (Clustering)
  - Oversampling
    - Random Oversampling
    - SMOTE
  - Class Weighted Learning
  - Ensemble Methods
    - Balanced Random Forest
    - AdaBoost
  - Data Augmentation
  - Anomaly Detection

- Neural Networks:
  - Biological Inspiration and the Perceptron

### **Deep Learning**

**Lecture 14** (![LaTeX: \pi](https://ncf.instructure.com/equation_images/%255Cpi?scale=1)):

- **Quiz 2**
- Neural Networks:
  - Biological Inspiration and the Perceptron
  - Interpreting Perceptrons as Boolean functions
  - Interpreting Perceptrons as linear models
  - Examining the perceptron decision boundary (it's just a straight line)
  - Multi-Level Perceptrons (MLPs)
  - Example: Construction an MLP with a triangular decision boundary.

🏖 SPRING BREAK 🏝

---

### **![Screen Shot 2024-02-20 at 10.10.59 PM.png](https://ncf.instructure.com/courses/8127/files/758903/preview)**

**Lecture 15** (Mar 26):

- Neural Networks and MLPs
  - MLPs are a specific type of Neural Net: (Namely, fully connected, feed forward NNs as opposed to something like a CNN)

- Architecture
  - Input, Hidden, and Output Layers
  - Number of layers and layer sizes are hyperparameters

- Activation

 Function
  - Each neuron computes a sum: ![LaTeX: z_i=\vec w_i \cdot \vec x + b_i](https://ncf.instructure.com/equation_images/z_i%253D%255Cvec%2520w_i%2520%255Ccdot%2520%255Cvec%2520x%2520%252B%2520b_i?scale=1) and then passes it though an activation function ![LaTeX: a_i = \phi(z_i)](https://ncf.instructure.com/equation_images/a_i%2520%253D%2520%255Cphi(z_i)?scale=1)
  - Common choices include: sigmoid, tanh, linear, ReLU, LeakyReLU...
  - The activation function of the output layer needs to be chosen with particular care to suit the problem. (ie, not choosing sigmoid for a regression problem where the network outputs need to be greater than 1)
  - If we use a **linear** activation function for our hidden layers it is equivalent to not having any hidden layers! We need non-linearity!
  - Linear activation still has a place though, it is often used for the output layer in regression problems (if the range of the output needs to be all real numbers ![LaTeX: \mathbb{R}](https://ncf.instructure.com/equation_images/%255Cmathbb%257BR%257D?scale=1))

- Gradient Descent requires that we compute derivatives of the loss with respect to each parameter
  - Back-propagation is how we efficiently find these derivatives
  - This shows us the critical importance of the derivative of the activation function:
    - It must not be 0 all the time or the network will never learn: This means we can't use a "step" activation function
    - It should be easy to compute: This motivates ReLU

- [https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning)**[https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning](https://stanford.edu/~shervine/teaching/cs-229/cheatsheet-deep-learning)**

**Lecture 16** (Mar 28):

- Gradient Descent
  - Perform updates to the parameters: ![LaTeX: w_i^\text{new} = w_i^\text{old} - \alpha \frac{\partial \mathscr L}{\partial w_i}](https://ncf.instructure.com/equation_images/w_i%255E%255Ctext%257Bnew%257D%2520%253D%2520w_i%255E%255Ctext%257Bold%257D%2520-%2520%255Calpha%2520%255Cfrac%257B%255Cpartial%2520%255Cmathscr%2520L%257D%257B%255Cpartial%2520w_i%257D?scale=1)
  - Computing the gradient of the loss function which involves taking a sum over the whole dataset. Full-batch gradient descent does this.
    - This is computationally expensive but very stable and converges more quickly
  - In Pure Stochastic Gradient Descent (SGD) you update only evaluate the loss of using one sample at a time.
    - This is computationally very cheap but is less stable and does not converge quickly (or at all)
  - Stability is not always desirable, as it can mean we get suck in local minima more easily
  - Dealing with lots of data with Batches
    - Mini-batches

- MLPs
  - What defines them?
    - Architecture
    - Activation Functions
    - Parameter values (weights and biases)
      - This is the secret sauce! We discussed the leak of Meta's LLaMA model which Meta likely spent on the order of $1,000,000 training.
    - Formulating MLPs with Matrix Multiplication
  - Counting parameters: one weight for each connection between neurons and one bias for each neuron.

- Review

**Lecture 17** (Apr 2):

**Midterm Exam**

**Lecture 18** (Apr 4):

- Exam Post-Mortem
  - [Analysis of scores](https://colab.research.google.com/drive/1awF3HspZ3ACz4K3oV7WeXhbwvEE9Ekdm?usp=sharing)

- Suggested reading ("Guessing the Teacher's Password", or what it means to actually learn something): [https://www.lesswrong.com/posts/NMoLJuDJEms7Ku9XS/guessing-the

-teacher-s-password](https://www.lesswrong.com/posts/NMoLJuDJEms7Ku9XS/guessing-the-teacher-s-password)

**Lecture 19** (Apr 9):

- Multi-Class Classification
  - Softmax output
  - Use Cross Entropy

**Lecture 20** (Apr 11):

- Hands on creation and training of Neural Networks

**Lecture 21** (Apr 16):

- Convolutional Neural Networks
  - Conv2D Layer
  - Max Pooling Layer

- Excellent "cheat sheet" resource: [https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks)

![image.png](https://ncf.instructure.com/courses/8127/files/774219/preview)

Architecture of "AlexNET"

### **Unsupervised Learning:**

**Lecture 22** (Apr 18):

- Regularization Techniques
  - L2 Regularization
  - Model Checkpoint
  - Early Stopping
  - Dropout (Original paper: [https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf))
  - Batch Normalization (Original paper: [https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167))
  - Different weight initialization schemes
  - Data Augmentation
  - Ensemble Methods (Bagging)

- Dimensionality Reduction
  - Reasons for doing it include:
    - Visualization
    - Feature Selection
    - Noise Reduction
    - Cluster Analysis
    - Compression
    - Interpretability
    - Anomaly Detection
    - **Manifold Learning**
    - etc...
  - Some Methods for Dimensionality Reduction
    - Principle Component Analysis (PCA)
    - Autoencoders
    - Ones we didn't have time to talk about include:
      - t-SNE
      - Linear Discriminant Analysis (LDA)
      - Singular Value Decomposition (SVD)
      - Non-negative Matrix Factorization (NMF)
  - Note: All of these techniques (except Autoencoders) are basically just **linear algebra** ideas which can be adapted for dimensionality reduction

- Use of Autoencoders for De-noising
- Use of Autoencoders for Anomaly Detections

#### **Generative Modeling**

- Use of the Encoder from an Autoencoder for Generating new data
  - This is our first real example of Generative AI
  - Compared to thispersondoesnotexist.com (which uses a GAN not an Autoencoder)
  - A more common approach is to use a variant of Autoencoders called a Variational Autoencoder (VAE).
    - Learn more about VAEs here: [https://www.youtube.com/watch?v=9zKuYvjFFS8&t=447s](https://www.youtube.com/watch?v=9zKuYvjFFS8&t=447s)

![image.png](https://ncf.instructure.com/courses/8127/files/774264/preview)

[I made this picture :)](https://www.mdpi.com/2073-8994/15/7/1352)

**Lecture 23** (Apr 23):

**Clustering:**

- **K Means Clustering**
- **DB-Scan**

**Lecture 24** (Apr 25):

- Review of HW5 submitted models
  - Effect of Model Scale
  - Image Augmentation
  - Improvement upon Fully Connected network (MLP) baseline
  - Discussion of L1 and L2 normalization

- Brief discussion of Generative Adversarial Networks (GANs):
  - Uses an NN as a generator of images, and then uses another NN trained to discriminate real from fake images as part of its loss function
  - I recommend watching this video to learn some more advanced ideas about GANs: [https://www.youtube.com/watch?v=dCKbRCUyop8&t=1067s](https://www.youtube.com/watch?v=dCKbRCUyop8&t=1067s)

**Tentative future schedule:**

**Lecture 25** (Apr 30):

- **Bayes' Theorem**
  - **Prior and Posterior Probabilities**
  - Using Bayes' Theorem to "update" prior beliefs and obtain "posterior" probabilities
  - Example of a test for a disease
  - Implications for Human Rationality: Bayesian Rationality movement (see: [lesswrong.com](http://lesswrong.com

))
  - The simple "odds formulation" of Bayes' Theorem:
    - **Posterior Odds = "Likelihood Ratio" x Prior Odds**

- **Naive Bayes Classifiers**
  - If we assume each feature is independent we can treat them as separate pieces of evidence and use the math of multiple updates
  - Example of a spam filter
  - Gaussian Naive Bayes

**Lecture 26** (May 2):

- **Gaussian Mixture Models**
- Unsupervised Learning Wrap Up
- **Natural Language Processing**

**Lecture 27** (May 7):

**Quiz 3**

- LLMs
  - How they work: Next Token Prediction
  - Transformers
  - Context size, temperature, and other ideas
  - RLHF
  - Prompting

- Foundation Models

**Lecture 28** (May 9): (Last Lecture)

- How language models work:
  - Word embeddings as trainable parameters
  - We represent language as a sequence of vectors, one for each word (or technically for each token)
  - Early "Neural Language Models" date back to 1991
  - These used Recurrent Neural Networks (RNNs)
    - In RNNs some neurons connect back upon themselves in successive time steps
    - Each word has its embedding fed in to the network sequentially and they are **trained to predict the next word** in a process known as "Self Supervised Learning"
    - We discussed the "Vanishing and Exploding Gradients" problem which effectively limits the length of sequence which the networks can process
    - Attempts to improve upon RNNs include LSTMs and GRUs
  - In 2017 a team at Google Brain invented the "Transformer" in their landmark paper: [Attention is All You Need](https://arxiv.org/pdf/1706.03762v7)
  - Transformers radically improve upon the previous state of the art, and at their hear is the "Attention Mechanism"
  - In an "Attention Head" each word's embedding is mapped to three vectors:
    - **Query**: Which essentially "asks a question" in the form of a vector
    - **Key**: Which says "I am relevant to these questions" in the form of a vector
    - **Value**: Which gives an "answer to the question" in the form of another vector
  - We can explore how this works in an example: Imagine the sentence "A fuzzy creature roamed the verdant forest"
    - One attention head may do something like this:
      - All nouns produce a **Query:** _if there are any adjectives preceding them,_ 
      - all adjectives produce a **Key: **_I am an adjective!_ 
      - and all adjectives produce a **Value:** _Here is what I describe_
    - Lets focus on what happens to the embedding of "creature" in this attention head:
    - creature's **Query** should line up with fuzzy's **Key**, in the sense that they have a large dot product.
    - We say "creature" will _attend to_ "fuzzy", and it will add part of the **Value** of "fuzzy" to its own embedding, _enriching_ its meaning.
    - Attention is All You Need also introduced a method for "positional encoding" which augments the embedding vectors with information about where they appear in the sequence. In an RNN this information is contained in the order in which you feed the words in. In a Transformer the whole sequence is processed in parallel, so this information would otherwise be lost.
    - This is needed for the operation we just described, as part of the query will "ask" about relative distance by including a linear transformation of its own positional encoding representing a certain shift, and the key will learn to answer with its own positional encoding.
  - Transformers combine many of these attention heads in parallel to form a "multi-headed attention" block, and alternate those blocks with MLP layers and stack them a large number of times.
    - In this fashion the network can learn complicated relationships which require comparisons of many different parts of the input sequence.
    - Note that long-range dependences are effortless for the transformer to model as inside the attention head each position in the sequence is able to attend to any other position.
    - This means however that the complexity scales with the length of the sequence, which naturally limits the longest sequences our models can process; this is known as the "context size" of the model. Earlier GPT-3 models had context sizes of ~2k tokens (about half the length of this entire page^) and thus would eventually forget the beginnings of long conversations. Pushing the models to larger scale has dramatically increased the context size of the state of the art: GPT-4 now supports up to 128k tokens, which is roughly 240 pages of text.
