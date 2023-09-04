# AI-CA3-Categorize-Articles-of-Digitala-Mag-Using-Classifier-Bayes-Naive

## Project No. 3: Working with Naive Bayes Classifier

### Introduction
In this project, our goal is to analyze articles from the Digitala Mag site using the Naive Bayes Classifier, categorize them, and predict their classification based on the description of each article.

### Import Libraries and Define Constants
We import 'hazm' for data preprocessing and use the 'stop_words' function from 'farsi_tools' to ignore Persian stop words.

### Phase One: Data Preprocessing
We use 'sent_tokenize' to separate sentences and 'word_tokenize' for word tokenization. We choose 'lemmatizer' for getting past and present roots of verbs since it's more accurate than stemming.

**Stop Words:**  
Stop words are generated from 'chars.txt' and the 'stop_words()' function in 'farsi_tools'. We create a set of stop words.

### Phase Two: Problem Process
We add an alpha value to the formula to avoid zero probabilities for words that may not appear in the training set but appear in the test set. We use the formula: `np.log((LABEL.get(token, 0) + alpha) / (LABEL_total_words + (alpha * LABEL_distinct_words)))` instead of `np.log(LABEL[token] / LABEL_total_words)`, with alpha set to 1.

**Train the Model:**  
We create dictionaries for each category (e.g., art_and_cinema, science_and_tech) to count the occurrence of tokens in each category.

**Posterior:** The probability of a test article with word count evidence calculated in the training set.

**Likelihood:** The probability of a word appearing in respective training articles.

**Class Prior:** The probability of an article belonging to a category.

**Predictor Prior:** The probability of words appearing in articles calculated according to every article in the training set.

### Bigrams
Bigrams consider the words before and after the current word, helping to differentiate between words with multiple meanings. For example, 'سیر' can mean 'not hungry' or 'garlic.' Bigrams can recognize phrases like 'سیر خراب' and determine the intended meaning based on context.

### Additive Smoothing
Without additive smoothing, if a word is only present in articles related to one category (e.g., video games) and not in other categories (e.g., science and technology), the model may incorrectly classify any article containing that word as related to video games. Additive smoothing helps avoid zero probabilities and biases, making the model more robust.

We add an alpha value to the formula to avoid zero probabilities for words not in the training set but present in the test set. The formula becomes: `np.log((LABEL.get(token, 0) + alpha) / (LABEL_total_words + (alpha * LABEL_distinct_words)))`.

### Phase Three: Evaluation 
Precision and Recall are not enough because they focus on different aspects of model performance.

- Precision measures what proportion of predicted positives are truly positive. It is essential when false positives are costly. High precision means fewer false positives.
- Recall measures what proportion of actual positives is correctly classified. It is crucial when false negatives are costly. High recall means fewer false negatives.

For example:
- In a medical test, high recall ensures that actual patients are correctly identified. However, high precision ensures that healthy individuals are not wrongly diagnosed.
- Conversely, in spam email detection, high precision ensures that legitimate emails are not classified as spam, while high recall ensures that most spam emails are detected.

Both precision and recall are necessary to assess a model comprehensively, considering the specific problem and its associated costs.

**Probabilities without Additive Smoothing:**
- **Precision:**
    - هنر و سینما: 95.35%
    - سلامت و زیبایی: 95.03%
    - علم و تکنولوژی: 95.36%
    - بازی ویدیویی: 97.88%
- **Recall:**
    - هنر و سینما: 98.20%
    - سلامت و زیبایی: 95.03%
    - علم و تکنولوژی: 96.39%
    - بازی ویدیویی: 93.91%
- **F1-score:**
    - هنر و سینما: 96.76%
    - سلامت و زیبایی: 95.03%
    - علم و تکنولوژی: 95.87%
    - بازی ویدیویی: 95.85%
