# E-commerce Product Classification

This project focuses on building a multi-class text classification model to predict product categories from their descriptions. The dataset used includes textual data about products, and the goal is to classify them into predefined categories effectively.

## Codebase Structure
- **Notebooks:** This is the folder containing all the Jupyter Notebooks that have been used for Exploratory Data Analysis, training and testing of the Machine Learning and Deep Learning models.
- **requirements.txt:** This file contains all the dependencies of the project that are needed to reproduce the development environment.
- **Dataset:** This folder contains all the datasets (imbalanced and balanced) in CSV format.
- **Presentation:** This folder contains the presentation which shows all the observations and conclusions made while working on the project.

## Introduction
Classifying e-commerce products is critical for improving user experience and enabling better product recommendations. In the modern eCommerce landscape, accurate product categorization is essential to enhance customer experiences, streamline product discoverability, and drive business growth. However, managing and categorizing thousands of products across diverse categories presents significant challenges, including ambiguities in product descriptions, unconventional naming conventions, and multilingual data.

This project aims to develop a robust machine learning and deep learning-based text classification model to categorize eCommerce products accurately. By leveraging product descriptions, we will transform raw text into meaningful features and build predictive models capable of handling the intricacies of the dataset. The solution will enable seamless integration into eCommerce platforms, ensuring scalability, efficiency, and enhanced customer satisfaction. This project tackles the challenge by employing various machine learning and deep learning algorithms, including logistic regression, random forests, and LSTM, for accurate classification.

## Dataset
The dataset contains the following columns:
**1. uniq id:**
-Description: A unique identifier for each product.
Purpose: Acts as the primary key to distinguish each product record uniquely.
**2. crawl timestamp:**
-Description: The timestamp when the product data was last scraped or collected.
Purpose: Helps identify the data's recency and track changes over time.
**3. product url:**
-Description: The URL linking directly to the product's page on the eCommerce platform.
Purpose: Allows direct access to the product's information and purchasing page.
**4. product_name:**
Description: The name or title of the product as displayed on the eCommerce platform.
- Purpose: Provides a searchable and readable identification of the product.
**5. product_category_tree:**
Description: The hierarchical structure representing the product's category on the platform.
-Purpose: Useful for categorization, analysis, and filtering of products.
**6. pid:**
specific - Description: A unique identifier specific to the eCommerce platform for each product.
-Purpose: Used to reference products internally on the platform.
**7. retail price:**
Description. The original or retail price of the product before any discounts.
Purpose: Helps understand the product's standard market value.
**8. discounted price:**
Description: The price of the product after applying any discounts or offers.
-Purpose: Reflects the final price a customer would pay.
**9. image:**
- Description: URL linking to the main image of the product.
- Purpose: Provides visual representation for the product.
**10. is FK Advantage_product:**
- Description: A boolean indicator (True/False) showing if the product is part of Advantage program.
- Purpose: Denotes if the product has additional benefits like faster delivery or special quality checks.
**11. description:**
-Description: Detailed information about the product, including features, specifications, and usage.
-Purpose: Helps customers understand the product's value proposition and unique selling points.
**12. product_rating:**
-Description: The product's overall rating on the platform, based on customer reviews.
-Purpose: Indicates customer satisfaction and product quality.
**13. overall_rating:**
Description: The aggregate rating of the product across different platforms ar periods.
-Purpose: Offers a comprehensive view of the product's reception.
**14. brand:**
-Description: The name of the brand or manufacturer of the product.
-Purpose: Assists in brand-based analysis and filtering.
**15. product specifications:**
Description: Detailed specifications of the product, often in JSON or structured format

## Approach
- **The following tasks were undertaken for the Multiclass Classification of e-commerce products based on their description:**
1. The dataset and several of its hidden parameters were visualised (using libraries like seaborn, matplotlib, yellowbrick, etc). This then helped in data cleaning as several words from the Word Cloud were removed from the corpus as they did not contribute much in terms of Product Classification.
2. It was decided to move forward by only using the root of the Product Category Tree as the Primary label/category for classification.
3. Data cleaning, preprocessing and resampling was then performed to balance out the given dataset.
4. After a detailed analysis of the dataset through visualisation and other parameters, it was decided to categorise the products in the following 13 categories and remove the noise (other miscellaneous categories having less than 10 products):
   - Clothing
   - Jewellery
   - Sports & Fitness
   - Electronics
   - Babycare
   - Home Furnishing & Kitchen
   - Personal Accessories
   - Automotive
   - Pet Supplies
   - Tools & Hardware
   - Ebooks
   - Toys & School Supplies
   - Footwear

5. Then, the following Machine Learning algorithms (using scikit-learn libraries) were applied on the dataset:
- Logistic Regression
- Random Forest
- SVM
- Naive Bayes
- Gradient Boosting (XGBoost)
- Deep Learning (LSTM)

## **STEP 1: Data Cleaning**
- Handled missing values.
- Removed noise and irrelevant details from product descriptions.
- **Text Vectorization:**
       - Used TF-IDF to transform text into numerical format.
## **STEP 2: Exploratory Data Analysis and Data Preprocessing**
- An in depth analysis of the dataset was done with the help of Word Clouds, Bar Graphs, TSNE Visualizations, etc to get an idea about the most frequent unigrams in the Product Description, distribution of products and brands across the different Product Categories, analysis of the length of the description, etc.
- For Data Cleaning, Contraction Mapping, removal of custom stopwords, URLs, Tokenization and Lemmatization was done.
- Because of the clear imbalance in the dataset, balancing techniques like Oversampling and Undersampling were performed on the dataset as well. These were then saved in the form of a CSV file.
## **STEP 3: Machine Learning Models for Product Categorization**
- The above mentioned 6 ML algorithms were applied on the imbalanced, oversampling balanced and undersampling balanced datasets. Noise was removed from each of these datasets and these datasets had already been cleaned and preprocessed in the previous notebook.
- Several evaluation metrics like Classification Report, Confusion Matix, Accuracy Score, ROC Curves and AUC Scores were used for the comparison of the models. The Validation score of the ML algorithms when applied on the dataset are tabulated below:
- ML Algorithm	Validation Accuracy on Imbalanced Dataset	Validation Accuracy on Balanced Dataset (Oversampling).

## Models Used
- **Logistic Regression:** A baseline model for text classification.
- **Random Forest:** Ensemble method for better generalization.
- **SVM:** For high-dimensional text classification.
- **Naive Bayes:** Leveraging the probabilistic nature of text data.
- **XGBoost:** Gradient boosting for better predictive power.
- **LSTM:** Deep learning approach to capture sequence dependencies in text.
From the above table, we can clearly see that Linear Support Vector Machine algorithm performed the best across all the three datasets.
## **STEP 4: Deep Learning Models for Product Categorization**
- The Deep Learning Models were only trained and evaluated on the dataset that was balanced using the Oversampling technique.
## **STEP 5: Evaluation:**
- Models were evaluated using metrics like accuracy, precision, recall, and F1-score.
- Fine-tuned models to improve performance.

**Confusion Matrix of Linear Regression Model**
![image](https://github.com/user-attachments/assets/56c130d1-97cf-4336-ab52-93315f4660b1)


## Technologies Used
- **Python:** For preprocessing, modeling, and evaluation.
- **Libraries:**
    Pandas, NumPy, Matplotlib, Seaborn (EDA)
    Scikit-learn, XGBoost (ML models)
    TensorFlow/Keras (LSTM)
    Imbalanced-learn (SMOTE)
- **GitHub:** Version control and collaboration.

## **Future Work**
- Feature extraction can be performed on the Product Category Tree column in order to find a more detailed class to which a product can belong.
- Using other advanced data balancing techniques like RandomOverSampling, etc.
- Training and evaluating the Deep Learning model on datasets other than the undersampled one. These models could then be tested on a variety of e-commerce data available online to understand the scalability of the model when it comes to dealing with real-world data.
- Using Named Entity Recognition techniques to figure out brands that make products belonging to a specific category.
