# WaterQualityPrediction_MLProject

<h2 align="center" ><font color='blue'> TOPIC : Water Quality</font></h2>
<h4 align="center"> (Drinking water potability)</h4>

<H3 align="right" ><font color='brown'>Author Name : SUSMY P B</font></H3>
<H3 align="right"><font color='brown'>Organization : Entri Elevate</font></H3>
<H3 align="right"><font color='brown'>Date : 01/07/2024</font></H3>

## TABLE OF CONTENTS
<OL>
    <LI>OVERVIEW OF PROBLEM STATEMENT</LI>
     <LI>OBJECTIVE</LI>
     <LI>DATA COLLECTION</LI>
     <LI>DATA DESCRIPTION</LI>
     <LI>EXPLORATORY DATA ANALYSIS</LI>
    <LI>DATA PREPROCESSING</LI>
    <LI>VISUALIZATION</LI>
    <LI>FEATURE ENGINEERING</LI>
    <LI>DATA SPLITTING</LI>
    <LI>MODEL SELECTION</LI>
    <LI>FEATURE SELECTION</LI>
     <LI>MODEL TRAINING</LI>
      <LI>MODEL EVALUATION</LI>
       <LI>HYPERPARAMETER TUNNING</LI>
        <LI>RESULT</LI>
         <LI>MODEL DEPLOYMENT</LI>
         <LI>LIMITATIONS</LI>
         <LI>CONCLUSION</LI>
         <LI>FUTURE WORK</LI> 
    
</OL>

## 1. OVERVIEW OF PROBLEM STATEMENT

<P>An overview of water quality prediction involves understanding the methods, techniques, and objectives related to assessing and forecasting the quality of water in various environments.</P>
<P>Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.</P>

## 2. OBJECTIVE
<UL>
<LI><B>Purpose</B>: The primary goal is to predict or assess the quality of water, ensuring it meets regulatory standards, supports aquatic life, and is safe for human consumption.</LI>
<LI><B>Scope</B>: The scope may vary from local water bodies (e.g., rivers, lakes) to larger ecosystems or industrial discharge points.</LI>
    <LI><B>Prediction and Assessment</B>: The primary objective is to predict and assess the quality of water based on various parameters and indicators.</LI>
    </UL>
<p>Develop a predictive model to accurately estimate water quality.Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable. 
</p>

## 3. DATA COLLECTION
<p>Uses the read_csv function from pandas to read the CSV file located at the specified path </p>

## 4. DATA DESCRIPTION

<p>DataSet : <a href="https://www.kaggle.com/datasets/adityakadiwal/water-potability/data" style="text-decoration: underline; color: blue;">Link to Water Potability Dataset on Kaggle</a>
</p>

<p>The water_potability.csv file contains water quality metrics for 3276 different water bodies.</p>
<ol>
<li><b>pH value</b>:
PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5. The current investigation ranges were 6.52–6.83 which are in the range of WHO standards.</li>

<li><b>Hardness</b>:
Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.</li>

<li><b>Solids (Total dissolved solids - TDS)</b>:
Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.</li>

<li><b> Chloramines</b>:
Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.</li>

<li><b>Sulfate</b>:
Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.</li>

<li><b> Conductivity</b>:
Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.</li>

<li><b>Organic_carbon</b>:
Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.</li>

<li><b>Trihalomethanes</b>:
THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.</li>

<li><b>Turbidity</b>:
The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.</li>

<li><b>Potability</b>:
Indicates if water is safe for human consumption where 1 means Potable and 0 means Not potable.</li>
</ol>

## 5. EDA (EXPLORATORY DATA ANALYSIS)

<p>There are 3276 rows and 10 columns in the dataset.</p>

## Identify numerical and categorical columns
<p>For numerical columns,</p>

<i>num_cols = df.select_dtypes(include='number').columns</i>
<i>print(num_cols)</i>

<P>->This pandas DataFrame method, <B>select_dtypes()</B>, is used to select columns based on their data types.</P>
<P>->The argument <B>include='number'</B> specifies that you want to select columns with numeric data types (float64 and int64).</P>
<P>->This method returns a DataFrame containing only the columns that match the specified data types.</P>

<p>For categorical columns,</p>

<i>cat_cols = df.select_dtypes(include='object').columns</i>
<i>print(cat_cols )</i>

<P>-> <B>select_dtypes()</B> is a pandas DataFrame method used to select columns based on their data types.</P>
<P>-> The argument <B>include='object'</B> specifies that you want to select columns with object data types. In pandas, object data type corresponds to string or categorical data.</P>
<P>-> This method returns a DataFrame containing only the columns that match the specified data types.</P>

<p>After the findings, Out of 10 columns, 0 are categorical and 10 are numerical,There is <b>no categorical columns</b> in it.</p>

## Checking for null values

<p> isnull() is a pandas DataFrame method that returns a boolean DataFrame of the same shape as df, where each element is True if the corresponding element in df is NaN, and False otherwise.</p>
<p> After applying df.isnull(), .sum() is used to sum up the True values for each column, because in Python, True is interpreted as 1 and False as 0 when summing.</p>

<p><b></b>In our data,we can find ph,Sulfate,Trihalomethanes of isnull().sum() is greather than zero.</b></p>

<p><b>Note</b> : If percentage of missing values is greater than 50%, we can drop that columns.</p>

<p>we handle the null values using <b>fillna()</b> method. df.fillna(...) is a pandas DataFrame method used to fill NaN (missing) values in the DataFrame.
</p>

## Checking for Duplicates

<p><B>There are no duplicate values in our data set.</B></p>

## Understand the distribution of data

<h3>Descriptive statistics</h3> 
<i>df.describe().T</i> 
<p> describe() is a pandas DataFrame method that generates descriptive statistics of numerical (numeric) columns in the DataFrame.</p>

<p>Here we plot <b>histograms</b> for the graphical representations of the distribution of numerical data.</p>
<h5><b>Histograms</b> are graphical representations of the distribution of numerical data. They are useful for understanding the shape, spread, and central tendency of a dataset.Histograms are versatile tools for exploring and visualizing the distribution of numerical data. They provide a quick visual summary that helps in understanding the nature of data, identifying outliers, and making initial assessments about its statistical properties.</h5>
<p>After draw the histogram, we can see Skewness in the feature Solids.It has <b>right skewness</b>.And Conductivity has slightly skewness in it.</p>
<p><b>Skewness</b> refers to the measure of asymmetry in the distribution of data points. There are three types of skewness:</p>
<p>1.Positive Skewness (Right Skewness)</p>
<p>2.Negative Skewness (Left Skewness)</p>
<p>3.Zero Skewness (Symmetric Distribution)</p>

<p>Then we plot <b> Histogram of a specific column (e.g., pH)</b>.sns.histplot(): This function from Seaborn is used to create a histogram (or KDE plot) of the 'ph' column in your DataFrame df.
</p>
<P>pH is the measure of hydrogen ions in water-based liquids. The pH scale shows the leaves of acidity and alkalinity of water and similar liquids.</P>
<P>In its purest form, water has a pH of 7, which is at the exact center of the pH scale. Particles in the water can change the pH of the water, and most water for use has a pH of somewhere between 6.5 and 8.5.</P>

<p>Then we plot <b> boxplot</b>, for the current numerical column (col) from the DataFrame df.</p>
<h4><b>Outliers</b>: By default, Matplotlib's boxplot() function will mark outliers as individual points beyond the whiskers of each box plot. These points are typically identified based on the interquartile range (IQR) method (1.5 times the IQR from Q1 and Q3).</h4>
<p>There is outlier in solids data.A question that,should we remove it or not?</p>
<p>Ans : It's upto you.That is a experimental thing.</p>
<p>If you don't remove it,the excess solids leads bad quality in water.If you remove,then we get the good water only.If we remove all bad elements your Data Science model will not good.</p>
<b>If we not removing the outliers, they may be important to decide the quality of water.But we removing the outliers.</b>

<p>Then we plot scatter plot to identify the relationship between dependent (Portability) and independent features. sns.scatterplot() is used here to create scatter plots, which are useful for visualizing the relationship between two numerical variables (col and 'Potability' in this case).</p>

## Checking if we need to dimensionality reduction

<p><b><i>Dimensionality reduction</i></b> refers to the process of reducing the number of random variables under consideration, either by selecting a subset of variables or by transforming them into a smaller set of variables. This is particularly useful in machine learning and data analysis for several reasons:</p>
<ol>
<li><b>Curse of Dimensionality</b>: High-dimensional data often suffers from the curse of dimensionality, where the algorithms become less efficient and more prone to overfitting.</li>

<li><b>Computational Efficiency</b>: Reducing the number of features can lead to faster training and testing of machine learning models.</li>

<li><b>Visualization</b>: It is difficult to visualize data in more than three dimensions. Dimensionality reduction helps in visualizing the data in a lower-dimensional space.</li></ol>

<p><b>Check correlation of each features</b></p>
<p>'1' denote 100% correlation.If two variables are coorelated more than 75 or 85 percentage,we can remove one of colum.</p>
<p>In the above matrix,there is no correlation between any varibles.So don't remove any columns.All columns are mandatory</p>

## Skewness and Curtosis
<p><b><i>Skewness</i></b>: Skewness measures the asymmetry of the probability distribution of a real-valued random variable about its mean. In simpler terms, it indicates whether the data is skewed to the left (negative skew) or to the right (positive skew) relative to a normal distribution.</p>
<br/>
<p><b><i>Kurtosis </i></b>:It is a statistical measure that describes the shape of the probability distribution of a real-valued random variable. Specifically, kurtosis quantifies whether the tails of the distribution are heavy (i.e., contain outliers) or light (i.e., lack outliers) relative to a normal distribution.</p>
<h5>Notes :</h5>
<ul><li>
    Ensure that <b>num_cols</b> contains the names of columns that contain numerical data.</li>
<li>Adjust the threshold values (<b>1</b> and <b>-1</b> for skewness, <b>3</b> for kurtosis) according to your specific analysis requirements.</li>
<li>The <b>lambda x: kurtosis(x, fisher=False)</b> syntax is used to calculate kurtosis with the <b>fisher=False</b> parameter to ensure the result matches the definition where kurtosis of a normal distribution is 3.
</li></ul>

## 6. DATA PREPROCESSING
## Remove unnecessary columns
 <H5><B>Columns with multicolinearity</B> :When we create correlation matrix,there is no relation between any columns.There is no columns with multicolinearity</H5>

<p><b>Multicollinearity</b> refers to the phenomenon in which two or more predictor variables in a regression model are highly correlated with each other. This can cause issues in the model estimation because it undermines the statistical significance of individual predictors.</p>

## Handling missing values
<H5>For the missing value treatment we can use mean, median, mode or KNNimputer. KNNImputer is a method used
for imputing missing values by using the k-nearest neighbors approach.<H5>
<ul>
    <li><b>Mean</b> : Replace missing values with the mean of the available values for that feature.</li>
        <li><b>Median</b> : Replace missing values with the median of the available values for that feature.</li>
        <li><b>Mode</b> : Replace missing categorical values with the mode (most frequent value) of the available values for that feature.</li>
        <li><b>KNNImputer</b> : Impute missing values based on the values of the nearest neighbors in the feature space. The imputation is done using the weighted or unweighted average of the values of k nearest neighbors.</li> 
</ul>

<p>We can find,there is no null values in it.</p>
<p>Note : </p>
<p><b><i>impute</i></b> creates an instance of the <b>KNNImputer</b> class. By default, <b>KNNImputer</b> uses a Euclidean distance metric to find nearest neighbors and impute missing values based on those neighbors.</p>
<p><i>imputer.fit_transform(df[[col]])</i> fits the KNNImputer on the column col and transforms it, replacing missing values with imputed values.</p>

## Handling Outliers

#### Here IQR method is used for outlier treatment
<ul><li>Calculates the quartiles (Q1 and Q3) and the Interquartile Range (IQR) for each column.</li>
<li>Defines lower and upper bounds for outlier detection based on the IQR method (typically Q1−1.5×IQR and 𝑄3+1.5×IQR).</li>
<li>Identifies outliers based on these bounds.</li>
<li>Replaces outliers with None (NaN) for this example. You can modify this part to suit your specific handling strategy (e.g., removing outliers, replacing with another value, etc.).</li>
</li></ul>

<p>The <b>whisker</b> function you've defined calculates the lower and upper whiskers for a boxplot, based on the interquartile range (IQR)</p>
<p>These <b>percentiles</b>  are used to determine the quartiles in the boxplot.</p>
<p>The <b>IQR</b> is a measure of statistical dispersion and is used to determine outliers.</p>

<p>Note : <b>np.where()</b> to replace outliers</p>

<p>After removing the outliers we have plot a boxplot.From that plot we can see that ,there is no outliers in it.</p>
<h4>All the outliers are removed.</h4>

## 7. VISUALIZATION
<p>1.<b>Histograms</b> for numerical columns after outlier treatment</p>
<p>2.<b>Pair plot</b> - To visualize relationships between multiple pairs of variables.<b>pairplot()</b> function, is a grid of plots that displays pairwise relationships among a set of variables. Each variable in the dataset is paired with every other variable, resulting in a matrix of scatterplots where variables along the diagonal are plotted against themselves (showing distributions), and variables off the diagonal are plotted against each other (showing relationships).</p>

<i>df['Potability'].value_counts()</i>
<p>Above displays the count of the class variable which is namely potability.The dataset contains <b>1998</b> number of 0s which implies <i>contaminated water</i> and <b>1278</b> number of 1s which implies <i>potable water</i></p>

<p>3.<b>Count plot </b>:<i>sns.countplot(data=df, x=df['Potability'], color='skyblue', edgecolor='black') </i> It simplifies the process of creating count plots by directly accepting the DataFrame and column name.</p>

<p>4.<b>Bar plot</b> :A bar plot is a common type of visualization used to display categorical data. It represents categorical data with rectangular bars, where the length of each bar is proportional to the count or frequency of a category in the data. </p>

<p>4.<b>Box plot</b> :A box plot, also known as a box-and-whisker plot, is a standardized way of displaying the distribution of data based on a five-number summary: minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum. It is particularly useful for visualizing the spread and skewness of the data along with identifying potential outliers. </p>


<p>Notes :</p>
<ul><li>
<b>Log-Normal Distribution</b>: The np.random.lognormal() function generates values that are log-normally distributed, which means the logarithm of the data follows a normal distribution.</li>

<li><b>Log Transformation</b>: Applying np.log() transforms the skewed log-normal data into a more symmetric distribution suitable for certain statistical analyses.</li>

<li><b>Histogram</b>: Histograms are useful for visualizing the distribution of data, showing the frequency of values within specified bins.</li></ul>


<p>5.<b>Pie Chart</b> : A pie chart is a circular statistical graphic that is divided into slices to illustrate numerical proportions. Each slice represents a category's contribution to the whole, making it easy to visualize parts of a whole or percentages of total data. </p>


<p>6.<b>scatter plot</b>: A scatter plot is a type of plot or mathematical diagram using Cartesian coordinates to display values for typically two variables for a set of data. Each dot in the plot represents a single data point, where the position of the dot on the horizontal and vertical axes corresponds to the values of the two variables.</p>

<p>7.<b>Line plot</b>: A line plot or line chart is a type of plot that displays information as a series of data points called 'markers' connected by straight line segments. It is commonly used to visualize data trends over time or any other continuous variable. Line plots are particularly effective for showing how data changes in relation to a single continuous variable.</p>

<p>8.<b>Violin plot</b>: Violin Plot is a useful visualization tool for depicting the distribution of numeric data across different categories or groups. It combines aspects of a box plot (which shows summary statistics such as median, quartiles, and outliers) with a kernel density plot (which shows the probability density of the data at different values).</p>

## 8. FEATURE ENGINEERING
<H4>Encode categorical features to numerical using techniques like one-hot encoding or label encoding to prepare the
data for machine learning algorithms. Use Label encoding for ordinal data and one-hot encoding for nominal data.</H4>
<H4>In our dataset there is no categorical value.So we don't want to using techniques like one-hot encoding or label encoding</H4>

### 8.1 Feature Extraction
<H4>Feature extraction is a crucial step in machine learning for water quality prediction, where the goal is to select or derive the most relevant features (input variables) from raw data that contribute the most to predicting water quality metrics. Here’s a general approach to feature extraction for water quality prediction:</H4>

#### Steps for Feature Extraction
<ol><li><B>Understanding the Dataset:</B></li>
<ul>
<li>Data Exploration: Analyze the dataset to understand the available variables (columns) and their types (numeric, categorical, etc.).</li>
<li>Domain Knowledge: Gain insights into water quality factors that are typically measured and their significance.</li>
    </ul>
<li><B>Data Cleaning and Preprocessing:</B></li>
<ul>
<li>
Handle Missing Values: Address missing data through imputation (e.g., mean, median, mode) or removal.</li>
<li>Data Transformation: Normalize or scale numerical features, encode categorical variables, and handle outliers if necessary.</li></ul>
<li><B>Feature Selection:</B></li>
<ul>
<li>
Correlation Analysis: Identify correlations between features and the target variable (e.g., Potability).</li>
<li>Feature Importance: Use techniques like RandomForestRegressor or XGBoost to rank features by importance.</li>
<li>Dimensionality Reduction: Apply techniques such as Principal Component Analysis (PCA) or feature aggregation to reduce the number of variables while preserving relevant information.</li>
<li><B>Feature Engineering:</B></li></ul>
<ul>
<li>
Create New Features: Derive new features that might better capture relationships in the data (e.g., ratios, interactions between variables).</li>
<li>Temporal Features: If the dataset includes temporal information (e.g., timestamps), extract relevant time-based features (e.g., month, day of week).</li></ul>
<li><B>Selecting the Final Feature Set:</B></li>
<ul>
<li>
Subset Selection: Choose a subset of features that contribute most to predicting water quality.</li>
<li>Validation: Validate selected features using techniques like cross-validation to ensure robustness.</li></ul></ol>

### 8.2 Feature scaling and transformation

<P><B>Feature scaling and transformation</B> are essential preprocessing steps in data analysis and machine learning pipelines. These techniques help to normalize or standardize the features of a dataset, making them more suitable for certain algorithms or improving the interpretability of results.</P>
<OL TYPE="i"> 
<li><i>Feature scaling</i> ensures that all features have a similar scale. It is particularly important when features have different ranges of values, as this can negatively impact the performance of some machine learning algorithms. Two common techniques for feature scaling are:
    <ol><li>Min-Max Scaling (Normalization)</li>
    <li>Standardization</li></ol>
</li>
    <li><i>Feature transformation</i> refers to modifying the distribution or relationship of data features. Common transformations include:
    <ol><li>Log Transformation</li>
    <li>Box-Cox Transformation</li>
    <li>Polynomial Transformation</li></ol>
</li>
</P>

<p>Scale numerical features to ensure that they have the same magnitude, preventing some features from
dominating others during model training.</p>

<p>Note : Scaling using StandardScaler - The StandardScaler is a preprocessing step in machine learning that standardizes features by removing the mean and scaling to unit variance. It is essential when working with algorithms that assume normally distributed data or require features to be on the same scale.</p>

## 9. DATA SPLITTING
<p>Splitting the dataset into 80-20, that is, 80% of the data is for training and 20% of the data is for testing.Split the data into training and testing sets.</p>
<i>X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)</i>

<P>The parameters include above :</P>
<ol>
<li><b>X_scaled</b>: Scaled features (input variables).</li>
<li><b>y</b>: Target variable (output variable).</li>
<li><b>test_size</b>: Specifies the proportion of the dataset to include in the test split (here, 20% is used for testing).</li>
<li><b>random_state</b>: Controls the shuffling applied to the data before splitting, ensuring reproducibility of results.</li>
</ol>
<p>The output variables are :</p>
<ol>
    <li><b>X_train</b>: Training set of scaled features.</li>
<li><b>X_test</b>: Test set of scaled features.</li>
<li><b>y_train</b>: Training set of target variable.</li>
<li><b>y_test</b>: Test set of target variable.</li>
</ol>

## 10. MODEL SELECTION
<p>Models Selected:</p>
<ol><li>LogisticRegression</li>
   <li> DecisionTreeRegressor</li>
   <li> ExtraTreeRegressor</li>
  <li>  RandomForestRegressor</li>
<li>GradientBoostingRegressor</li>
  <li>  SVR</li>
   <li> MLPRegressor</li>
   <li> XGBRegressor</li></ol>

## 11. FEATURE SELECTION
### 11.1. SelecKBbest
<P><B>SelectKBest</B> is a method for univariate feature selection, which means it selects features based on univariate statistical tests. It operates under the assumption that the target variable is categorical (classification tasks) or numerical (regression tasks).</P>

### 11.2. Recursive Feature Elimination (RFE) with Random Forest Classifier
<P><B>Recursive Feature Elimination (RFE)</B> is another popular technique for feature selection, particularly useful when you have a large number of features. It works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target variable.</P>

## 12. MODEL TRAINING
<p><b>Train the models before feature selection</b></p>

### 12.1. LogisticRegression
<H4><B>Logistic regression</B> is a supervised machine learning algorithm used for classification tasks where the goal is to predict the probability that an instance belongs to a given class or not. Logistic regression is a statistical algorithm which analyze the relationship between two data factors. </H4>

### 12.2.DecisionTreeClassifier
<H4><B>Decision trees</B> are a popular and powerful tool used in various fields such as machine learning, data mining, and statistics. They provide a clear and intuitive way to make decisions based on data by modeling the relationships between different variables.</H4>

### 12.3.RandomForestClassifier
<H4><B>Random Forest algorithm</B> is a powerful tree learning technique in Machine Learning. It works by creating a number of Decision Trees during the training phase. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition. This randomness introduces variability among individual trees, reducing the risk of overfitting and improving overall prediction performance.</H4>

### 12.4.Support Vector Machine
<H4><B>Random Forest algorithm</B> is a powerful tree learning technique in Machine Learning. It works by creating a number of Decision Trees during the training phase. Each tree is constructed using a random subset of the data set to measure a random subset of features in each partition. This randomness introduces variability among individual trees, reducing the risk of overfitting and improving overall prediction performance.</H4>

### 12.5.XGBoost
<H4><B>XGBoost</B> is an optimized distributed gradient boosting library designed for efficient and scalable training of machine learning models. It is an ensemble learning method that combines the predictions of multiple weak models to produce a stronger prediction. XGBoost stands for “Extreme Gradient Boosting” and it has become one of the most popular and widely used machine learning algorithms due to its ability to handle large datasets and its ability to achieve state-of-the-art performance in many machine learning tasks such as classification and regression.</H4>

### 12.6.AdaBoost
<H4><B>AdaBoost</B> is one of the first boosting algorithms to have been introduced. It is mainly used for classification, and the base learner (the machine learning algorithm that is boosted) is usually a decision tree with only one level, also called as stumps. 
It makes use of weighted errors to build a strong classifier from a series of weak classifiers.</H4>

### 12.7.KNeighboursClassifier
<H4><B>KNN</B> is a simple, supervised machine learning (ML) algorithm that can be used for classification or regression tasks - and is also frequently used in missing value imputation. It is based on the idea that the observations closest to a given data point are the most "similar" observations in a data set, and we can therefore classify unforeseen points based on the values of the closest existing points. By choosing K, the user can select the number of nearby observations to use in the algorithm.</H4>

## 13. MODEL EVALUATION

<table>
  <tr>
    <th></th>
    <th>Model</th>
    <th>Accuracy Score</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 score</th>
  </tr>
  <tr>
    <td>0</td>
    <td>Logistic Regression</td>
    <td>0.493902 </td>
    <td>0.362500</td>
    <td>0.475410</td>
    <td>0.411348</td>
  </tr>
  <tr>
    <td>1</td>
    <td>Decision Tree</td>
    <td>0.570122	 </td>
    <td>0.435811	</td>
    <td>0.528689</td>
    <td>0.477778</td>
  </tr>
    <tr>
    <td>2</td>
    <td>Random Forest</td>
    <td>0.649390	 </td>
    <td>0.531818	</td>
    <td>0.479508</td>
    <td>0.504310</td>
  </tr>
    <tr>
    <td>3</td>
    <td>SVM</td>
    <td>0.644817	 </td>
    <td>0.520147	</td>
    <td>0.581967</td>
    <td>0.549323</td>
  </tr>
    <tr>
    <td>4</td>
    <td>XGBoost</td>
    <td>0.626524	 </td>
    <td>0.497942	</td>
    <td>0.495902</td>
    <td>0.496920</td>
  </tr>
     <tr>
    <td>5</td>
    <td>AdaBoost</td>
    <td>0.570122		 </td>
    <td>0.425781	</td>
    <td>0.446721</td>
    <td>0.436000</td>
  </tr>
     <tr>
    <td>6</td>
    <td>KNN</td>
    <td>0.620427		 </td>
    <td>0.491803	</td>
    <td>0.614754</td>
    <td>0.546448</td>
  </tr>
</table>


<OL><LI>The <B><I>Accuracy Score</I></B> is a metric used to evaluate classification models. It measures the proportion of correct predictions (both true positives and true negatives) out of all the predictions made by the model. In simpler terms, it tells us how often the model is correct.
<P><U>Accuracy= Total number of predictions/Number of correct predictions</U></P>
</LI>
<LI><B>Precision</B> is a metric used to evaluate the performance of a classification model, particularly in binary classification tasks. It measures the proportion of true positive predictions (correctly predicted positive cases) out of all positive predictions made by the model. Precision is useful when the cost of false positives is high or when you want to be confident about the predicted positive cases.
<P><U>Precision= True Positives/(True Positives+False Positives)</U></P>
</LI>
    <LI><B>Recall</B>, also known as Sensitivity or True Positive Rate, is another important metric used to evaluate the performance of a classification model, particularly in binary classification tasks. It measures the proportion of true positive predictions (correctly predicted positive cases) out of all actual positive cases in the dataset.
<P><U>Recall= True Positives/(True Positives+False Negatives)</U></P>
</LI>
</OL>

<p><b><U>Train the models with selected features using SelectKBest </U></b></p>


<table>
  <tr>
    <th>Model-Name</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 score</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>SVC</td>
    <td>0.493274 </td>
    <td>0.450820</td>
    <td>0.471092	</td>
    <td>0.623476</td>
  </tr>
  <tr> 
    <td>AdaBoostClassifier</td>
    <td>0.424342	 </td>
    <td>0.528689	</td>
    <td>0.470803</td>
    <td>0.557927</td>
  </tr>
    <tr> 
    <td>XGBClassifier</td>
    <td>0.440945	 </td>
    <td>0.459016	</td>
    <td>0.449799</td>
    <td>0.582317</td>
  </tr>
    <tr> 
    <td>DecisionTreeClassifier</td>
    <td>0.419355	 </td>
    <td>0.479508	</td>
    <td>0.447419</td>
    <td>0.559451</td>
  </tr>
    <tr> 
    <td>RandomForestClassifier</td>
    <td>0.472222	 </td>
    <td>0.418033	</td>
    <td>0.443478</td>
    <td>0.609756</td>
  </tr>
     <tr> 
    <td>LogisticRegression</td>
    <td>0.386581		 </td>
    <td>0.495902		</td>
    <td>0.434470</td>
    <td>0.519817</td>
  </tr>
     <tr> 
    <td>KNeighborsClassifier</td>
    <td>0.397163		 </td>
    <td>0.459016	</td>
    <td>0.425856</td>
    <td>0.539634</td>
  </tr>
</table>

<p><b><U>Train the models with selected features using Random Forest Regressor</U></b></p>


<table>
  <tr>
    <th>Model-Name</th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 score</th>
    <th>Accuracy</th>
  </tr>
  <tr> 
    <td>SVC</td>
    <td>0.567100 </td>
    <td>0.536885</td>
    <td>0.551579	</td>
    <td>0.675305</td>
  </tr>
   <tr> 
    <td>RandomForestClassifier</td>
    <td>0.541322	 </td>
    <td>0.536885	</td>
    <td>0.539095</td>
    <td>0.658537</td>
  </tr>
   <tr> 
    <td>KNeighborsClassifier</td>
    <td>0.501767		 </td>
    <td>0.581967	</td>
    <td>0.538899</td>
    <td>0.629573</td>
  </tr>
   <tr> 
    <td>XGBClassifier</td>
    <td>0.510040	 </td>
    <td>0.520492	</td>
    <td>0.515213</td>
    <td>0.635671</td>
  </tr>
   <tr> 
    <td>DecisionTreeClassifier</td>
    <td>0.456081	 </td>
    <td>0.553279	</td>
    <td>0.500000</td>
    <td>0.588415</td>
  </tr>
  <tr> 
    <td>AdaBoostClassifier</td>
    <td>0.438819	 </td>
    <td>0.426230	</td>
    <td>0.432432</td>
    <td>0.583841</td>
  </tr> 
     <tr> 
    <td>LogisticRegression</td>
    <td>0.350000		 </td>
    <td>0.430328		</td>
    <td>0.386029</td>
    <td>0.490854</td>
  </tr>
    
</table>

### 14. HYPER PARAMETER TUNNING

<H4><B>Hyperparameter tuning</B> is the process of selecting the optimal values for a machine learning model’s hyperparameters. Hyperparameters are settings that control the learning process of the model, such as the learning rate, the number of neurons in a neural network, or the kernel size in a support vector machine. The goal of hyperparameter tuning is to find the values that lead to the best performance on a given task.</H4>
<P>Techniques for Hyperparameter Tuning:</P>
<UL><LI>Manual Search</LI>
<LI>Grid Search</LI>
<LI>Random Search</LI>
<LI>Automated Hyperparameter Optimization</LI>
</UL>

## 15. RESULT
<H4>From our research, We test with all features before feature selection ,we have <b>Random Forest Classifier</b> has the highest Accuracy.</H4>
<H4>After feature Selection,<b>Random Forest Classifier</b> has the highest Accuracy when using SelectKBest.</H4>
<H4>After feature Selection,<b>SVC(Support Vector Classifier)</b> has the highest Accuracy when using RFE.</H4>

## 16. MODEL DEPLOYMENT

### 16.1 Save the Model
<p>Saving a machine learning model involves serializing it to disk so that it can be reused later without the need to retrain it. This process is crucial for deploying models in production systems or sharing them with others for evaluation or inference.</p>

<p><b>Save Model to a file Using Python Pickle</b></p>
<p>'pickle' is a built-in Python module that serializes (pickles) and deserializes (unpickles) Python objects, including machine learning models.</p>

### 16.2 Load Saved Model
<p><b>joblib</b> is part of the scikit-learn library and is optimized for efficiently serializing scikit-learn models, which often include large numpy arrays internally.It offers an alternative to Python's built-in pickle module, providing better performance for objects that contain large data buffers, such as trained machine learning models.</p>

<p><b>Load Saved Model</b></p>

<i>mj=joblib.load('model_joblib')</i>

### 16.3 Test with unseen data
<p>Testing a machine learning model with unseen data is a crucial step to evaluate its performance and generalization ability. Below, I'll outline a basic approach to test a machine learning model trained for water quality prediction using scikit-learn and joblib for model serialization:</p>

<p><b>Load the Trained Model (for Testing)</b></p>

<p>Load the saved model from the file 'water_quality_model.joblib' to ensure consistency in testing:</p>

 <p><b>Prepare Unseen Data for Testing</b></p>

<p>Prepare your unseen data <b>(X_unseen)</b> for testing the model's performance. This data should have the same structure (features) as the training data but should not have been used during model training.</p>
<h4>Accuracy on unseen data: 0.33</h4>

 <p><b>Evaluate Model Performance</b></p>
<p>Use appropriate metrics to evaluate the model's performance on the unseen data. For classification tasks, metrics like accuracy, precision, recall, and F1-score are commonly used. Adjust the evaluation metrics based on your specific problem and goals.</p>

<p>Then we get a confusion matrix,</p>
<table border="0">
  <tr>
    <th></th>
    <th>Precision</th>
    <th>Recall</th>
    <th>F1 score</th>
    <th>Support</th>
  </tr>
  <tr> 
    <td>0</td>
    <td>0.33 </td>
    <td>1.00</td>
    <td>0.50	</td>
    <td>1</td>
  </tr>
   <tr> 
    <td>1</td>
    <td>0.00	 </td>
    <td>0.00	</td>
    <td>0.00</td>
    <td>2</td>
  </tr>
   <tr> 
    <td>accuracy</td>
    <td> 		 </td>
    <td> 	</td>
    <td>0.33</td>
    <td>3</td>
  </tr>
   <tr> 
    <td>macro avg</td>
    <td>0.17	 </td>
    <td>0.50	</td>
    <td>0.25</td>
    <td>3</td>
  </tr>
   <tr> 
    <td>weighted avg</td>
    <td>0.11	 </td>
    <td>0.33	</td>
    <td>0.17</td>
    <td>3</td>
  </tr>
</table>
## 17. LIMITATIONS
<p>In our project certain limitations include,</p>
<p>1.In case of dimensionality reduction,we couldn't find any columns with correlation.So, there is <b>no Multicollinearity</b> here.So we need to retain all features.</p>
<p>2.Then our data set not contain any categorical values,So don't do any methods like one-hot encoding and label encoding.</p>

## 18.CONCLUSION
<p>The project is meant to be replacement to the existing manual system of water testing as the existing system is very time consuming and includes human labour.</p>
<p>The system automates the process of testing the water samples using the various parameters of water.</p>
<p>In our project gives conclusion that <b>Random Forest</b> method gives the best accuracy when it comes to water quality prediction against various other machine learning methods.Out of all algorithms utilized,it was found the <b>random forest and SVC were the best at achieving accurate results</b>,with accuracy ratings of 0.649390 and 0.644817, respectively.</p>

<p>After our feature selection algorithms used SelectKBest & RFE, we get more accuracy in the <b>SVC model</b> 0.493274 & 0.567100,repectively.</p>

## 19. FUTURE WORK

<p>Future work in water quality prediction using advanced techniques like machine learning (ML) can focus on several promising avenues to enhance accuracy, efficiency, and applicability. Here are some key areas for future research and development:</p>

<ol>
   <li> <b>Integration of Multi-Source Data:</b> Incorporating diverse data sources such as satellite imagery, remote sensing data, real-time sensor networks, and social media inputs can provide a more comprehensive understanding of water quality dynamics. ML algorithms can be further refined to handle heterogeneous and high-dimensional data effectively.</li>

<li> <b>Enhanced Spatial and Temporal Resolution:</b> Improving the spatial and temporal resolution of predictive models can enable more localized and precise predictions. This involves developing models that can capture fine-scale variations in water quality parameters over time and across different geographical regions.</li>

<li> <b>Dynamic and Adaptive Modeling:</b> Developing adaptive ML models that can dynamically adjust to changing environmental conditions and emerging water quality trends is crucial. Techniques such as online learning and reinforcement learning can be explored to continuously update models based on new data inputs.</li>

<li> <b>Uncertainty Quantification:</b> Addressing and quantifying uncertainties in ML predictions is essential for robust decision-making. Future research can focus on integrating probabilistic models, Bayesian approaches, or ensemble methods to provide probabilistic forecasts and assess prediction confidence intervals.</li>

<li> <b>Interdisciplinary Approaches:</b> Encouraging interdisciplinary collaborations between ML experts, hydrologists, environmental scientists, and policymakers can lead to more holistic and actionable insights. This can facilitate the development of decision support systems that integrate predictive models with policy-relevant information.</li>

<li> <b>Real-Time Monitoring and Early Warning Systems:</b> Enhancing real-time monitoring capabilities and developing early warning systems for water quality issues (e.g., algal blooms, contamination events) using ML can help mitigate risks to human health and ecosystems. This involves leveraging advanced anomaly detection algorithms and continuous data integration.</li>

<li> <b>User-Friendly Interfaces and Stakeholder Engagement:</b> Designing user-friendly interfaces and visualization tools that communicate ML predictions effectively to stakeholders (e.g., water managers, community members) is essential. Ensuring transparency and facilitating stakeholder engagement can improve the uptake and application of predictive models in practical decision-making processes.</li>

<li> <b>Ethical and Social Considerations:</b> Addressing ethical considerations such as data privacy, bias in algorithms, and equity in access to water quality information is crucial. Future research should prioritize developing fair and inclusive approaches to water quality prediction using ML.</li>
</ol>
<br/>
<p><b>Note : </b>Finally,I am trying to save the the csv after into a new csv(visualization.csv) for <b>Data Visualization</b>.</p>

<h2 align="center" ><font color='blue'>Thank You</font></h2>






















































