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
<p>Here we plot histograms for the graphical representations of the distribution of numerical data.</p>

<h5><b>Histograms</b> are graphical representations of the distribution of numerical data. They are useful for understanding the shape, spread, and central tendency of a dataset.Histograms are versatile tools for exploring and visualizing the distribution of numerical data. They provide a quick visual summary that helps in understanding the nature of data, identifying outliers, and making initial assessments about its statistical properties.</h5>







