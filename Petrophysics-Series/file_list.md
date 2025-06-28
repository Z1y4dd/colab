# Well Log Data Analysis Project Workflow

## 1. Data Loading and Integration

These notebooks will help you import, load, and combine various well data files (CSV, LAS, etc.) into a format suitable for analysis.

-   **[01 - Loading and Displaying Well Data From CSV.ipynb](#)**  
     Loading and displaying data from CSV files, likely your starting point for importing data.

-   **[15 - Loading Multiple LAS Files to a Dataframe.ipynb](#)**  
     Loading LAS files and converting them into a DataFrame for analysis.

-   **[17 - Loading DLIS Data.ipynb](#)**  
     If your data includes DLIS files, this will teach you how to handle them.

-   **[36 - Combining Multiple LAS Files and Formation Data.ipynb](#)**  
     Combining data from multiple LAS files along with formation data into a unified DataFrame.

-   **[22 - CSV File to LAS File.ipynb](#)**  
     If needed, converting CSV data back into LAS format for compatibility.

---

## 2. Data Cleaning and Preprocessing

At this stage, you will need to clean the data, handle missing values, and standardize it for further analysis.

-   **[08 - Curve Normalisation.ipynb](#)**  
     Normalizing curves (log data) to ensure consistency across different well logs.

-   **[33 - Auto Outlier Detection - Isolation Forest.ipynb](#)**  
     Outlier detection to identify and remove invalid or extreme values in your data.

-   **[23 - Pandas Profiling.ipynb](#)**  
     Automated exploratory data analysis (EDA) using Pandas profiling to quickly summarize and understand the dataset.

---

## 3. Exploratory Data Analysis (EDA)

Now that you have the data cleaned, you will explore the relationships between different parameters and gain insights into the data.

-   **[10 - Exploratory Data Analysis with Well Log Data.ipynb](#)**  
     Performing exploratory data analysis on your well log data to uncover patterns, distributions, and correlations.

-   **[03 - Displaying Histograms and Crossplots.ipynb](#)**  
     Visualizing the distribution of well log data through histograms and crossplots to explore relationships between variables.

-   **[21 - Identifying Outliers with Boxplots in Matplotlib.ipynb](#)**  
     Using **boxplots** to visualize and identify outliers in your data.

-   **[09 - Visualising Data Coverage - Multi Well.ipynb](#)**  
     Visualizing the **coverage** of data across multiple wells to ensure comprehensive data representation.

---

## 4. Visualization and Plotting

These notebooks will guide you in visualizing the well log data with different types of plots, enhancing your understanding of the formation and data.

-   **[02 - Displaying a Well Plot with MatPlotLib.ipynb](#)**  
     Visualizing individual well logs using **matplotlib**.

-   **[06 - Displaying Formations on Log Plots.ipynb](#)**  
     Adding **formation data** to your well log plots to better understand geological layers.

-   **[12 - Enhancing Log Plots With Plot Fills.ipynb](#)**  
     Improving the visualizations of log plots by **filling areas** of interest such as hydrocarbon-bearing zones.

-   **[13 - Displaying LWD Image Data.ipynb](#)**  
     Displaying **LWD (Logging While Drilling)** image data for a more comprehensive view of well conditions.

-   **[20 - Core Data Visualisation - Matplotlib subplot2grid.ipynb](#)**  
     Visualizing core data alongside well log data using a **subplot grid** for better organization.

-   **[24 - Creating Poro-Perm Crossplots With Seaborn.ipynb](#)**  
     Visualizing **porosity-permeability** relationships using **crossplots** in **Seaborn**.

-   **[28 - Seaborn Boxplots of Well Log Data.ipynb](#)**  
     Using **Seaborn boxplots** to further analyze well log data distributions and outliers.

-   **[38 - Visualising Lithology Data - Alternatives to Pie Charts.ipynb](#)**  
     Visualizing **lithology** data with alternatives to pie charts for clearer representation.

---

## 5. Petrophysical Calculations and Analysis

These notebooks will guide you through **petrophysical calculations** such as estimating **porosity**, **permeability**, and their interrelationships.

-   **[05 - Petrophysical Calculations.ipynb](#)**  
     Performing basic **petrophysical calculations**, such as calculating **porosity** and **permeability**.

-   **[04 - Displaying Core Data and Deriving a Poro Perm Relationship.ipynb](#)**  
     Analyzing **core data** and deriving relationships between **porosity** and **permeability**.

-   **[11 - Poro-Perm Relationships.ipynb](#)**  
     Understanding **poro-perm** relationships and their significance in **reservoir characterization**.

---

## 6. Machine Learning for Prediction and Classification

These notebooks introduce advanced techniques, such as **Random Forest**, for **classification** or **regression** tasks.

-   **[27 - Random Forest for Lithology Classification - Multi Class Output.ipynb](#)**  
     Using **Random Forest** for **multi-class lithology classification** based on well log data.

-   **[29 - Random Forest for Regression - Prediction of Continuous Well Logs-Copy1.ipynb](#)**  
     Applying **Random Forest** regression to predict **continuous well log** data like **porosity** or **resistivity**.

-   **[37 - Classification with K-Nearest Neighbors.ipynb](#)**  
     Using **K-Nearest Neighbors** for **lithology classification** based on well log data.

-   **[33 - Auto Outlier Detection - Isolation Forest.ipynb](#)**  
     Using **Isolation Forest** for **outlier detection** in well log data before applying machine learning models.

---

## 7. Geospatial and Mapping

These notebooks focus on visualizing well locations and other spatial aspects of well data.

-   **[34 - Folium for Well Mapping.ipynb](#)**  
     Using **Folium** to create interactive maps for visualizing **well locations** and related data.

-   **[32 - Creating Geospatial Heatmaps of DT Measurements.ipynb](#)**  
     Creating **geospatial heatmaps** to visualize the distribution of **Delta-T (DT)** measurements across multiple wells.

---

## 8. Miscellaneous and Advanced Techniques

These notebooks provide additional utilities or advanced techniques for handling or visualizing well data.

-   **[39 - dtale for EDA.ipynb](#)**  
     Using **D-Tale** to quickly explore data interactively, which can be useful for a deeper EDA process.

-   **[22 - CSV File to LAS File.ipynb](#)**  
     Converting **CSV** files to **LAS** format, in case you need to export data to LAS.

-   **[16 - Advanced Well Log Plots - Adding Formation Data to a Well Log Plot.ipynb](#)**  
     Advanced techniques for adding **formation data** to **well log plots** to enhance visualization.

---

## 9. Not Needed for Your Project (Optional)

-   **[14 - Displaying Lithology Data.ipynb](#)**
-   **[26 - Creating Stereonets in Python.ipynb](#)**
-   **[41 - Creating Waffle Charts of Lithology Data.ipynb](#)**
-   **[37 - Classification with K-Nearest Neighbors.ipynb](#)**
-   **[21 - Identifying Outliers with Boxplots in Matplotlib.ipynb](#)**
