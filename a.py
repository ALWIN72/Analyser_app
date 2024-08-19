#=======================================================================
## 0. Importing libraries and setting up streamlit web app

#Importing the necessary packages
import sys
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# Example HTML content
import streamlit as st

# Define your HTML content
html_content = """
<html>
<head>
    <title>Test</title>
    <style>
        body { font-family: Arial, sans-serif; }
        h1 { color: #007bff; }
        p { font-size: 16px; }
    </style>
</head>
<body>
    <h1>DATASET ANALY$ER</h1>
    
</body>
</html>
"""

st.markdown(html_content, unsafe_allow_html=True)


#Setting up web app page
#st.set_page_config(page_title='Exploratory Data Analysis App', page_icon=None, layout="wide")



#Creating section in sidebar
st.sidebar.write("****A) File upload****")

#User prompt to select file type
ft = st.sidebar.selectbox("*What is the file type?*",["Excel", "csv"])

#Creating dynamic file upload option in sidebar
uploaded_file = st.sidebar.file_uploader("*Upload file here*")

if uploaded_file is not None:
    file_path = uploaded_file

    if ft == 'Excel':
        try:
            #User prompt to select sheet name in uploaded Excel
            sh = st.sidebar.selectbox("*Which sheet name in the file should be read?*",pd.ExcelFile(file_path).sheet_names)
            #User prompt to define row with column names if they aren't in the header row in the uploaded Excel
            h = st.sidebar.number_input("*Which row contains the column names?*",0,100)
        except:
            st.info("File is not recognised as an Excel file")
            sys.exit()
    
    elif ft == 'csv':
        try:
            #No need for sh and h for csv, set them to None
            sh = None
            h = None
        except:
            st.info("File is not recognised as a csv file.")
            sys.exit()

    #Caching function to load data
    @st.cache_data(experimental_allow_widgets=True)
    def load_data(file_path,ft,sh,h):
        
        if ft == 'Excel':
            try:
                #Reading the excel file
                data = pd.read_excel(file_path,header=h,sheet_name=sh,engine='openpyxl')
            except:
                st.info("File is not recognised as an Excel file.")
                sys.exit()
    
        elif ft == 'csv':
            try:
                #Reading the csv file
                data = pd.read_csv(file_path)
            except:
                st.info("File is not recognised as a csv file.")
                sys.exit()
        
        return data

    data = load_data(file_path,ft,sh,h)

#=====================================================================================================
## 1. Overview of the data
    st.write( '### 1. Dataset Preview ')

    try:
      #View the dataframe in streamlit
      st.dataframe(data, use_container_width=True)

    except:
      st.info("The file wasn't read properly. Please ensure that the input parameters are correctly defined.")
      sys.exit()



      ## 2. Understanding the data
    st.write( '### 2. High-Level Overview ')

    #Creating radio button and sidebar simulataneously
    selected = st.sidebar.radio( "**B) What would you like to know about the data?**", 
                                ["Data Dimensions",
                                 "Field Descriptions",
                                "Summary Statistics", 
                                "Value Counts of Fields"])

    #Showing field types
    if selected == 'Field Descriptions':
        fd = data.dtypes.reset_index().rename(columns={'index':'Field Name',0:'Field Type'}).sort_values(by='Field Type',ascending=False).reset_index(drop=True)
        st.dataframe(fd, use_container_width=True)

    #Showing summary statistics
    elif selected == 'Summary Statistics':
        ss = pd.DataFrame(data.describe(include='all').round(2).fillna(''))
        st.dataframe(ss, use_container_width=True)

    #Showing value counts of object fields
    elif selected == 'Value Counts of Fields':
        # creating radio button and sidebar simulataneously if this main selection is made
        sub_selected = st.sidebar.radio( "*Which field should be investigated?*",data.select_dtypes('object').columns)
        vc = data[sub_selected].value_counts().reset_index().rename(columns={'count':'Count'}).reset_index(drop=True)
        st.dataframe(vc, use_container_width=True)

    #Showing the shape of the dataframe
    else:
        st.write('###### The data has the dimensions :',data.shape)


      #=====================================================================================================
import plotly.express as px

# Visualization section
vis_select = st.sidebar.checkbox("**C) Is visualization required for this dataset?**")

if vis_select:
    st.write('### 3. Visual Insights')

    # Dropdown to select visualization type
    viz_type = st.sidebar.selectbox("**Select Visualization Type**", [
        "Distribution of Numeric Features",
        "Trend Analysis",
        "Clustering",
        "Outlier Detection",
        "Pie Chart",
        "Graph Interpreter"
    ])
    numeric_columns = data.select_dtypes(include='number').columns
    categorical_columns = data.select_dtypes(include='object').columns
    if viz_type == "Distribution of Numeric Features":
        st.write("#### Distribution of Numeric Features")

        numeric_columns = data.select_dtypes(include='number').columns
        selected_column = st.sidebar.selectbox("Select a numeric column", numeric_columns)

        if selected_column:
            fig = px.histogram(data, x=selected_column, title=f'Distribution of {selected_column}')
            st.plotly_chart(fig)

    elif viz_type == "Trend Analysis":
        st.write("#### Trend Analysis")

        if 'Date' in data.columns:
            fig = px.line(data, x='Date', y= numeric_columns, title='Trend Analysis')
            st.plotly_chart(fig)
        else:
            st.write("No 'Date' column found for trend analysis.")

    elif viz_type == "Clustering":
        st.write("#### Clustering")

        # Ensure there are numeric columns for clustering
        if numeric_columns.size > 0:
            # Select features for clustering
            features = st.sidebar.multiselect("Select features for clustering", numeric_columns)
            
            if len(features) >= 2:
                # Standardize features
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(data[features])

                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=3, random_state=0).fit(scaled_data)
                data['Cluster'] = kmeans.labels_

                fig = px.scatter_matrix(data, dimensions=features, color='Cluster', title='Clustering Visualization')
                st.plotly_chart(fig)
            else:
                st.write("Please select at least two features for clustering.")
        
    elif viz_type == "Outlier Detection":
        st.write("#### Outlier Detection")

        # Use a simple Z-score method for outlier detection
        from scipy import stats

        numeric_data = data[numeric_columns]
        z_scores = stats.zscore(numeric_data)
        abs_z_scores = np.abs(z_scores)
        outliers = (abs_z_scores > 3).all(axis=1)

        data['Outlier'] = outliers
        fig = px.scatter_matrix(data, dimensions=numeric_columns, color='Outlier', title='Outlier Detection')
        st.plotly_chart(fig)

    elif viz_type == "Pie Chart":
        st.write("#### Pie Chart")
        if categorical_columns.size > 0:
            pie_column = st.sidebar.selectbox("Select a categorical column", categorical_columns)
            if pie_column:
                pie_data = data[pie_column].value_counts()
                fig = px.pie(names=pie_data.index, values=pie_data.values, title=f'Pie Chart of {pie_column}')
                st.plotly_chart(fig)
        else:
            st.write("No categorical columns available for pie chart.")

    

    

    if viz_type == "Graph Interpreter":
        st.write("#### Graph Interpreter")
        st.write("Here you can use additional graphing tools or code to provide more advanced visualizations.")

        # Select plot type
        plot_type = st.sidebar.selectbox("Select Plot Type", [
            "Scatter Plot", "Line Plot", "Bar Plot", "Box Plot", "Heatmap", "Histogram", "Area Plot", "Violin Plot"
        ])

        if numeric_columns.size > 0:
            x_axis = st.sidebar.selectbox("Select X-axis", numeric_columns)
            y_axis = st.sidebar.selectbox("Select Y-axis", numeric_columns)

            if x_axis and y_axis:
                fig = go.Figure()

                # Plot type logic
                if plot_type == "Scatter Plot":
                    fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_axis], mode='markers', name='Scatter Plot'))
                elif plot_type == "Line Plot":
                    fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_axis], mode='lines+markers', name='Line Plot'))
                elif plot_type == "Bar Plot":
                    fig.add_trace(go.Bar(x=data[x_axis], y=data[y_axis], name='Bar Plot'))
                elif plot_type == "Box Plot":
                    fig.add_trace(go.Box(y=data[y_axis], name=f'Box Plot of {y_axis}'))
                elif plot_type == "Heatmap":
                    # Create a heatmap if both columns are numeric
                    if data[x_axis].nunique() <= 10 and data[y_axis].nunique() <= 10:
                        heatmap_data = pd.crosstab(data[x_axis], data[y_axis])
                        fig = go.Figure(data=go.Heatmap(z=heatmap_data.values, x=heatmap_data.columns, y=heatmap_data.index))
                        fig.update_layout(title='Heatmap', xaxis_title=x_axis, yaxis_title=y_axis)
                    else:
                        st.write("For heatmap, both X and Y axes should have categorical or limited numeric values.")
                        st.stop()
                elif plot_type == "Histogram":
                    fig.add_trace(go.Histogram(x=data[x_axis], name=f'Histogram of {x_axis}'))
                elif plot_type == "Area Plot":
                    fig.add_trace(go.Scatter(x=data[x_axis], y=data[y_axis], mode='lines', fill='tozeroy', name='Area Plot'))
                elif plot_type == "Violin Plot":
                    fig.add_trace(go.Violin(y=data[y_axis], box_visible=True, line_color='purple', name=f'Violin Plot of {y_axis}'))

                # Update layout with custom options
                fig.update_layout(
                    title=f'{plot_type} of {x_axis} vs {y_axis}',
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    template="plotly_dark"  # Customize the template as needed
                )

                st.plotly_chart(fig)
            else:
                st.write("Please select both X and Y axes for the plot.")
        else:
            st.write("No numeric columns available for custom graphs.")
