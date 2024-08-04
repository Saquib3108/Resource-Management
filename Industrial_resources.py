import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Function to load and preprocess data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(0, inplace=True)
    for col in data.columns:
        if 'Workers' in col:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    data['Total Main Workers'] = data['Main Workers - Total - Males'] + data['Main Workers - Total - Females']
    data['Total Marginal Workers'] = data['Marginal Workers - Total - Males'] + data['Marginal Workers - Total - Females']
    return data

# Function to perform NLP and clustering
def nlp_clustering(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(data['NIC Name'])
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X)
    data['Industry Cluster'] = kmeans.labels_
    return data

# Function to create gender and area distribution visualization as pie charts
def plot_gender_area_distribution(data):
    main_workers_gender = data[['Main Workers - Total - Males', 'Main Workers - Total - Females']].sum()
    marginal_workers_gender = data[['Marginal Workers - Total - Males', 'Marginal Workers - Total - Females']].sum()
    main_workers_urban_rural = data[['Main Workers - Rural -  Persons', 'Main Workers - Urban -  Persons']].sum()
    marginal_workers_urban_rural = data[['Marginal Workers - Rural -  Persons', 'Marginal Workers - Urban -  Persons']].sum()

    fig_main_gender = go.Figure(data=[go.Pie(labels=['Males', 'Females'], values=[main_workers_gender['Main Workers - Total - Males'], main_workers_gender['Main Workers - Total - Females']], hole=.3)])
    fig_main_gender.update_layout(title_text='Main Workers by Gender')
    st.plotly_chart(fig_main_gender)

    fig_marginal_gender = go.Figure(data=[go.Pie(labels=['Males', 'Females'], values=[marginal_workers_gender['Marginal Workers - Total - Males'], marginal_workers_gender['Marginal Workers - Total - Females']], hole=.3)])
    fig_marginal_gender.update_layout(title_text='Marginal Workers by Gender')
    st.plotly_chart(fig_marginal_gender)

    fig_main_area = go.Figure(data=[go.Pie(labels=['Rural', 'Urban'], values=[main_workers_urban_rural['Main Workers - Rural -  Persons'], main_workers_urban_rural['Main Workers - Urban -  Persons']], hole=.3)])
    fig_main_area.update_layout(title_text='Main Workers by Area')
    st.plotly_chart(fig_main_area)

    fig_marginal_area = go.Figure(data=[go.Pie(labels=['Rural', 'Urban'], values=[marginal_workers_urban_rural['Marginal Workers - Rural -  Persons'], marginal_workers_urban_rural['Marginal Workers - Urban -  Persons']], hole=.3)])
    fig_marginal_area.update_layout(title_text='Marginal Workers by Area')
    st.plotly_chart(fig_marginal_area)

# Function to create top industries visualization
def plot_top_industries(data):
    top_industries = data.groupby('NIC Name')['Total Main Workers'].sum().nlargest(10)
    fig = px.bar(top_industries, x=top_industries.values, y=top_industries.index, orientation='h', 
                 title='Top 10 Industries by Number of Main Workers')
    st.plotly_chart(fig)

# Function to create geographical map visualization
def plot_geographical_map(data):
    with open('india_states.geojson') as f:
        geojson = json.load(f)
    
    states = data['India/States'].unique()
    selected_state = st.selectbox("Select a state for geographical map", states)
    state_data = data[data['India/States'] == selected_state]
    fig = px.choropleth(state_data, 
                        geojson=geojson,
                        locations='District Code', 
                        color='Total Main Workers',
                        hover_name='District Code',
                        title=f'Geographical Distribution of Main Workers in {selected_state}',
                        featureidkey='properties.NAME_1')
    fig.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig)

# Function to create industry clusters visualization
def plot_industry_clusters(data):
    cluster = st.selectbox('Select Industry Cluster', range(10))
    filtered_data = data[data['Industry Cluster'] == cluster]
    fig = px.bar(filtered_data, x='NIC Name', y='Total Main Workers', title=f'Total Main Workers in Cluster {cluster}')
    st.plotly_chart(fig)

# Function to create geographical map visualization
def plot_geographical_map(data):
    states = data['India/States'].unique()
    selected_state = st.selectbox("Select a state for geographical map", states)
    state_data = data[data['India/States'] == selected_state]
    
    fig = px.choropleth(state_data, 
                        geojson='https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json',
                        locations='District Code', 
                        color='Total Main Workers',
                        hover_name='District Code',
                        title=f'Geographical Distribution of Main Workers in {selected_state}')
    st.plotly_chart(fig)

# Function to create total workers by state visualization
def plot_total_workers_by_state(data):
    total_workers_by_state = data.groupby('India/States')['Total Main Workers', 'Total Marginal Workers'].sum()
    total_workers_by_state = total_workers_by_state.reset_index()
    
    fig = go.Figure(data=[
        go.Bar(name='Total Main Workers', x=total_workers_by_state['India/States'], y=total_workers_by_state['Total Main Workers']),
        go.Bar(name='Total Marginal Workers', x=total_workers_by_state['India/States'], y=total_workers_by_state['Total Marginal Workers'])
    ])
    fig.update_layout(barmode='group', title='Total Main and Marginal Workers by State')
    st.plotly_chart(fig)

# Function to create histograms
def plot_histograms(data):
    fig = px.histogram(data, x='Total Main Workers', nbins=50, title='Distribution of Total Main Workers')
    st.plotly_chart(fig)
    
    fig = px.histogram(data, x='Total Marginal Workers', nbins=50, title='Distribution of Total Marginal Workers')
    st.plotly_chart(fig)
    
    fig = px.histogram(data, x='Main Workers - Total - Males', nbins=50, title='Distribution of Main Workers - Males')
    st.plotly_chart(fig)
    
    fig = px.histogram(data, x='Main Workers - Total - Females', nbins=50, title='Distribution of Main Workers - Females')
    st.plotly_chart(fig)
    
    fig = px.histogram(data, x='Marginal Workers - Total - Males', nbins=50, title='Distribution of Marginal Workers - Males')
    st.plotly_chart(fig)
    
    fig = px.histogram(data, x='Marginal Workers - Total - Females', nbins=50, title='Distribution of Marginal Workers - Females')
    st.plotly_chart(fig)

# Function to display statistical analysis
def display_statistical_analysis(data):
    st.subheader('Statistical Analysis')
    st.write(data.describe())

# Main Streamlit app
def main():
    st.title('Industrial Human Resource Geo-Visualization')

    # Load data
    file_path = 'D:/CT_Project_ML/merged_data.csv'  # Update with your file path
    data = load_data(file_path)

    # Perform NLP and clustering
    data = nlp_clustering(data)

    # Sidebar for selecting visualizations
    st.sidebar.title("Visualization Options")
    option = st.sidebar.selectbox("Select a visualization", 
                                  ["Distribution by Gender and Area", "Top Industries", "Geographical Distribution", "Industry Clusters", "Geographical Map", "Total Workers by State", "Histograms", "Statistical Analysis"])

    # Display visualizations
    if option == "Distribution by Gender and Area":
        st.subheader('Distribution of Workers by Gender and Urban/Rural Areas')
        plot_gender_area_distribution(data)
    elif option == "Top Industries":
        st.subheader('Top Industries by Number of Workers')
        plot_top_industries(data)
    elif option == "Geographical Distribution":
        st.subheader('Geographical Distribution of Workers')
        plot_geographical_map(data)
    elif option == "Industry Clusters":
        st.subheader('Industry Clusters Overview')
        plot_industry_clusters(data)
    elif option == "Geographical Map":
        st.subheader('Geographical Map of Workers in India')
        plot_geographical_map(data)
    elif option == "Total Workers by State":
        st.subheader('Total Workers by State')
        plot_total_workers_by_state(data)
    elif option == "Histograms":
        st.subheader('Histograms of Worker Distributions')
        plot_histograms(data)
    elif option == "Statistical Analysis":
        display_statistical_analysis(data)

if __name__ == "__main__":
    main()