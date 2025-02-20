import plotly.express as px
import pandas as pd
import streamlit as st

def create_income_distribution_plot(sample_data):
    income_dist = pd.DataFrame({
        'Income Category': ['≤$50K', '>$50K'],
        'Count': [
            (sample_data['income'] == 0).sum(),
            (sample_data['income'] == 1).sum()
        ]
    })
    
    fig = px.pie(income_dist, names='Income Category', values='Count', 
                title="Income Distribution in Dataset",
                color_discrete_sequence=['#FFA726', '#66BB6A'])
    return fig

def create_age_income_plots(sample_data):
    fig1 = px.box(sample_data, x='income', y='age', 
            color='income',
            title="Age Distribution by Income Level",
            labels={'income': 'Income >$50K', 'age': 'Age'},
            color_discrete_map={0: '#FFA726', 1: '#66BB6A'})
    st.plotly_chart(fig1, use_container_width=True)

    # Add age distribution histogram
    fig2 = px.histogram(sample_data, x='age', color='income',
            title="Age Distribution by Income",
            labels={'age': 'Age', 'income': 'Income >$50K'},
            opacity=0.7,
            color_discrete_map={0: '#FFA726', 1: '#66BB6A'})
    st.plotly_chart(fig2, use_container_width=True)

    return fig1, fig2


def create_education_income_plot(sample_data):
        # Count by education and income
    edu_counts = sample_data.groupby(['education', 'income']).size().reset_index(name='count')
    
    fig = px.bar(edu_counts, x='education', y='count', color='income',
                title="Education Level vs Income",
                labels={'education': 'Education Level', 'count': 'Count', 'income': 'Income >$50K'},
                color_discrete_map={0: '#FFA726', 1: '#66BB6A'})
    st.plotly_chart(fig, use_container_width=True)
    return fig

def create_marital_income_plot(sample_data):
    marital_counts = sample_data.groupby(['marital-status', 'income']).size().reset_index(name='count')

    fig = px.bar(marital_counts, x='marital-status', y='count', color='income',
            title="Marital Status vs Income",
            labels={'marital-status': 'Marital Status', 'count': 'Count', 'income': 'Income >$50K'},
            color_discrete_map={0: '#FFA726', 1: '#66BB6A'})
    st.plotly_chart(fig, use_container_width=True)
    return fig


def create_occupation_income_plot(sample_data):
        # Count by occupation and income
    occ_counts = sample_data.groupby(['occupation', 'income']).size().reset_index(name='count')
    
    fig = px.bar(occ_counts, x='occupation', y='count', color='income',
                title="Occupation vs Income",
                labels={'occupation': 'Occupation', 'count': 'Count', 'income': 'Income >$50K'},
                color_discrete_map={0: '#FFA726', 1: '#66BB6A'})
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    st.plotly_chart(fig, use_container_width=True)
    return fig


def create_workclass_income_plot(sample_data):
    work_counts = sample_data.groupby(['workclass', 'income']).size().reset_index(name='count')

    fig = px.bar(work_counts, x='workclass', y='count', color='income',
            title="Work Class vs Income",
            labels={'workclass': 'Work Class', 'count': 'Count', 'income': 'Income >$50K'},
            color_discrete_map={0: '#FFA726', 1: '#66BB6A'})
    st.plotly_chart(fig, use_container_width=True)
    return fig





def create_feature_vs_income_plot(sample_data, viz_type):
    if viz_type == "Age vs Income":
        return create_age_income_plots(sample_data)
    elif viz_type == "Education vs Income":
        return create_education_income_plot(sample_data)
    # [Add other visualization functions]
    elif viz_type == "Marital Status vs Income":
        return create_marital_income_plot(sample_data)
    elif viz_type == "Occupation vs Income":
        return create_occupation_income_plot(sample_data)
    elif viz_type == "Workclass vs Income":
        return create_workclass_income_plot(sample_data)
    
    else:
        return 'Choose from the available visualization types'





        # # Data summary
        # st.subheader("Dataset Overview")
        
        # # Sample income distribution
        # income_dist = pd.DataFrame({
        #     'Income Category': ['≤$50K', '>$50K'],
        #     'Count': [
        #         (sample_data['income'] == 0).sum(),
        #         (sample_data['income'] == 1).sum()
        #     ]
        # })
        
        # fig = px.pie(income_dist, names='Income Category', values='Count', 
        #             title="Income Distribution in Dataset",
        #             color_discrete_sequence=['#FFA726', '#66BB6A'])
        # st.plotly_chart(fig, use_container_width=True)
        
        # # Visualization options
        # st.subheader("Explore Data Relationships")
        # viz_type = st.selectbox("Select visualization type", [
        #     "Age vs Income", 
        #     "Education vs Income",
        #     "Marital Status vs Income",
        #     "Occupation vs Income",
        #     "Workclass vs Income"
        # ])
        
        # # Make sure all expected columns exist in the sample data
        # expected_columns = ['age', 'education', 'marital-status', 'occupation', 'workclass', 'income']
        # missing_columns = [col for col in expected_columns if col not in sample_data.columns]
        
        # if missing_columns:
        #     st.warning(f"Missing columns in dataset: {', '.join(missing_columns)}. Some visualizations may not work.")
        

            
