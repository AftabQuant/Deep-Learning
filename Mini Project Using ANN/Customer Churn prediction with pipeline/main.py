import gradio as gr
import joblib
import numpy as np
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')


def load_model() :
    model = joblib.load('pipepline.pkl')
    return model


def prediction(CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
               HasCrCard, IsActiveMember, EstimatedSalary):

    columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
               'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
    
    input_data = pd.DataFrame([[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts,
               HasCrCard, IsActiveMember, EstimatedSalary]],
                              columns=columns)
    pipeline = load_model()
    prediction = pipeline.predict(input_data)
    return prediction[0]


demo = gr.Interface(
    fn=prediction,
    inputs=[
        gr.Number(label="CreditScore"),
        gr.Radio(choices=["France", "Germany", "Spain"], label="Geography"),
        gr.Radio(choices=["Male", "Female"], label="Gender"),
        gr.Number(label="Age"),
        gr.Dropdown(choices=[0, 1, 2, 3, 4, 5, 6 , 7, 8, 9, 10], label="Tenure"),
        gr.Number(label="Balance"),
        gr.Radio(choices=[1, 2, 3, 4], label="NumOfProducts"),
        gr.Radio(choices=["Yes", "No"], label="HasCrCard"),
        gr.Radio(choices=["Yes", "No"], label="IsActiveMember"),
        gr.Number(label="EstimatedSalary"),
    ],
    outputs=[gr.Textbox(label='Prediction')],
    title="Customer Churn Prediction",
    description="Fill in user and usage details to predict behavior or classification outcome."
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
