import pandas as pd
import torch
from transformers import TapasTokenizer, TapasForQuestionAnswering
import streamlit as st
import numpy as np
class DataQueryAI:
    def init(self):
        """Initialize the TAPAS model and tokenizer"""
        self.model_name = 'google/tapas-base-finetuned-wtq'
        self.tokenizer = TapasTokenizer.from_pretrained(self.model_name)
        self.model = TapasForQuestionAnswering.from_pretrained(self.model_name)

    def prepare_data(self, df):
        """
        Prepare the dataframe for TAPAS processing

        Args:
            df (pd.DataFrame): Input dataframe

        Returns:
            pd.DataFrame: Processed dataframe
        """
        # Convert all columns to string type for TAPAS
        for col in df.columns:
            df[col] = df[col].astype(str)

        # Reset index to ensure proper processing
        df = df.reset_index(drop=True)
        return df

    def get_answer(self, question, table):
        """
        Get answer for a question about the table

        Args:
            question (str): Natural language question
            table (pd.DataFrame): Input table

        Returns:
            str: Answer to the question
        """
        # Encode the input
        inputs = self.tokenizer(
            table=table,
            queries=[question],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get model outputs
        outputs = self.model(**inputs)

        # Process the outputs
        predicted_answer_coordinates = self.tokenizer.convert_logits_to_predictions(
            inputs,
            outputs.logits,
            outputs.logits_aggregation
        )

        # Extract coordinates and aggregation indices
        coordinates = predicted_answer_coordinates[0][0]
        aggregation_indices = predicted_answer_coordinates[1][0]

        # Get the answer
        if coordinates:
            # Handle cell selection
            answer = []
            for coordinate in coordinates:
                answer.append(table.iat[coordinate])
            answer = ", ".join(answer)
        else:
            answer = "No answer found"

        return answer
def create_streamlit_app():
    """Create the Streamlit web interface"""
    st.title("DataQuery AI - Intelligent Data Analysis")
    st.write("Upload your CSV file and ask questions about your data!")

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load and display the data
        df = pd.read_csv(uploaded_file)
        st.write("### Your Data:")
        st.dataframe(df)

        # Initialize DataQueryAI
        query_ai = DataQueryAI()

        # Prepare the data
        processed_df = query_ai.prepare_data(df)

        # Question input
        question = st.text_input("Ask a question about your data:")

        if question:
            try:
                # Get the answer
                answer = query_ai.get_answer(question, processed_df)

                # Display results
                st.write("### Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if name == "main":
    create_streamlit_app()
