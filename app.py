import pandas as pd
import streamlit as st
import numpy as np
import re
import time
import logging
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(_name_)

class DataAnalyzer:
    """Handles data analysis and query processing."""
    
    def _init_(self):
        self.keywords = {
            'max': self._get_max,
            'min': self._get_min,
            'average': self._get_average,
            'mean': self._get_average,
            'sum': self._get_sum,
            'count': self._get_count,
            'show': self._show_values,
            'display': self._show_values,
            'greater than': self._greater_than,
            'less than': self._less_than,
            'between': self._between,
        }
    
    def _get_column_match(self, question: str, columns: List[str]) -> Optional[str]:
        """Find the best matching column for the question."""
        question_lower = question.lower()
        
        # Check exact match first
        for col in columns:
            if col.lower() in question_lower:
                return col
        
        # Check partial match using individual words
        for col in columns:
            for word in col.lower().split():
                if word in question_lower:
                    return col
        
        return None
    
    def _is_numeric_column(self, df: pd.DataFrame, column: str) -> bool:
        """Check if a column contains numeric data."""
        try:
            return pd.api.types.is_numeric_dtype(df[column])
        except (ValueError, TypeError):
            return False
    
    def _get_max(self, df: pd.DataFrame, column: str) -> str:
        if self._is_numeric_column(df, column):
            return f"The maximum value in {column} is {df[column].max()}"
        return f"Cannot calculate maximum for non-numeric column {column}"
    
    def _get_min(self, df: pd.DataFrame, column: str) -> str:
        if self._is_numeric_column(df, column):
            return f"The minimum value in {column} is {df[column].min()}"
        return f"Cannot calculate minimum for non-numeric column {column}"
    
    def _get_average(self, df: pd.DataFrame, column: str) -> str:
        if self._is_numeric_column(df, column):
            return f"The average value in {column} is {df[column].mean():.2f}"
        return f"Cannot calculate average for non-numeric column {column}"
    
    def _get_sum(self, df: pd.DataFrame, column: str) -> str:
        if self._is_numeric_column(df, column):
            return f"The sum of values in {column} is {df[column].sum()}"
        return f"Cannot calculate sum for non-numeric column {column}"
    
    def _get_count(self, df: pd.DataFrame, column: str) -> str:
        return f"The count of values in {column} is {df[column].count()}"
    
    def _show_values(self, df: pd.DataFrame, column: str) -> str:
        values = df[column].unique()
        return f"Values in {column}: {', '.join(map(str, values[:10]))}" + \
               ("..." if len(values) > 10 else "")
    
    def _greater_than(self, df: pd.DataFrame, column: str, value: float) -> str:
        if self._is_numeric_column(df, column):
            count = (df[column] > value).sum()
            return f"There are {count} values in {column} greater than {value}"
        return f"Cannot compare values for non-numeric column {column}"
    
    def _less_than(self, df: pd.DataFrame, column: str, value: float) -> str:
        if self._is_numeric_column(df, column):
            count = (df[column] < value).sum()
            return f"There are {count} values in {column} less than {value}"
        return f"Cannot compare values for non-numeric column {column}"
    
    def _between(self, df: pd.DataFrame, column: str, min_val: float, max_val: float) -> str:
        if self._is_numeric_column(df, column):
            count = ((df[column] >= min_val) & (df[column] <= max_val)).sum()
            return f"There are {count} values in {column} between {min_val} and {max_val}"
        return f"Cannot compare values for non-numeric column {column}"
    
    def process_question(self, question: str, df: pd.DataFrame) -> Tuple[str, float]:
        """Process a natural language question about the data."""
        try:
            column = self._get_column_match(question, df.columns)
            if not column:
                return "I couldn't identify which column you're asking about.", 0.0
            
            question_lower = question.lower()
            numbers = re.findall(r'\d+\.?\d*', question)
            
            for keyword, func in self.keywords.items():
                if keyword in question_lower:
                    if keyword in ['greater than', 'less than'] and numbers:
                        return func(df, column, float(numbers[0])), 0.8
                    elif keyword == 'between' and len(numbers) >= 2:
                        return func(df, column, float(numbers[0]), float(numbers[1])), 0.8
                    elif keyword not in ['greater than', 'less than', 'between']:
                        return func(df, column), 0.9
            
            return self._show_values(df, column), 0.5
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return f"Sorry, I couldn't process that question: {str(e)}", 0.0

def create_streamlit_app():
    """Create the Streamlit web interface."""
    
    st.set_page_config(page_title="DataQuery AI", page_icon="📊", layout="wide")
    
    st.title("📊 DataQuery AI - Simple Data Analysis")
    st.write("""
    Upload your CSV file and ask questions about your data in natural language!
    
    Example questions:
    - What is the maximum value in [column]?
    - Show me the values in [column]
    - What is the average of [column]?
    - How many values in [column] are greater than 100?
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Your Data:")
            st.dataframe(df)
            
            analyzer = DataAnalyzer()
            
            with st.form(key="query_form"):
                question = st.text_input("Ask a question about your data:")
                submit_button = st.form_submit_button("Analyze")
            
            if submit_button and question:
                start_time = time.time()
                answer, confidence = analyzer.process_question(question, df)
                processing_time = time.time() - start_time
                
                st.write("### Answer:")
                st.write(answer)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence Score", f"{confidence:.2%}")
                with col2:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")

if _name_ == "_main_":
    create_streamlit_app()
