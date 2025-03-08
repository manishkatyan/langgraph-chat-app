import os
import streamlit as st
from arize.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# Setup OTel
if "tracer_provider" not in st.session_state:
    st.session_state.tracer_provider = register(
        space_id = os.getenv("ARIZE_SPACE_ID"),
        api_key = os.getenv("ARIZE_API_KEY"),
        project_name = os.getenv("ARIZE_PROJECT_NAME"),
    )
    # Initialize instrumentation only once
    LangChainInstrumentor().instrument(tracer_provider=st.session_state.tracer_provider)

tracer = st.session_state.tracer_provider.get_tracer(__name__)