"""
Conversational AI Streamlit Web Application
Interactive web interface for the Conversational AI Bot.
Usage:
    streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path
import torch

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))
from model_handler import ModelHandler

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Conversational AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .example-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .bot-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load & cache model
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_name: str, adapter_path: str = None):
    """Load and cache the model with CPU/GPU awareness"""
    use_8bit = torch.cuda.is_available()  # Enable 8-bit only on GPU
    return ModelHandler(
        model_name=model_name,
        adapter_path=adapter_path,
        load_in_8bit=use_8bit
    )

# ----------------------------
# Initialize session state
# ----------------------------
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are a conversational AI assistant. "
                    "You provide accurate, helpful, and concise answers to user questions. "
                    "When answering coding questions, provide clear explanations and code examples."
                )
            }
        ]
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False

# ----------------------------
# Main Application
# ----------------------------
def main():
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🤖 Conversational AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI Assistant for Technology & IT Questions</p>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        model_name = st.selectbox(
            "Select Model",
            ["meta-llama/Llama-3.2-1B-Instruct"],
            index=0
        )
        adapter_path = st.text_input(
            "Fine-tuned Adapter Path (optional)",
            placeholder="models/conversational-ai-finetuned",
            help="Leave empty to use base model"
        )
        if not adapter_path:
            adapter_path = None

        st.subheader("🎛️ Generation Settings")
        temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        max_tokens = st.slider("Max Response Length", 128, 1024, 512, 128)
        top_p = st.slider("Top P", 0.1, 1.0, 0.9, 0.05)

        if st.button("🚀 Load Model", type="primary"):
            with st.spinner("Loading model... This may take a minute..."):
                try:
                    st.session_state.model_handler = load_model(model_name, adapter_path)
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")

        if st.button("🧹 Clear Conversation"):
            st.session_state.messages = [st.session_state.messages[0]]  # keep system prompt
            st.rerun()

        # About section
        st.divider()
        st.subheader("ℹ️ About")
        st.info("This is a conversational AI assistant.")

    # Main content area
    if not st.session_state.model_loaded:
        st.info("👈 Please load a model from the sidebar to start chatting!")
    else:
        # Display conversation
        for message in st.session_state.messages[1:]:
            role = message["role"]
            content = message["content"]
            if role == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(content)
            elif role == "assistant":
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(content)

        # Chat input
        if prompt := st.chat_input("Ask a technical question..."):
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })

            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.model_handler.chat(
                            messages=st.session_state.messages,
                            max_new_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                        st.markdown(response)

                        # Add to conversation history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

                    except Exception as e:
                        st.error(f"❌ Error generating response: {e}")


if __name__ == "__main__":
    main()
