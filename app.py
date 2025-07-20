import streamlit as st
import PyPDF2
import io
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv("config.env")

# For Streamlit Cloud deployment, use st.secrets
def get_api_key():
    # Try Streamlit secrets first (for cloud deployment)
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        if api_key:
            return api_key
    except Exception as e:
        st.write(f"Debug: Secrets error: {e}")
    
    # Fallback to environment variable (for local development)
    env_key = os.getenv("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    
    return None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def query_claude(document_text, question):
    """Query Claude with document context and user question"""
    try:
        client = Anthropic(api_key=get_api_key())
        
        prompt = f"""Based on the following document content, please answer the question. If the answer cannot be found in the document, please say so.

Document content:
{document_text}

Question: {question}

Answer:"""
        
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    except Exception as e:
        return f"Error querying Claude: {str(e)}"

def main():
    st.title("📄 Document Q&A App")
    st.write("Upload a PDF document and ask questions about its content!")
    
    # Initialize session state
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'document_name' not in st.session_state:
        st.session_state.document_name = ""
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Extract text from PDF
            document_text = extract_text_from_pdf(uploaded_file)
            st.session_state.document_text = document_text
            st.session_state.document_name = uploaded_file.name
            
        st.success(f"✅ Document '{uploaded_file.name}' processed successfully!")
        
        # Show document stats
        word_count = len(document_text.split())
        char_count = len(document_text)
        st.info(f"📊 Document stats: {word_count} words, {char_count} characters")
        
        # Show preview of document content
        with st.expander("📖 Document Preview"):
            st.text_area("First 500 characters:", document_text[:500], height=100, disabled=True)
    
    # Q&A Section
    if st.session_state.document_text:
        st.divider()
        st.subheader("🤔 Ask Questions")
        
        question = st.text_input("What would you like to know about the document?", 
                                placeholder="e.g., What is the main topic of this document?")
        
        if st.button("Ask Claude", type="primary"):
            if question:
                with st.spinner("Getting answer from Claude..."):
                    try:
                        answer = query_claude(st.session_state.document_text, question)
                        st.success("🎯 Answer:")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Make sure you have set your ANTHROPIC_API_KEY in the .env file")
            else:
                st.warning("Please enter a question!")
    else:
        st.info("👆 Please upload a PDF document to start asking questions")

if __name__ == "__main__":
    # Check for API key
    if not get_api_key():
        st.error("❌ ANTHROPIC_API_KEY not found. Please set it in your .env file or Streamlit secrets")
        st.stop()
    
    main()