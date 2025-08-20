import streamlit as st
from langchain_community.vectorstores import FAISS

from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_huggingface import HuggingFaceEmbeddings

# Set Streamlit Page Config
st.set_page_config(page_title="🩺 Medical Chatbot (AI Doctor)", layout="centered")

st.title("🩺 Ask your Medical Questions")
st.caption("Built using FAISS + Mistral + RAG")

@st.cache_resource(show_spinner=True)
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)

@st.cache_resource(show_spinner=True)
def load_llm_pipeline():
    model_id = "google/flan-t5-small"   # works on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    gen_pipeline = pipeline(
        "text2text-generation",  # <- for seq2seq models like T5
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)

#############

# Load vectorstore and LLM
vectorstore = load_vectorstore()
llm = load_llm_pipeline()

# Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
        "k": 3,                # fetch only top 3 results
        "score_threshold": 0.7  # ignore irrelevant matches
        }
    ),
    return_source_documents=True
)

# Streamlit Input
query = st.text_input("❓ Ask your question here:")
if query:
    with st.spinner("Generating Answer..."):
        result = qa_chain.invoke(query)
        st.markdown("### ✅ Answer:")
        st.success(result['result'])

        with st.expander("📚 Sources"):
            for i, doc in enumerate(result['source_documents']):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content[:400] + "...")
