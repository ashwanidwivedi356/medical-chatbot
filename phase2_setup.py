import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ Step 1: Load Vector Store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

# ✅ Step 2: Load Mistral LLM from HuggingFace
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # requires GPU or will use CPU if no GPU
    trust_remote_code=True,
)

# Create LLM pipeline
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ✅ Step 3: Create Retrieval QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Can use "map_reduce" or "refine" for large contexts
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# ✅ Test Query
query = "डायबिटीज के मरीज को कौन से फल खाने चाहिए?"
response = qa_chain.invoke(query)

print("Answer:")
print(response['result'])

# Optional: Show source docs
print("\nSources:")
for doc in response['source_documents']:
    print(">>>", doc.metadata.get('source', 'N/A'))
    print(doc.page_content[:300], "...\n")
