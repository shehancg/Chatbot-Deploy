from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from langchain_community.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env file
load_dotenv()

# Initialize API
app = FastAPI()

# Now you can access the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)

# Load existing Pinecone index
index_name = "mindchatbotstaging4"
embeddings = OpenAIEmbeddings()
index = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# Create a query model
class Query(BaseModel):
    question: str

# Define the system prompt
system_prompt = """You are Mindfullness Chatbot, well-versed in the teachings of mindfulness and well-being.
You must only answer questions related to the book think like a monk by Jay Shetty. Do not answer questions that are not related to
to the book think like a monk by Jay Shetty

{question}

Relevant information from the book:
{context}

Answer:"""

# Create the prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template=system_prompt
)

# Define API endpoint
@app.post("/query")
async def query_model(query: Query, request: Request):
    logger.info(f"Received query: {query.question}")
    try:
        # Retrieve relevant documents
        docs = index.similarity_search(query.question)
        
        if not docs:
            logger.warning("No relevant documents found")
            raise HTTPException(status_code=404, detail="No relevant documents found")
        
        # Extract the content from the documents
        context = "\n".join([doc.page_content for doc in docs])
        
        # Use the QA chain to answer the query
        llm = OpenAI(temperature=0, api_key=OPENAI_API_KEY)
        chain = LLMChain(llm=llm, prompt=prompt_template)
        answer = chain.run(question=query.question, context=context)
        
        logger.info(f"Generated answer: {answer}")
        return {"answer": answer}
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    logger.error(f"404 error: {request.url}")
    return {"detail": "The requested resource was not found"}

@app.exception_handler(500)
async def custom_500_handler(request: Request, exc: HTTPException):
    logger.error(f"500 error: {str(exc)}")
    return {"detail": "An internal server error occurred"}

# Start the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)