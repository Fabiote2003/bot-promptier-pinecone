import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec, list_indexes
import uuid

def ingest():
    # 1. Cargar variables de entorno desde .env
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")

    
    if not all([openai_api_key, pinecone_api_key]):
        raise ValueError("Asegúrate de que las variables OPENAI_API_KEY, PINECONE_API_KEY y PINECONE_ENVIRONMENT estén configuradas en .env")

    # 2. Inicializar Pinecone
    pc = Pinecone(
        api_key=pinecone_api_key
    )
    
    # 3. Cargar tu archivo .md con la información
    loader = TextLoader("promptier_info.md", encoding="utf-8")
    documents = loader.load()
    
    # 4. Dividir el texto en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    # 5. Crear embeddings usando OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    #6. Crear el índice en Pinecone si no existe
    index_name = "promptier-index"
    print(pc.list_indexes())
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=1536, 
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            )
        )
    
    # 7. Ingestar los documentos en Pinecone
    index = pc.Index(index_name)
    
    records = []
    for doc in docs:
        vector = embeddings.embed_query(doc.page_content)
        record = {
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": {
                'text': doc.page_content,
                'source': doc.metadata.get('source', '')
            }
        }
        records.append(record)
    
    index.upsert(
        vectors=records,
        namespace="promptier-namespace"
    )
    
    print("¡Ingesta completa! Los documentos se han indexado en Pinecone.")

if __name__ == "__main__":
    ingest()