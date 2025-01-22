1. Technologies and their utilization:
  - The technologies I used for the project were Python, Flask, Pinecone for vectorial databases, OpenAI API for LLM and embeddings, and Chromadb for local practice.

2. Initial Challenges and Strategic Learning:
  - At the beginning, I was really stuck because I had no idea about doing chatbots. So, I thought it would be a good strategy to read the provided documentation, which worked, but I really didn't understand the basis. Therefore, I started to read about four key concepts: RAG Architecture, embeddings, vectorial spaces and vectorial databases.

3. Project restart:
  - When I had a better understanding of the main concepts, I restarted the project. However, I faced many challenges, such as working with a vectorial database for the first time. During the entire challenge, my main focus was understanding how to get a query, embed it, and then search in the index.

4. Detailed implementation process:
  1. Document Ingestion Process (my source of knowledge):
    - Utilizes environment variables to securely handle APIs.
    - Loads a document from a .md file, splitting them into text segments (chunks).
    - Converts these text segments into embeddings using OpenAI.
    - Stores these embeddings in a Pinecone index.
  2. Chatbot Implementation:
    - Upon receiving a query, the system generates embeddings of the query and do a search in the Pinecone index to find the most relevant documents.
    - Uses OpenAI's language model to generate responses based on the retrieved documents.
    - Returns the generated response and the sources of the documents through the API.