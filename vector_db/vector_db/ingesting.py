import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()



if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("/home/echeadle/Working_on/udemy-eden-langchain/intro_to_vectro_dbs/mediumblog1.txt")
    document = loader.load()
    
    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts =text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")
    
    emdbeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    
    print("Ingesting...")
    PineconeVectorStore.from_documents(texts,emdbeddings, index_name=os.environ['INDEX_NAME'])
    print("finished")