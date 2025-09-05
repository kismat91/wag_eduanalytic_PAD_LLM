import os
import streamlit as st
import pandas as pd
import PyPDF2
import re
from io import BytesIO
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Check if API keys are loaded
if not os.getenv("OPENAI_API_KEY"):
    st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")

if not os.getenv("GROQ_API_KEY"):
    st.warning("⚠️ Groq API key not found. Please set GROQ_API_KEY in your .env file")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

st.title("PAD Activity Definition Matcher")

model_choice = st.selectbox("Choose a model to run:", ["OpenAI GPT-4o-mini", "DeepSeek LLaMA-70B"])

if model_choice == "OpenAI GPT-4o-mini":
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
elif model_choice == "DeepSeek LLaMA-70B":
    model = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=1024,
        timeout=None,
        max_retries=2
    )

pdf_file = st.file_uploader("Upload PAD PDF", type=["pdf"])
csv_file = st.file_uploader("Upload Activities File", type=["xlsx", "xls", "csv"])

def calculate_similarity_score(definition, matched_content):
    """Calculate similarity score between definition and matched content"""
    if matched_content == "NO RELEVANT CONTEXT FOUND" or not matched_content.strip():
        return 0.0
    
    # Clean the matched content (remove asterisks and extra whitespace)
    cleaned_content = re.sub(r'^\*\s*', '', matched_content, flags=re.MULTILINE)
    cleaned_content = cleaned_content.strip()
    
    if not cleaned_content:
        return 0.0
    
    try:
        # Use TF-IDF vectorization to calculate cosine similarity
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([definition.lower(), cleaned_content.lower()])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Convert to percentage and round to 2 decimal places
        score = round(similarity * 100, 2)
        return score
        
    except Exception as e:
        # Fallback to simple word overlap scoring if TF-IDF fails
        definition_words = set(definition.lower().split())
        content_words = set(cleaned_content.lower().split())
        
        if len(definition_words) == 0:
            return 0.0
            
        overlap = len(definition_words.intersection(content_words))
        score = round((overlap / len(definition_words)) * 100, 2)
        return score

if pdf_file and csv_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    raw_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])

    #headings = ["A. PDO", "Project Beneficiaries", "A. Project Components", "B. Project Financing", "Project Cost and Financing", "Annex 2: Detailed Project Description"]
    #headings = ["A. Project Development Objectives (PDO)", "A. Programs to be supported", "B. Project Financing", "Project Cost and Financing", "Annex 2: Detailed Project Description"]
    #headings = [
    
    #"A. Project Development Objective",
    #"B. Project Components",
    #"C. Project Beneficiaries",
    #"D. Results Chain"
    #"D. Theory of Change"

    #]
    headings = [
    "A. PDO",
    "B. Project Beneficiaries",
    "A. Project Components",
    "B. Project Cost and Financing",
    "ANNEX 1B: THEORY OF CHANGE AND RESULTS CHAIN"
]
   

    pattern = rf"({'|'.join(re.escape(h) for h in headings)})(.*?)(?=\n(?:[IVXLCDM\\d]+\\.?\\s+)?[A-Z][A-Z\\s]+\n|Annex\\s+\\d+|$)"
    #pattern = rf"({'|'.join(headings)})(.*?)(?=\n(?:[IVXLCDM\d]+\.?\s+)?[A-Z][A-Z\s]+\n|Annex\s+\d+|$)"

    matches = re.findall(pattern, raw_text, re.DOTALL)

    content_dict = {}
    for heading, content in matches:
        content = re.sub(r'\n\d+\s*$', '', content.strip(), flags=re.MULTILINE)
        content = re.sub(r'(\n\d+[\t ]+\S+.*?(\n|$))+', '', content)
        content = re.sub(r'(\b[A-Z]+\b[\t ]+\d+(\.\d+)?[\t ]+\S+.*?(\n|$))+', '', content)
        content_dict[heading.strip()] = content.strip()

    extracted_text = "\n".join(content_dict.values())

    docs = [Document(page_content=extracted_text)]
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=20).split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", k=5)

    def load_definitions(file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Ensure required columns exist
        if 'Activity Name' not in df.columns or 'Definition' not in df.columns:
            st.error("Uploaded file must contain 'Activity Name' and 'Definition' columns.")
            st.stop()

        return df[['Activity Name', 'Definition']].dropna().reset_index(drop=True)

    definitions_df = load_definitions(csv_file)
    definitions_df = definitions_df.drop_duplicates(subset=["Activity Name", "Definition"])

    system_template = """
Reminder:
DO NOT PRINT WHAT YOU THINK
DO NOT GIVE REASON FOR YOUR ANSWERS AT ALL.
Start each of the retrieved sentence with a '*' character
You are an expert in World Bank Global Education and education policy analysis.

Your task is to determine if the activity name and definition provided in the query  aligns with relevant content given by the user in the context or not.
Very carefully understand the activity name and definition to retrieve relevant content as there might be lot of irrelevant content. You need not fetch that and state "NO RELEVANT CONTEXT FOUND" for that.(DO NOT GIVE REASON FOR YOUR ANSWERS AT ALL.)
Your task is to accurately retrieve text ONLY from the provided context(relevant content given by the user) that aligns with a given definition based on semantic meaning.

Instructions:
Your response must consist solely of exact sentences from the provided context. Do not generate new sentences, rephrase, summarize, or add external information.
Extract at max 3 sentences that best matches the meaning of the given definition from the provided relevant content.
If no relevant content exists, respond with: "NO RELEVANT CONTEXT FOUND", without any explanation. DO NOT MAKE ANS OF YOUR OWN.
Ensure that the extracted text from the relevant content is contextually and semantically aligning with the definition. Do not infer meaning beyond what is explicitly stated in the context.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", """
Context:
{context}

Definition:
{definition}

Remember:
Only return exact sentences from the above context. Do not add thinking or analysis. Begin each sentence with '*'. Limit output to top 3 relevant sentences only. No summaries, no rephrasing.
""")
    ])

    chain = prompt_template | model | StrOutputParser()

    results = []
    for _, row in definitions_df.iterrows():
        definition = row['Definition']
        activity_name = row['Activity Name']
        docs = retriever.get_relevant_documents(definition)
        context = "".join([doc.page_content for doc in docs])
        if not context.strip():
            matched_content = "NO RELEVANT CONTEXT FOUND"
            similarity_score = 0.0
        else:
            try:
                output = chain.invoke({"context": context, "definition": definition})
                matched_content = output.strip()
            except Exception as e:
                st.warning(f"Model failed: {e}")
                if model_choice == "DeepSeek LLaMA-70B":
                    st.info("Retrying with OpenAI GPT-4o-mini instead...")
                    fallback_model = init_chat_model("gpt-4o-mini", model_provider="openai")
                    fallback_chain = prompt_template | fallback_model | StrOutputParser()
                    matched_content = fallback_chain.invoke({"context": context, "definition": definition}).strip()
                else:
                    matched_content = "MODEL ERROR"
            
            # Calculate similarity score
            similarity_score = calculate_similarity_score(definition, matched_content)
        
        results.append((activity_name, definition, matched_content, similarity_score))
            

    final_df = pd.DataFrame(results, columns=["Activity Name", "Definition", "Matched Content", "Similarity Score (%)"])
    
    # Sort by similarity score in descending order
    final_df = final_df.sort_values("Similarity Score (%)", ascending=False).reset_index(drop=True)
    
    st.dataframe(final_df)

    csv = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "matched_output.csv", "text/csv")