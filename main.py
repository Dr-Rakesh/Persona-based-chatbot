from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import openai
import logging
import uvicorn
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

# Initialize FastAPI app
app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔹 Azure OpenAI Configuration
credential = DefaultAzureCredential()
openai.api_type = "azure_ad"
openai.api_base = "https://openai-aiattack-msa-001758-westus-adi-02.openai.azure.com"
openai.api_version = "2024-08-01-preview"
openai.api_key = credential.get_token("https://cognitiveservices.azure.com/.default").token
deployment_name = "gpt-4o"

# 🔹 Azure Cognitive Search Configuration
search_key = "qbkHiwv1bloqJxJYlVbGVsT9hqvPIvxtDrqPaHXpvkAzSeB2EEPS"
endpoint = "https://cs-adt-08.search.windows.net"
index_name = "tc_nx_2412_ada3_index"

# 🔹 Initialize Azure Cognitive Search Client
search_client = SearchClient(
    endpoint=endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_key)
)

class Source(BaseModel):
    title: str
    url: str
    content: str = ""
    relevance: float = 0.0

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    function: Optional[str] = None
    domain: Optional[str] = None
    industry: Optional[str] = None
    access_level: Optional[str] = None

class SearchResult(BaseModel):
    answer: str
    sources: List[Source]
    context_used: List[str] = []

def hybrid_search(query: str, top_k: int = 3) -> tuple[List[str], List[Source]]:
    """Perform hybrid search using Azure Cognitive Search with improved relevance."""
    try:
        search_results = search_client.search(
            search_text=query,
            select=["content", "title", "url"],
            top=top_k,
            query_type="full",
            highlight_fields="content",
            highlight_pre_tag="<hit>",
            highlight_post_tag="</hit>"
        )

        results = []
        sources = []

        for result in search_results:
            content = result.get('content', '')
            title = result.get('title', '')
            url = result.get('url', '')
            score = result.get('@search.score', 0)

            if score > 0.3:
                context_entry = (
                    f"Title: {title}\n"
                    f"Content: {content}\n"
                    f"URL: {url}\n"
                    f"Relevance Score: {score}"
                )
                results.append(context_entry)
                
                source = Source(
                    title=title,
                    url=url,
                    content=content,
                    relevance=score
                )
                sources.append(source)

        return results, sources

    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail="Search operation failed")

def get_role_based_prompt(function: Optional[str], domain: Optional[str],
                          industry: Optional[str], access_level: Optional[str]) -> str:
    """Generate role-specific system prompt based on selected roles."""
    base_prompt = (
        "You are a knowledgeable assistant that provides accurate, well-structured answers "
        "based on the given context. Include source references using markdown links [Title](URL) "
        "when citing specific information."
    )

    role_specifications = []

    role_prompts = {
        "Engineer/Designer": "Provide detailed technical explanations, focusing on design specifications and engineering principles.",
        "System Admin": "Focus on system architecture, maintenance procedures, and technical implementation details.",
        "Program Manager": "Emphasize project management aspects, timelines, and high-level strategic considerations.",
        "Non-Engineers": "Explain technical concepts in simple terms, avoiding complex technical jargon."
    }
    domain_prompts = {
        "Mechanical": "Focus on mechanical engineering principles and physical systems.",
        "Electrical": "Emphasize electrical systems, circuits, and power distribution concepts.",
        "Manufacturing": "Focus on manufacturing processes, production systems, and quality control.",
        "System Simulation": "Emphasize simulation methodologies, modeling approaches, and system analysis."
    }
    industry_prompts = {
        "Automotive": "Frame answers in the context of automotive industry standards and practices.",
        "Aerospace": "Consider aerospace regulations, safety requirements, and industry specifications.",
        "Heavy Equipment": "Focus on heavy machinery considerations and industrial applications."
    }
    access_prompts = {
        "Advanced": "Provide in-depth technical details and advanced concepts.",
        "Administrator": "Include system administration and configuration details.",
        "Senior": "Include strategic considerations and best practices.",
        "Basic": "Focus on fundamental concepts and basic explanations."
    }

    role_specifications.extend([
        role_prompts.get(function, ""),
        domain_prompts.get(domain, ""),
        industry_prompts.get(industry, ""),
        access_prompts.get(access_level, "")
    ])

    return f"{base_prompt} {' '.join(filter(None, role_specifications))}"

def generate_answer(query: str, context: List[str], sources: List[Source], request: QueryRequest) -> str:
    """Generate an answer using Azure OpenAI with improved context understanding."""
    try:
        system_prompt = get_role_based_prompt(
            request.function,
            request.domain,
            request.industry,
            request.access_level
        )

        # Create a source reference guide
        source_guide = "\nAvailable Sources:\n"
        for idx, source in enumerate(sources, 1):
            source_guide += f"{idx}. [{source.title}]({source.url})\n"

        # Enhanced prompt for better intent detection and source citation
        prompt = (
            f"Based on the following context and the user's query, provide a comprehensive and accurate answer.\n\n"
            f"User Query: {query}\n\n"
            f"Available Context:\n{'\n\n'.join(context)}\n"
            f"{source_guide}\n"
            f"Instructions:\n"
            f"1. First, identify the main intent of the query.\n"
            f"2. Then, extract relevant information from the context.\n"
            f"3. When citing information, use markdown links to reference sources: [Title](URL)\n"
            f"4. Provide a well-structured answer that directly addresses the user's intent.\n"
            f"5. If the context doesn't contain sufficient information, clearly state what information is missing.\n"
            f"6. Use appropriate technical depth based on the user's role and access level.\n\n"
            f"Answer format:\n"
            f"- Start with a direct response to the main question\n"
            f"- Provide supporting details and examples from the context\n"
            f"- Include relevant technical specifications when appropriate\n"
            f"- Give an well explained implementation based example.\n"
            f"- Reference sources using markdown links\n"
            f"- End with any important caveats or considerations"
        )

        response = openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.3
        )

        return response.choices[0].message['content'].strip() if response.choices else "No valid response."

    except Exception as e:
        logger.error(f"OpenAI error: {str(e)}")
        raise HTTPException(status_code=500, detail="Answer generation failed")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/rag/query", response_model=SearchResult)
async def query_rag(request: QueryRequest):
    """RAG endpoint that combines search and answer generation."""
    try:
        logger.info(f"Processing query: {request.query}")

        # Retrieve relevant documents
        context, sources = hybrid_search(request.query, request.top_k)

        if not context:
            return SearchResult(
                answer="I could not find any relevant information.",
                sources=[],
                context_used=[]
            )

        # Generate answer using retrieved context
        answer = generate_answer(request.query, context, sources, request)

        return SearchResult(
            answer=answer,
            sources=sources,
            context_used=context
        )

    except Exception as e:
        logger.error(f"RAG query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint that verifies connections to Azure services."""
    try:
        search_client.get_document_count()
        openai.ChatCompletion.create(
            engine=deployment_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return {"status": "healthy", "services": {"azure_search": "connected", "azure_openai": "connected"}}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.get("/check-schema")
async def check_schema():
    """Check the schema of the Azure Cognitive Search index."""
    try:
        index = search_client._client.get_index(name=index_name)
        return {"fields": [field.name for field in index.fields]}
    except Exception as e:
        logger.error(f"Schema check error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


#=====================================END OF CODE==========================================

# from fastapi import FastAPI, HTTPException
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# from typing import List, Optional
# import openai
# import logging
# import uvicorn
# from azure.search.documents import SearchClient
# from azure.core.credentials import AzureKeyCredential
# from azure.identity import DefaultAzureCredential

# # Initialize FastAPI app
# app = FastAPI()

# # Mount the static directory
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # 🔹 Azure OpenAI Configuration (Hardcoded)
# credential = DefaultAzureCredential()
# openai.api_type = "azure_ad"
# openai.api_base = "https://openai-aiattack-msa-001758-westus-adi-02.openai.azure.com"
# openai.api_version = "2024-08-01-preview"
# openai.api_key = credential.get_token("https://cognitiveservices.azure.com/.default").token
# deployment_name = "gpt-4o"

# # 🔹 Azure Cognitive Search Configuration (Hardcoded)
# search_key = "qbkHiwv1bloqJxJYlVbGVsT9hqvPIvxtDrqPaHXpvkAzSeB2EEPS"
# endpoint = "https://cs-adt-08.search.windows.net"
# index_name = "tc_nx_capital_index"

# # 🔹 Initialize Azure Cognitive Search Client
# search_client = SearchClient(
#     endpoint=endpoint,
#     index_name=index_name,
#     credential=AzureKeyCredential(search_key)
# )


# class QueryRequest(BaseModel):
#     query: str
#     top_k: Optional[int] = 3
#     function: Optional[str] = None
#     domain: Optional[str] = None
#     industry: Optional[str] = None
#     access_level: Optional[str] = None


# class SearchResult(BaseModel):
#     answer: str
#     sources: List[str]

# def hybrid_search(query: str, top_k: int = 3):
#     """Perform hybrid search using Azure Cognitive Search with improved relevance."""
#     try:
#         # Corrected search parameters (Removed 'query_speller')
#         search_results = search_client.search(
#             search_text=query,
#             select=["content", "title"],
#             top=top_k,
#             query_type="full",  # Use 'full' instead of 'semantic' if needed
#             highlight_fields="content",
#             highlight_pre_tag="<hit>",
#             highlight_post_tag="</hit>"
#         )

#         results = []
#         sources = []

#         for result in search_results:
#             content = result.get('content', '')
#             title = result.get('title', '')
#             score = result.get('@search.score', 0)

#             # Only include results with reasonable relevance
#             if score > 0.3:  # Adjust threshold as needed
#                 context_entry = f"Title: {title}\nContent: {content}\nRelevance: {score}"
#                 results.append(context_entry)
#                 sources.append(title)

#         return results, sources

#     except Exception as e:
#         logger.error(f"Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Search operation failed")


# def get_role_based_prompt(function: Optional[str], domain: Optional[str],
#                           industry: Optional[str], access_level: Optional[str]) -> str:
#     """Generate role-specific system prompt based on selected roles."""
#     base_prompt = "You are a knowledgeable assistant that provides accurate, well-structured answers based on the given context."

#     role_specifications = []

#     role_prompts = {
#         "Engineer/Designer": "Provide detailed technical explanations, focusing on design specifications and engineering principles.",
#         "System Admin": "Focus on system architecture, maintenance procedures, and technical implementation details.",
#         "Program Manager": "Emphasize project management aspects, timelines, and high-level strategic considerations.",
#         "Non-Engineers": "Explain technical concepts in simple terms, avoiding complex technical jargon."
#     }
#     domain_prompts = {
#         "Mechanical": "Focus on mechanical engineering principles and physical systems.",
#         "Electrical": "Emphasize electrical systems, circuits, and power distribution concepts.",
#         "Manufacturing": "Focus on manufacturing processes, production systems, and quality control.",
#         "System Simulation": "Emphasize simulation methodologies, modeling approaches, and system analysis."
#     }
#     industry_prompts = {
#         "Automotive": "Frame answers in the context of automotive industry standards and practices.",
#         "Aerospace": "Consider aerospace regulations, safety requirements, and industry specifications.",
#         "Heavy Equipment": "Focus on heavy machinery considerations and industrial applications."
#     }
#     access_prompts = {
#         "Advanced": "Provide in-depth technical details and advanced concepts.",
#         "Administrator": "Include system administration and configuration details.",
#         "Senior": "Include strategic considerations and best practices.",
#         "Basic": "Focus on fundamental concepts and basic explanations."
#     }

#     role_specifications.extend([role_prompts.get(function, ""),
#                                 domain_prompts.get(domain, ""),
#                                 industry_prompts.get(industry, ""),
#                                 access_prompts.get(access_level, "")])

#     return f"{base_prompt} {' '.join(filter(None, role_specifications))}"

# def generate_answer(query: str, context: List[str], request: QueryRequest) -> str:
#     """Generate an answer using Azure OpenAI with improved context understanding."""
#     try:
#         system_prompt = get_role_based_prompt(
#             request.function,
#             request.domain,
#             request.industry,
#             request.access_level
#         )

#         # Enhanced prompt for better intent detection
#         prompt = (
#             f"Based on the following context and the user's query, provide a comprehensive and accurate answer.\n\n"
#             f"User Query: {query}\n\n"
#             f"Available Context:\n{'\n\n'.join(context)}\n\n"
#             f"Instructions:\n"
#             f"1. First, identify the main intent of the query.\n"
#             f"2. Then, extract relevant information from the context.\n"
#             f"3. Finally, provide a well-structured answer that directly addresses the user's intent.\n"
#             f"4. If the context doesn't contain sufficient information, clearly state what information is missing.\n"
#             f"5. Use appropriate technical depth based on the user's role and access level.\n\n"
#             f"Answer format:\n"
#             f"- Start with a direct response to the main question\n"
#             f"- Provide supporting details and examples from the context\n"
#             f"- Include relevant technical specifications when appropriate\n"
#             f"- End with any important caveats or considerations"
#         )

#         response = openai.ChatCompletion.create(
#             engine=deployment_name,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.3,  # Lower temperature for more focused responses
#             max_tokens=800,   # Increased token limit for more detailed responses
#             top_p=0.9,        # Nucleus sampling for better coherence
#             frequency_penalty=0.5,  # Reduce repetition
#             presence_penalty=0.3    # Encourage addressing different aspects
#         )

#         return response.choices[0].message['content'].strip() if response.choices else "No valid response."

#     except Exception as e:
#         logger.error(f"OpenAI error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Answer generation failed")


# @app.get("/")
# async def read_root():
#     return FileResponse("static/index.html")


# @app.post("/rag/query", response_model=SearchResult)
# async def query_rag(request: QueryRequest):
#     """RAG endpoint that combines search and answer generation."""
#     try:
#         logger.info(f"Processing query: {request.query}")

#         # Retrieve relevant documents
#         context, sources = hybrid_search(request.query, request.top_k)

#         if not context:
#             return SearchResult(answer="I could not find any relevant information.", sources=[])

#         # Generate answer using retrieved context
#         answer = generate_answer(request.query, context, request)

#         return SearchResult(answer=answer, sources=sources)

#     except Exception as e:
#         logger.error(f"RAG query error: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))


# @app.get("/health")
# async def health_check():
#     """Health check endpoint that verifies connections to Azure services."""
#     try:
#         search_client.get_document_count()
#         openai.ChatCompletion.create(
#             engine=deployment_name,
#             messages=[{"role": "user", "content": "test"}],
#             max_tokens=5
#         )
#         return {"status": "healthy", "services": {"azure_search": "connected", "azure_openai": "connected"}}
#     except Exception as e:
#         logger.error(f"Health check failed: {str(e)}")
#         return {"status": "unhealthy", "error": str(e)}


# @app.get("/check-schema")
# async def check_schema():
#     """Check the schema of the Azure Cognitive Search index."""
#     try:
#         index = search_client._client.get_index(name=index_name)
#         return {"fields": [field.name for field in index.fields]}
#     except Exception as e:
#         logger.error(f"Schema check error: {str(e)}")
#         return {"error": str(e)}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    
    
    
    
    
    
# def hybrid_search(query: str, top_k: int = 3):
#     """Perform hybrid search using Azure Cognitive Search."""
#     try:
#         search_results = search_client.search(
#             search_text=query,
#             select=["content", "title"],
#             top=top_k
#         )

#         results = []
#         sources = []

#         for result in search_results:
#             content = result.get('content', '')
#             title = result.get('title', '')

#             context_entry = f"Title: {title}\nContent: {content}"
#             results.append(context_entry)
#             sources.append(title)

#         return results, sources

#     except Exception as e:
#         logger.error(f"Search error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Search operation failed")

# def generate_answer(query: str, context: List[str], request: QueryRequest) -> str:
#     """Generate an answer using Azure OpenAI with retrieved context and role-based prompting."""
#     try:
#         system_prompt = get_role_based_prompt(
#             request.function,
#             request.domain,
#             request.industry,
#             request.access_level
#         )

#         prompt = (
#             f"Based on the following context, provide a comprehensive and accurate answer to the question."
#             f"\n\nContext:\n{'\n\n'.join(context)}\n\n"
#             f"Question: {query}\n\n"
#             f"Provide a clear and concise answer, citing specific information from the context where relevant."
#         )

#         response = openai.ChatCompletion.create(
#             engine=deployment_name,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": prompt}
#             ],
#             temperature=0.5,
#             max_tokens=500
#         )

#         return response.choices[0].message['content'].strip() if response.choices else "No valid response."

#     except Exception as e:
#         logger.error(f"OpenAI error: {str(e)}")
#         raise HTTPException(status_code=500, detail="Answer generation failed")
