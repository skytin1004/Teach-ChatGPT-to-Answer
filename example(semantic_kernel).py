# Library imports
from collections import OrderedDict
import requests

# Semantic_kernal library imports
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureTextEmbedding

# Configuration imports
from config import (
    SEARCH_SERVICE_ENDPOINT,
    SEARCH_SERVICE_KEY,
    SEARCH_SERVICE_API_VERSION,
    SEARCH_SERVICE_INDEX_NAME1,
    SEARCH_SERVICE_SEMANTIC_CONFIG_NAME,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_API_VERSION,
)

# Cognitive Search Service header settings
HEADERS = {
    'Content-Type': 'application/json',
    'api-key': SEARCH_SERVICE_KEY
}

async def search_documents(question):
    """Search documents using Azure Cognitive Search"""
    # Construct the Azure Cognitive Search service access URL
    url = (SEARCH_SERVICE_ENDPOINT + 'indexes/' +
               SEARCH_SERVICE_INDEX_NAME1 + '/docs')
    # Create a parameter dictionary
    params = {
        'api-version': SEARCH_SERVICE_API_VERSION,
        'search': question,
        'select': '*',
        '$top': 3,
        'queryLanguage': 'en-us',
        'queryType': 'semantic',
        'semanticConfiguration': SEARCH_SERVICE_SEMANTIC_CONFIG_NAME,
        '$count': 'true',
        'speller': 'lexicon',
        'answers': 'extractive|count-3',
        'captions': 'extractive|highlight-false'
        }
    # Make a GET request to the Azure Cognitive Search service and store the response in a variable
    resp = requests.get(url, headers=HEADERS, params=params)
    # Return the JSON response containing the search results
    return resp.json()
        
async def filter_documents(search_results):
    """Filter documents that score above a certain threshold in semantic search"""
    file_content = OrderedDict()
    for result in search_results['value']:
        # The '@search.rerankerScore' range is 1 to 4.00, where a higher score indicates a stronger semantic match.
        if result['@search.rerankerScore'] > 1.5:
            file_content[result['metadata_storage_path']] = {
                'chunks': result['pages'][:10],
                'captions': result['@search.captions'][:10],
                'score': result['@search.rerankerScore'],
                'file_name': result['metadata_storage_name']
            }

    return file_content

async def create_kernel(sk):
    """Create a semantic kernel"""
    return sk.Kernel()

async def create_embeddings(kernel):
    """Create an embedding model"""
    return kernel.add_text_embedding_generation_service(
        "text-embedding-ada-002", # This parameter is related to the prompt templates, but is not covered in this tutorial. You can call it whatever you want.
        AzureTextEmbedding(
            "text-embedding-ada-002",
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_KEY,
            AZURE_OPENAI_API_VERSION
        ))

async def create_vector_store(kernel):
    """Create a vector store"""
    kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())
    kernel.import_skill(sk.core_skills.TextMemorySkill())

async def store_documents(kernel, file_content):
    """Store documents in the vector store"""
    for key, value in file_content.items():
        page_number = 1
        for page in value['chunks']:
            page_id = f"{value['file_name']}_{page_number}"
            await kernel.memory.save_information_async(
                collection='TeachGPTtoPDF',
                id=page_id,
                text=page
            )
            page_number += 1

async def search_with_vector_store(kernel, question):
    """Search for documents related to your question from the vector store"""
    related_page = await kernel.memory.search_async('TeachGPTtoPDF', question)
    return related_page

async def answer_with_sk(kernel, question, related_page):
    """Answer question with related_page using the semantic kernel"""

    # Add a chat service
    kernel.add_chat_service(   
        'gpt-35-turbo', # This parameter is related to the prompt templates, but is not covered in this tutorial. You can call it whatever you want.
        AzureChatCompletion(
            'gpt-35-turbo', # Azure OpenAI Deployment name
            AZURE_OPENAI_ENDPOINT,
            AZURE_OPENAI_KEY,
            AZURE_OPENAI_API_VERSION
        )
        )

    prompt = """
    Provide a detailed answer to the <question> using the information from the <related_page>.

    <question>
    {{$question}}
    </question>

    <related_page>
    {{$related_page}}
    </related_page>

    Answer:
    """
    chat_function = kernel.create_semantic_function(prompt, max_tokens=500, temperature=0.0, top_p=0.5)
    context = kernel.create_new_context()
    context['question'] = question
    context['related_materials'] = related_page[0].text
    return await chat_function.invoke_async(context=context)


async def main():

    QUESTION = 'Tell me about effective prompting strategies'

    # Search for documents with Azure Cognitive Search
    
    search_results = await search_documents(QUESTION)

    file_content = await filter_documents(search_results)

    print('Total Documents Found: {}, Top Documents: {}'.format(
        search_results['@odata.count'], len(search_results['value'])))
    
    # Answer your question using the semantic kernel

    kernel = await create_kernel(sk)

    await create_embeddings(kernel)

    await create_vector_store(kernel)

    await store_documents(kernel, file_content)

    related_page = await search_with_vector_store(kernel, QUESTION)

    answer = await answer_with_sk(kernel, QUESTION, related_page)

    print('Question: ', QUESTION)
    print('Answer: ', answer)
    print('Reference: ', related_page[0].id)

# execute the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
