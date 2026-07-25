

import config
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import tools

def run_agent(
            question: str, 
            collection:chromadb.Collection, 
            encoder: SentenceTransformer
            ) -> str:                                            

    messages = [ #models working memory chain. we append to this as we go                                  
        {"role": "system", "content": config.TOOL_PROMPT},                             
        {"role": "user", "content": question}      
    ]                                              
                                                
    # first call — model decides whether to use a tool                                         
    response = ollama.chat(                        
        model=config.MODEL,                       
        messages=messages,                     
        tools=config.TOOLS                        
    )

    # if no tool call, model answered directly (e.g. a trap question)
    if not response.message.tool_calls:
      return "I don't have information on that in the available documents.", [] 
                                                
    # tool was called — extract the query argument 
    tool_call = response.message.tool_calls[0]
    query = tool_call.function.arguments["query"]  

    #embed question and get context
    q_emb = encoder.encode(query)     

    # run retrieval chunks, 
    chunks, sources = tools.retrieve(q_emb, config.TOP_K, collection)                              

    # format chunks with sources for context       
    context = "\n---\n".join(f"[{src}]\n{chunk}" for src, chunk in zip(sources, chunks))            
                                                
    # second call — model generates answer using retrieved context                                
    messages.append({"role": "assistant",      
    "content": "", "tool_calls":                       
    response.message.tool_calls})
    messages.append({"role": "tool", "content":
    context, "name": "search_knowledge_base"})
    messages.append({"role": "system", "content": config.ANSWER_PROMPT})

    final = ollama.chat(
        model=config.MODEL,
        messages=messages
        )                                                      
       
    return final.message.content, sources