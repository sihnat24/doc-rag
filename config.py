

#model related
TOOL_PROMPT = "You are a program analyst assistant. You have access to a local knowledge base. Always use the search_knowledge_base tool before answering any question — do not answer from memory."
ANSWER_PROMPT = "You are a program analyst assistant. Answer the question using only the retrieved document excerpts provided. Always cite the source document. If the excerpts do not contain the answer, explicitly say 'I don't have information on that in the available documents.' Do not use outside knowledge."
MODEL = "llama3.1"


#retrieval
TOP_K = 3
COLLECTION = "curated_files"

#caffeine corpus
CAFFEINE_COLLECTION = "caffeine_papers"
CAFFEINE_DATA_DIR = "data/caffeine"
CAFFEINE_METADATA_CSV = "data/caffeine/metadata.csv"

#encoder
ENCODER = "all-MiniLM-L6-v2"

#data extraction
DATA_DIR = 'data'
DATA_TYPES = ['.docx','.html','.pdf','.csv']

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#PDF relateed
PIXEL_THRESH = 15

#tools

TOOLS = [                                          
      {                                            
          "type": "function",                    
          "function": {                             
              "name": "search_knowledge_base",
              "description": "Search local database for information relevant to query.",                  
              "parameters": {
                  "type": "object",                  
                  "properties": {                  
                      "query": {                 
                          "type": "string",         
                          "description": "question user wants answered using proprietary local data"
                      }                              
                  },
                  "required": ["query"]              
              }                                    
          }                                      
      }                                             
  ]
