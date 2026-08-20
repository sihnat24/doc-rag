

#model related
TOOL_PROMPT = (
    "You are a program analyst assistant supporting a DoD program office. "
    "You have four tools available:\n"
    "- search_knowledge_base: search local program documents for policy, budget, risk, or program details.\n"
    "- list_knowledge_base: list all documents currently indexed — use this when asked what documents are available.\n"
    "- query_spreadsheet: run a SQL SELECT query against the example_domain_data table. "
    "Use this for precise lookups, filtering, or aggregations on paper metadata "
    "(e.g. all papers using NMR, papers by a specific author, papers in the GYF domain).\n"
    "- web_search: search the public web for general background knowledge not likely in the local documents "
    "(e.g. definitions, public policy, external standards).\n"
    "Prefer local tools over web search for program-specific questions. "
    "When neither local search nor the database contains the answer, say explicitly: "
    "'I don't have information on that in the available documents.' "
    "Only fall back to web_search if the question is clearly general knowledge."
)
ANSWER_PROMPT = "You are a program analyst assistant. Answer the question using only the retrieved document excerpts provided. Always cite the source document. If the excerpts do not contain the answer, explicitly say 'I don't have information on that in the available documents.' Do not use outside knowledge."
MODEL = "llama3.1"

from langchain_ollama import ChatOllama
LLM = ChatOllama(model=MODEL)


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
DATA_TYPES = ['.docx', '.html', '.pdf']   # prose → Chroma
TABLE_TYPES = ['.csv', '.xlsx']            # tabular → SQLite

#database
DB_PATH = "program.db"
INGEST_LOG = "ingest_log.json"

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
