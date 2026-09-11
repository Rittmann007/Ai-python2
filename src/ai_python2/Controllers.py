from ai_python2.Models import InterviewRequest,ChatRequest
from ai_python2.UtilFuncs import chunk_text
from ai_python2.UtilFuncs import get_chunk_embedding,get_query_results
from fastapi import Request,HTTPException
from fastapi.responses import JSONResponse
from bson import ObjectId
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_agent


# payload is the JSON body the client sends to your endpoint. request is the full FastAPI request object.
# In controller:
# payload: InterviewRequest gives you the data fields like interviewID, userID, resumeText, etc.
# request: Request gives you access to the HTTP request itself, including request.app.state.mongo_client
# use request when you need app state, headers, query params, client info, and similar request-level details
async def ingestController(payload:InterviewRequest,request:Request):
    client = request.app.state.mongo_client
    collection=client["Genai_resumeChecker"]["ragCollection"]

    if not ObjectId.is_valid(payload.interviewID):
      raise HTTPException(status_code=400, detail="Invalid interviewID")

    # chunk the input
    resumeChunk = chunk_text(payload.resumeText, chunk_size=200, chunk_overlap=30)
    jdChunk = chunk_text(payload.jobDescription, chunk_size=200, chunk_overlap=30)
    sdChunk = chunk_text(payload.selfDescription, chunk_size=200, chunk_overlap=30)

    # get the embedding and create the doc to insert
    resumeDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in resumeChunk
    ]
    jdDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in jdChunk
    ]
    sdDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in sdChunk
    ]

    collection.insert_many(resumeDocs)
    collection.insert_many(jdDocs)
    collection.insert_many(sdDocs)

    return JSONResponse(status_code=200, content={"message": "ingested successfully"})


# chat
async def chatController(payload: ChatRequest, request: Request):
    client = request.app.state.mongo_client
    collection = client["Genai_resumeChecker"]["ragCollection"]
    checkpointer = request.app.state.checkpointer

    if not ObjectId.is_valid(payload.interviewID):
        raise HTTPException(status_code=400, detail="Invalid interviewID")

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")  
    web_search_tool = TavilySearchResults(max_results=3)

    interview_id = payload.interviewID  # closure captures these, LLM never sees them

    @tool
    def search_resume_context(query: str) -> str:
        """Search the candidate's resume/interview context for relevant information."""
        docs = get_query_results(query, collection, interview_id)
        for d in docs:
            print(d.get("interview_id"), d.get("_id"), d["text"][:100])
        return "\n\n".join(doc["text"] for doc in docs)

    tools = [web_search_tool, search_resume_context]

    prompt = """
    You are a helpful assistant.
    You can use internal resume context and web search when needed.

    Use the resume context first when the question is about the interview data.
    Use web search only when the answer is not available in the resume context or when the user asks for current/public information.
    """

    agent = create_agent(
        llm, 
        tools=tools, 
        system_prompt=prompt,
        checkpointer=checkpointer
    )

    config = {"configurable": {"thread_id": payload.sessionID}}

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": payload.message}]},
        config=config,
    )

    return {
        "message": "response given successfully",
        "response": result["messages"][-1].content,
    }