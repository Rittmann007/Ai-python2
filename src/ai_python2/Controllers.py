from ai_python2.Models import InterviewRequest,ChatRequest
from ai_python2.UtilFuncs import chunk_text
from ai_python2.UtilFuncs import get_chunk_embedding,get_query_results
from fastapi import Request,HTTPException
from fastapi.responses import JSONResponse
from bson import ObjectId
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
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
    resumeChunk = chunk_text(payload.resumeText, chunk_size=400, chunk_overlap=50)
    jdChunk = chunk_text(payload.jobDescription, chunk_size=400, chunk_overlap=50)
    sdChunk = chunk_text(payload.selfDescription, chunk_size=400, chunk_overlap=50)

    # get the embedding and create the doc to insert
    resumeDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "sourceType": "resume",
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in resumeChunk
    ]
    jdDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "sourceType": "jobDescription",
            "text" : chunk,
            "embedding" : get_chunk_embedding(chunk)
        } for chunk in jdChunk
    ]
    sdDocs = [
        {   
            "interviewID" : ObjectId(payload.interviewID),
            "sourceType": "selfDescription",
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

    #tool definition
    web_search_tool = TavilySearch(max_results=3)

    interview_id = payload.interviewID  # closure captures these, LLM never sees them

    @tool
    def search_resume(query: str) -> str:
        """Search only the candidate's resume for relevant information."""
        docs = get_query_results(query, collection, interview_id, source_type="resume")
        return "\n\n".join(d["text"] for d in docs)# returning the chunks into a single string
    
    @tool
    def search_job_description(query: str) -> str:
        """Search the job description/posting for this interview."""
        docs = get_query_results(query, collection, interview_id, source_type="jobDescription")
        return "\n\n".join(d["text"] for d in docs)
    
    @tool
    def search_self_description(query: str) -> str:
        """Search the candidate's own self-description for this interview."""
        docs = get_query_results(query, collection, interview_id, source_type="selfDescription")
        return "\n\n".join(d["text"] for d in docs)

    tools = [web_search_tool, search_resume, search_job_description, search_self_description]

    prompt = """
    You are an interview-prep assistant helping a candidate understand and reflect on a specific interview.
    
    You have three tools to search the candidate's stored interview data, plus web search. Always pick the most specific tool for the question — never guess or answer from general knowledge when a tool can give you the real data.
    
    - search_resume: use for anything about the candidate's own background — work history, employer, job titles, skills, education, projects, experience.
    - search_job_description: use for anything about the role being interviewed for — responsibilities, requirements, qualifications, location/work type (remote/hybrid/onsite), salary, or what the employer is looking for.
    - search_self_description: use for anything about how the candidate describes themselves, their goals, strengths, or personal pitch.
    - web_search_tool (Tavily): use only when the question needs current/public information that isn't part of this candidate's stored data (e.g. "what does this company do", "is React 19 out yet").
    
    Rules:
    1. Always call the relevant search tool before answering a question about the candidate, the job, or the interview — never answer from assumption or general knowledge.
    2. If a question could relate to more than one source (e.g. "am I a good fit for this role"), call multiple tools and combine what you find.
    3. If the tool returns no relevant information, say so plainly — do not fabricate details or infer things that aren't in the retrieved text.
    4. Keep answers grounded strictly in what the tools return. Do not blend information from one source into another (e.g. don't attribute job description details to the candidate's own experience).
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