from dotenv import load_dotenv
from pydantic import BaseModel
# ✅ Correct imports
from langchain_google_genai import ChatGoogleGenerativeAI  # ← Gemini import

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

# Load environment variables (expects GOOGLE_API_KEY in .env)
load_dotenv()

# ✅ Define structured response model
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# ✅ Initialize Gemini model (no OpenAI or Anthropic key needed)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # You can also use "gemini-1.5-pro"
    temperature=0.7
)

# ✅ Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that helps generate structured research summaries.
            Use tools when needed and respond strictly in this JSON format:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ✅ Define tools and agent
tools = [search_tool, wiki_tool, save_tool]

agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(agent=agent, verbose=True, tools=tools)

# ✅ Run the agent
query = input("What can I help you research? ")
raw_response = agent_executor.invoke({"query": query})

# try:
#     structured_response = parser.parse(raw_response.get("output")[0]["text"])
#     print(structured_response)
# except Exception as e:
#     print("Error parsing response:", e, "\nRaw response:", raw_response)
try:
    output_text = raw_response.get("output")

    # Remove Markdown formatting if present (```json ... ```)
    if output_text.startswith("```"):
        output_text = output_text.strip("`")
        if "json" in output_text:
            output_text = output_text.split("json", 1)[-1].strip()

    structured_response = parser.parse(output_text)
    print(structured_response)

except Exception as e:
    print("Error parsing response:", e, "\nRaw response:", raw_response)
