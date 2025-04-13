
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
import asyncio
import os
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
load_dotenv(override=True)

async def search_web(query: str) -> str:
    """Useful for searching the web for information."""
    print(f"Searching for {query}")
    client = AsyncTavilyClient(os.getenv("TAVILY_API_KEY"))
    response =  await client.search(query)
    return str(response)


async def record_notes(ctx: Context, note_title: str, notes: str):
    """Useful for recording notes. Use this tool to record notes from the web. 
    Always add a title to each note you record. 
    Make sure the note titles are unique to each note. Add atlease 5 notes.
    """
    print(f"Recording notes for {note_title}")
    print(f"Notes: {notes}")
    curr_state = await ctx.get("state")
    if "notes" not in curr_state:
        curr_state["notes"] = {}
    curr_state["notes"][note_title] = notes
    await ctx.set("state", curr_state)
    return f"Notes for {note_title} recorded successfully."

async def write_report(ctx: Context, report_content: str):
    """Useful for writing a report. Use this tool to write a report from the notes you have recorded.
    Make sure the report is formatted in markdown.
    """
    print("Writing report")
    curr_state = await ctx.get("state")
    curr_state["report"] = report_content
    await ctx.set("state", curr_state)
    return "Report written successfully."

async def review_report(ctx: Context, review_content: str):
    """Useful for reviewing a report. Use this tool to review a report you have written. 
    Your input should be a review of the report alredy written.
    """
    print("Reviewing report")
    print(f"Report: {review_content}")
    curr_state = await ctx.get("state")
    curr_state["review"] = review_content
    await ctx.set("state", curr_state)
    return "Report reviewed successfully."

async def main():
    print("Hello from agent-workflow!")
    research_agent = FunctionAgent(
        llm=OpenAI(model="gpt-4o-mini"),
        tools=[search_web, record_notes],
        name= "ResearchAgent",
        description="Useful for searching the web for information and recording notes on a specific topic.",
        system_prompt="""You are a helpful assistant that can search the web for information and record notes on a specific topic.
        Make sure you have multiple notes on the topic. You can use the search_web tool to search the web for information.
        You can use the record_notes tool to record notes on a specific topic.
        Once you have recorded multiple notes, you can hand over the report to the WriteReportAgent to write a report.""",
        can_handoff_to=["WriteReportAgent"]
    )

    write_report_agent = FunctionAgent(
        llm=OpenAI(model="gpt-4o-mini"),
        tools=[write_report],
        name= "WriteReportAgent",
        description="Useful for writing a report based on the notes you have recorded.",
        system_prompt="""You are a helpful assistant that can write a report based on the notes you have recorded.
        Make sure the report is formatted in markdown. 
        You can use the write_report tool to write a report.
        Once you have written the report, you can hand over the report to the ReviewReportAgent to review the report.
        Make  sure you review the report atleast 3 times. The report should be within 400 to 500 words.""",
        can_handoff_to=["ReviewReportAgent"]
    )

    review_report_agent = FunctionAgent(
        llm=OpenAI(model="gpt-4o-mini"),
        tools=[review_report],
        name= "ReviewReportAgent",
        description="Useful for reviewing a report and providing feedback.",
        system_prompt="""You are a helpful assistant that can review a report and provide feedback.
        Your review should either approve the report or provide feedback to the WriteReportAgent to improve the report.
        If you have feedback that require changes, you should hand over the control to the WriteReportAgent to make the changes.
        Always make sure the WriteReportAgent has considered the feedback before writing the report again.
        Make sure you have reviewed the report atlease 3 times.""",
        can_handoff_to=["WriteReportAgent"]
    )


    agent = AgentWorkflow(
        agents=[research_agent, write_report_agent, review_report_agent],
        verbose=True,
        root_agent=research_agent.name,
        initial_state={
            "notes": {},
            "report": "No report written yet",
            "review": "No review done yet"
        }
    )

    response = await agent.run(user_msg= "Write me a report on the history of the internet. "
        "Briefly describe the history of the internet, including the development of the internet, the development of the web, "
        "and the development of the internet in the 21st century.")
    print("Response: " + str(response))


if __name__ == "__main__":
    asyncio.run(main())
