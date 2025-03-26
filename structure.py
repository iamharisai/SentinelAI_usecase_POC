import os, dotenv
import pandas as pd
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Set your OpenAI API key here
# os.environ["OPENAI_API_KEY"] = "sk-......"  # Replace with your actual API key
dotenv.load_dotenv()

# Initialize our LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)


class MetricState(TypedDict):
    metric_name: str
    metric_usage: pd.DataFrame
    is_concern: Optional[bool]
    last_1day_average: int
    last_7days_average: int
    last_30days_average: int
    last_90days_average: int
    last_180days_average: int
    last_365days_average: int
    threshold: Optional[int]


import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Initialize LLM
model = ChatOpenAI( model="gpt-4o",temperature=0)


# Define nodes
def process_metric(state: MetricState):
    metric_name = state["metric_name"]
    metric_usage = state["metric_usage"]
    print(metric_name)
    return {}

def classify_concern(state: MetricState):
    print(f"is_concern is processing")
    metric_name = state["metric_name"]
    metric_usage = state["metric_usage"]  

    prompt = f"""
        As an AI assistant, analyze the metric usage data for {metric_name} and determine if there is a concern.
        The usage data for {metric_name} is {metric_usage}
        The metric usage is available as pandas dataframe which has time and value columns.
        answer with YES or NO if there is a concern. Only return the answer and explanation
        Answer :
        Explanation :
    """
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    response_text = response.content.lower()
    is_concern = "yes" in response_text.split('\n')[0] and "no" not in response_text.split('\n')[0]
    print("Response: \n", response_text)
    print("Is this a concern: ", is_concern)
    return {is_concern: is_concern}


# Create the graph
metric_graph = StateGraph(MetricState)
# Add nodes
metric_graph.add_node("process_metric", process_metric)
metric_graph.add_node("classify_concern", classify_concern)

metric_graph.add_edge(START, "process_metric")
metric_graph.add_edge("process_metric", "classify_concern")
metric_graph.add_edge("classify_concern", END)

compiled_metric_graph = metric_graph.compile()

metric_usage = pd.read_csv("disk_usage.csv")
metric_state = MetricState(
    metric_name="disk_usage",
    metric_usage=metric_usage,
    is_concern=None,
    last_1day_average=0,
    last_7days_average=0,
    last_30days_average=0,
    last_90days_average=0,
    last_180days_average=0,
    last_365days_average=0,
    threshold=None
)


processed_result = compiled_metric_graph.invoke(metric_state)