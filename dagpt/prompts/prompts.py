INSTRUCTIONS_PROMPT = """
You are handling a pandas dataframe in Python, referred to as `df`.
Utilize the tools provided below to respond to the question presented to you:
{tools}

Follow this structure:

Question: the input query you need to address
Thought: always consider your next steps
Action: the action to take, which should be one of [{tool_names}]
Action Input: the input required for the action
Observation: the outcome of the action
... (this Thought/Action/Action Input/Observation cycle can repeat multiple times)
Thought: I now understand the final answer. 

Final Answer: the conclusive answer to the initial input question.

Here is the output of `print(df.head())`:
{df_head}

Let's start!
Question: {input}
{agent_scratchpad}
"""

OPENAI_INSTRUCTION_PROMPT = """
You are handling a pandas dataframe in Python, referred to as `df`.
The result of `print(df.head())` is shown below:
{df_head}
"""
