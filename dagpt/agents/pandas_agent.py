from typing import Any, Literal, Optional, Sequence, cast

from dotenv import load_dotenv
from langchain.agents import (
    AgentType,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain.agents.agent import AgentExecutor, RunnableAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel, LanguageModelLike
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool

from dagpt.agents.base import BaseAgent
from dagpt.prompts.prompts import INSTRUCTIONS_PROMPT, OPENAI_INSTRUCTION_PROMPT
from dagpt.tools.tools_ori import PythonAstREPLTool
from dagpt.utils.utils import BaseLogger

# Load environment variables
load_dotenv()

# Initialize the logger
logger = BaseLogger()


class DataAnalysisAgent(BaseAgent):
    """
    A class for creating a data analysis agent \
    using different types of language models.

    Attributes:
        llm: The language model to be used.
        df: The dataframe to be analyzed.
        agent_type: The type of agent to be created.
        AgentType.OPENAI_FUNCTIONS or AgentType.ZERO_SHOT_REACT_DESCRIPTION.
        callback_manager: The callback manager for handling events.
        number_of_head_rows: The number of rows to display \
            from the head of the dataframe.
        instruction_prompt: Custom instructions for the agent.
        engine: The engine to be used for handling dataframes ('pandas' or 'modin').
        verbose: Whether to enable verbose logging.
        return_intermediate_steps: Whether to return intermediate steps.
        max_iterations: Maximum number of iterations for the agent.
        max_execution_time: Maximum execution time for the agent.
        early_stopping_method: Method for early stopping.
    """

    def __init__(
        self,
        llm: LanguageModelLike,
        df: Any,
        agent_type: AgentType = AgentType.OPENAI_FUNCTIONS,
        callback_manager: Optional[BaseCallbackManager] = None,
        number_of_head_rows: int = 5,
        instruction_prompt: str = "",
        engine: Literal["pandas", "modin"] = "pandas",
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        max_iterations: Optional[int] = 15,
        max_execution_time: Optional[float] = None,
        early_stopping_method: str = "force",
        extra_tools: Sequence[BaseTool] = (),
    ):
        self.llm = llm
        self.df = df
        self.agent_type = agent_type
        self.callback_manager = callback_manager
        self.number_of_head_rows = number_of_head_rows
        self.instruction_prompt = instruction_prompt
        self.engine = engine
        self.verbose = verbose
        self.return_intermediate_steps = return_intermediate_steps
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time
        self.early_stopping_method = early_stopping_method
        self.extra_tools = extra_tools

    def get_prompt(self) -> str:
        """Get the appropriate prompt based on the agent type."""
        if self.agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
            logger.info("INFO: Loaded zero-shot react instruction prompt!")
            return self._get_react_prompt()
        elif self.agent_type == AgentType.OPENAI_FUNCTIONS:
            logger.info("INFO: Loaded OpenAI functions instruction prompt!")
            return self._get_openai_functions_prompt()
        else:
            raise ValueError(f"Invalid agent type: {self.agent_type}")

    def create_agent(self, **kwargs: Any) -> AgentExecutor:
        """Create an agent based on the specified configurations."""
        if self.engine == "pandas":
            import pandas as pd
        elif self.engine == "modin":
            import modin.pandas as pd
        else:
            raise ValueError(f"Invalid engine: {self.engine}")

        # Validate and prepare the dataframe(s)
        for _df in self.df if isinstance(self.df, list) else [self.df]:
            if not isinstance(_df, pd.DataFrame):
                raise ValueError("df must be a pandas.DataFrame")

        df_locals = {}
        if isinstance(self.df, list):
            for i, dataframe in enumerate(self.df):
                df_locals[f"df{i+1}"] = dataframe
        else:
            df_locals["df"] = self.df

        tools = [PythonAstREPLTool(locals=df_locals)] + list(self.extra_tools)

        if self.agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
            logger.info("INFO: Created zero-shot react agent!")
            react_agent = create_react_agent(
                llm=self.llm, tools=tools, prompt=self.get_prompt()
            )

            agent = RunnableAgent(
                runnable=react_agent,
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            )

        elif self.agent_type == AgentType.OPENAI_FUNCTIONS:
            logger.info("INFO: Created OpenAI functions agent!")

            runable = create_openai_functions_agent(
                cast(BaseLanguageModel, self.llm),
                tools=tools,
                prompt=self.get_prompt(),
            )

            agent = RunnableAgent(
                runnable=runable,
                input_keys_arg=["input"],
                return_keys_arg=["output"],
            )
        else:
            raise ValueError(f"Invalid agent type: {self.agent_type}")

        return AgentExecutor(
            agent=agent,
            tools=tools,
            callback_manager=self.callback_manager,
            verbose=self.verbose,
            return_intermediate_steps=self.return_intermediate_steps,
            max_iterations=self.max_iterations,
            max_execution_time=self.max_execution_time,
            early_stopping_method=self.early_stopping_method,
            **kwargs,
        )

    def _get_react_prompt(self) -> str:
        """Get the react prompt for the agent."""
        prompt = PromptTemplate.from_template(INSTRUCTIONS_PROMPT).partial()
        df_head = str(self.df.head(self.number_of_head_rows).to_markdown())
        return prompt.partial(df_head=df_head)

    def _get_openai_functions_prompt(self) -> str:
        """Get the OpenAI functions prompt for the agent."""
        openai_prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=SystemMessage(content=OPENAI_INSTRUCTION_PROMPT)
        )
        return openai_prompt


# if __name__ == "__main__":
#     from langchain_openai import ChatOpenAI

#     # Initialize the language model
#     llm = ChatOpenAI(
#         temperature=0,
#         model="gpt-3.5-turbo",
#     )

#     # Load the dataframe
#     df = pd.read_csv("../../data/sample_data.csv")

#     # Create and configure the data analysis agent
#     daagent = DataAnalysisAgent(df=df, llm=llm, verbose=True)

#     # Create the agent executor
#     agent = daagent.create_agent()

#     # Invoke the agent with a query
#     res = agent.invoke("Plot the correlation matrix of the dataframe.")
#     print(res["output"])
