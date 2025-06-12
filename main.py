import os
from dotenv import load_dotenv

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig # type: ignore

load_dotenv()


def main():
    MODEL_NAME = "gemini-2.0-flash"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    external_client = AsyncOpenAI(
    api_key =  os.getenv("GEMINI_API_KEY"),
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model= OpenAIChatCompletionsModel(
     model = MODEL_NAME,
     openai_client = external_client,   
    )
    
    config = RunConfig(
        model = model,
        model_provider= external_client,
        tracing_disabled= True
    )

    assistant  = Agent(
    name = "Assistant",
    instructions = "You are the simple Agent, your job is to resolve queries.",
    model = model
    )

    result = Runner.run_sync(assistant , "tell me about power of AI",run_config = config)
    print(result.final_output)


if __name__ == "__main__":
    main()
