from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

class JokeModel(BaseModel):
    joke: str
    explanation: str

prompt1 = PromptTemplate(
    template="Tell me one joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the joke: {text}",
    input_variables=['text']
)

parser = StrOutputParser()

par_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": prompt2 | model | parser
})


chain = prompt1 | model | parser | par_chain


res_raw = chain.invoke({"topic": "AI"})
res = JokeModel(**res_raw)

print(res)
chain.get_graph().print_ascii()
