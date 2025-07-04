from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableLambda,RunnablePassthrough
from pydantic import BaseModel


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

def wcount(text):
    return len(text.split())
    

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
    "explanation": prompt2 | model | parser,
    "word_count":RunnableLambda(wcount)
})


chain=prompt1|model|parser|par_chain
res=chain.invoke({"topic":"world Poltics"})

class res_r(BaseModel):
    joke: str
    explanation: str
    word_count: int

n=res_r(**res)

n = res_r(**res)

print(f"Joke: {n.joke}")
print(f"Explanation: {n.explanation}")
print(f"Word count: {n.word_count}")


chain.get_graph().print_ascii()
