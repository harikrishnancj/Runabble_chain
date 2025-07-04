from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="write a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template=" summarize {text}",
    input_variables=['text']
)

parser = StrOutputParser()


branch=RunnableBranch(
    (lambda x:len(x.split())>300,prompt2|model|parser),
    RunnablePassthrough()
)

chain=prompt1|model|parser|branch
print(chain.invoke({"topic":"isarel vs palestine"}))

chain.get_graph().print_ascii()