from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="chat"
)

model1 = ChatHuggingFace(llm=llm)
model2= ChatHuggingFace(llm=llm)

parser = StrOutputParser()

template = PromptTemplate(
    template="describe about the {topic}",
    input_variables=['topic']
)

template2 = PromptTemplate(
    template="summarize in 3 point {text}",
    input_variables=['text']
)

chain = template | model1 | parser | template2 | model2 | parser

#chain=RunnableSequence(template,model2,parser,template2,model1,parser)
res=chain.invoke({'topic':"Blackhole"})

print(res)
