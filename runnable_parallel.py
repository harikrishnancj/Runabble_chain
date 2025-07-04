from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model1=ChatHuggingFace(llm=llm)

model2=ChatHuggingFace(llm=llm)


prompt2=PromptTemplate(
    template="summarise in 3 line {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(
    template="make 5 quiz from the {text}",
    input_variables=['text']
)

prompt4=PromptTemplate(
    template="merg the  {note} and {quiz} into singe documents",
    input_variables=['note','quiz']
)

parser=StrOutputParser()


parallel =RunnableParallel({
    "note":prompt2|model1|parser,
    "quiz":prompt3|model2|parser
}
)
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

merg_chain=prompt4|model1|parser

chain=parallel|merg_chain

res=chain.invoke({"text":text})
print(res)

chain.get_graph().print_ascii()