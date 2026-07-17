from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever


llm = OllamaLLM(model="gemma3:4b")

template = """
You are an expert in answering question about restaurants using the provided context.
Here are some relevant reviews:{reviews}
Here is a question about restaurants: {question}
Please provide the rating title along with the answer.
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

while True:
    question = input("Enter your question about restaurants (or 'q' to quit): ")
    print("\n\n")
    if question.lower() == 'q':
        break

    reviews = retriever.invoke(question)
    result = chain.invoke({ "reviews": reviews,"question": question})
    print("Answer:", result)
    print("\n\n--------------------------------------------\n\n ")






