from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant
here are some relevant review : {reviews}
here is the question to answer : {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = prompt | model

while True : 
    print("="*60)
    user_question = input("Type 'exit' for quit : ")
    print("="*60)
    if user_question.lower() == "exit" :
        print("Have a good day")
        break

    reviews = retriever.invoke(user_question)
    result = chain.invoke({"reviews": reviews, "question": user_question})
    print(result)