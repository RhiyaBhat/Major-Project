from langchain_community.llms import LlamaCpp

model_path = "./models/Llama-3.2-3B-Instruct-Q4_K_L.gguf"

llm = LlamaCpp(
    model_path=model_path,
    n_ctx=4096,
    n_threads=6,
    temperature=0.2,
)

print(llm("Introduce yourself in one friendly sentence."))
