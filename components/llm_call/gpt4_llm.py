from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()




def call_llm_gpt4(prompt):
    response = client.chat.completions.create(
    model="gpt-4",
    temperature= 0,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{prompt}"},
    ]
    )

    return response.choices[0].message.content
    







