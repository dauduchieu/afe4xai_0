from google import genai
from dotenv import load_dotenv
import os
from pydantic import BaseModel

load_dotenv()
GOOGLE_GENAI_API_KEY = os.getenv('GOOGLE_GENAI_API_KEY')

def llm_structed_output(prompt='', api_key=GOOGLE_GENAI_API_KEY, res_type='application/json', res_schema=None):
    if res_schema is None: 
        return None
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config={
            'response_mime_type': res_type,
            'response_schema': res_schema
        }
    )
    return response.parsed

if __name__ == '__main__':
    class Recipe(BaseModel):
        recipe_name: str
        ingredients: list[str]
    r = llm_structed_output(
        prompt='List a few popular cookie recipes. Be sure to include the amounts of ingredients.',
        res_type='application/json',
        res_schema=list[Recipe]
    )
    print(r)

