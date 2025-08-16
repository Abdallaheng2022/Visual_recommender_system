
# Main title of the app
#Text-based query recommender system
import os
from dotenv import load_dotenv
import base64
from langchain_community.vectorstores import FAISS, AzureSearch
import openai
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from langchain_community.embeddings import OpenAIEmbeddings
import streamlit as st
from PIL import Image
import math
import subprocess
import re
import base64
from urllib.parse import urljoin
import io
from concurrent.futures import ThreadPoolExecutor

env_path = os.path.join('.env')
load_dotenv(env_path)
azure_api=os.getenv("AZURE_API")
azure_end=os.getenv("AZURE_END")
azure_ai_search=os.getenv("AI_KEY")
azure_ai_search_end= os.getenv("AZURE_AI_SEARCH_END")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


nrows=1000
styles1000=pd.read_csv("/home/abdo/Downloads/LLMs_apps_gihub/visual_recommender_system/fashion_product_small/styles{nrows}_with_decription.csv".format(nrows=nrows))

system_msg = """
              TASK:
                 I'm building a fashion items recommendations engine. I will give you a list of fashion items, with id and description, and a user query. Your task is to find matching items from the list I gave you to user query.

              INPUT:
                  You will be given a json object with list of available stock items, expressed as a list of json objects, each has an id and description, and a user query which decribes the item the user is looking for.

              Example Input:
                              {{[
                          {{
                            "id": 15970,
                            "description": "The fashion item is a men's button-up shirt. It features a blue and white plaid pattern with long sleeves that are rolled up. The collar appears to be a darker blue, adding a contrasting detail to the shirt. The fit is tailored and casual."
                          }},
                          {{
                            "id": 39386,
                            "description": "The fashion item is a pair of men's jeans. They are a dark blue wash with some lighter fading, giving them a slightly worn look. The jeans have a classic straight-leg fit and feature standard details like a button and zip closure, front and back pockets, and belt loops. The stitching is visible and appears to be in a contrasting color, adding detail to the design. The jeans are paired with casual boat shoes in a dark color."
                          }},
                          {{
                            "id": 59263,
                            "description": "The image shows a wristwatch. It features a minimalist design with a round silver-colored case and a white face. The watch has simple hour and minute hands without any hour markers. The band is a thin, silver mesh strap. This watch appears to be a unisex fashion item, suitable for both men and women. It also includes branding details with the words 'Obaku' and 'Titan' on the watch face."
                          }},
                          {{
                            "id": 21379,
                            "description": "The fashion item is a pair of men's sweatpants. They are black with white side stripes. There is a logo on the right thigh area, and the text 'MAN UTD' is printed vertically on the left side. The pants have a drawstring waist for adjustment."
                          }}
                        ],
                        'user_query':"I want some pants that goes with mean sport shoes"

                        }}
            OUTPUT:

              You are requested to return the top {{n}} matching results, with id and score between 0 and 1 expressing the match score of the user query, with 1 being the highest match.
              Always use id as given in the input list of stock items. Do not hallucinate non-existing ids.
              Return the list ordered by score in descending order.
              When making a match, focus on the main fashion item in the product description, regardless of what other items might appear in its image.
              Also, focus on the gender of the matched items and the type of clothing, as well as the color and style of the items.
              Important: Match exactly the type of item requested by the user query. For example, if the user query asks for pants, do not recommend shirts or shoes, even if they are of the same color or style.

            Example Output:

            {{
              "results": [
                {{
                  "id": "the identifier of the item as given in the input list",
                  "description": "the description as given in the input list",
                  "score": 1,
                  "justification": "justify your choice here"
                }},
                {{
                  "id": "the identifier of the item as given in the input list",
                  "description": "the description as given in the input list",
                  "score": 0.8,
                  "justification": "justify your choice here"
                }}
              ]
            }}


            """
MAX_MIN_RETRIES=20
def get_gpt_llm_response(prompt,system_msg,model="gpt-4o"):
  """
   it configure the LLMs to work as recommender systems
  """
  for i in range(MAX_MIN_RETRIES):
     try:

          response = client.chat.completions.create(
          model = model,
          messages=[{"role": "user",
                      "content":prompt}
                    ,{"role": "system",
                      "content":system_msg}

                    ],
          temperature=0.0,
          response_format={"type":"json_object"}
          )

          return response.choices[0].message.content

     except Exception as e:
         print(f"Error: {e}")
         print(f"Retrying GPT Call {i}")
         time.sleep(21)

def parse_llm_response(response):
    return eval(response)['results']

def get_recommendations(user_query,top_n,system_msg,stock_items,model="gpt-4o"):
        #reshape the prompt of user query
        system_msg = system_msg.format(n=top_n)
        prompt = """{{
                       'user_query':{user_query},
                       'stock_items':{stock_items}
                   }}"""
        prompt = prompt.format(user_query=user_query,stock_items=stock_items)
        response = get_gpt_llm_response(prompt,system_msg,model=model)
        results = parse_llm_response(response)
        results= sorted(results, key=lambda x: x['score'], reverse=True)
        return results


def get_random_images(styles,path,size):
    img_ls=[]
    random_images = styles['id'][styles['gender']=="male"].sample(n=size)
    for idx in random_images.index:
        img_ls+=[path+str(random_images[idx])+'.jpg']

    return img_ls

def visualize_recommendations(results,image_path,styles,n_cols=3):
    """
      It will take recommendations results from LLM and image paths and visualize them in columns
    """
    #configure the number of columns in streamlit
    cols=st.columns(n_cols)
    #Organize the shown recommended images based on the user text query or user text and image query
    for idx, result in enumerate(results):
           #alternate between the number of columns
           with cols[idx%n_cols]:
                  try:
                     #visualize each recomended image with its justification and score.
                     st.image(os.path.join(image_path,str(result['id'])+'.jpg'),width=100,caption=styles[styles['id']==result['id']]['productDisplayName'].values[0])
                     st.write(result['justification'])
                     st.write(result['score'])

                  except Exception as e:
                       print(f"Error when displaying the image {results['id']}:{e}")
                       continue

# كاش قراءة CSV (اختياري لكنه يسرّع)
@st.cache_data(show_spinner=False)
def load_styles(nrows=3000, path="1/fashion-dataset/styles{nrows}_with_decription.csv"):
    return pd.read_csv(path.format(nrows=nrows))

# كاش توليد ثَمبنيل من مسار صورة
@st.cache_data(show_spinner=False)
def load_thumb_bytes(img_path: str, thumb_px: int = 256, jpg_quality: int = 70) -> bytes:
    with Image.open(img_path) as im:
        im = im.convert("RGB")            # توحيد الصيغة
        im.thumbnail((thumb_px, thumb_px))# توليد ثَمبنيل سريع
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=jpg_quality, optimize=True)
        return buf.getvalue()

def create_image_grid_fast(image_paths_list, cols: int = 4, thumb_px: int = 256, workers: int = 8):
    """
    Grid سريع:
    - يسبق التحميل (prefetch) بالثريدز
    - يعرض ثَمبنيلز صغيرة مع use_container_width
    """
    if not image_paths_list:
        return

    # تحميل بالتوازي (مع احترام الكاش)
    max_workers = min(workers, max(1, (os.cpu_count() or 8)))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        thumbs = list(ex.map(lambda p: load_thumb_bytes(p, thumb_px), image_paths_list))

    columns = st.columns(cols, gap="small")
    for i, thumb in enumerate(thumbs):
        with columns[i % cols]:
            st.image(thumb, use_container_width=True)

# ── بديل أسرع لدالتك القديمة ──────────────────────────────────────
def create_image_grid(image_paths_list, cols=4):
    # استدعِ الدالة السريعة
    create_image_grid_fast(image_paths_list, cols=cols, thumb_px=256, workers=8)


          
def main():
    # Use the tunnel URL if available
    img_path_ls=get_random_images(styles1000,"/home/abdo/Downloads/LLMs_apps_gihub/visual_recommender_system/fashion_product_small/images/",8)
    st.title("VS AI Recommender 🤖💡")
    # Add a subtitle or description
    st.subheader("Fashion Products Search by Prompt")
    # Here you will face the problem that we didn't the intention of the user
    # we will use another LLM to rephrase the prompt of users.
    # Add a longer description with markdown
    st.markdown("""
    This app helps you search for fashion product suitable for your current clothes:
    * Write your product description.
    * Upload your Image(Optional).
    * Get top-rank products Are highly related to your product.
    """)
    top_n = 10
    nrows=3000
    stock_items=pd.read_csv("/home/abdo/Downloads/LLMs_apps_gihub/visual_recommender_system/fashion_product_small/styles{nrows}_with_decription.csv".format(nrows=nrows))
    image_path = "/home/abdo/Downloads/LLMs_apps_gihub/visual_recommender_system/fashion_product_small/images/"
    if img_path_ls is not []:
        create_image_grid(img_path_ls)
    product_description=st.text_area("Please, write the text here")
    search=st.button("search")
    if search:
            results=get_recommendations(product_description,top_n,system_msg,stock_items[['id','description']],model="gpt-4o")
            visualize_recommendations(results,image_path,stock_items[['id','description','productDisplayName']],n_cols=3)
if __name__ == "__main__":
    main()
