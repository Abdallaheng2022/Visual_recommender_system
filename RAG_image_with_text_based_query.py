#Vector Databases are used to store the data
#Main title of the app
import os
from dotenv import load_dotenv
import base64
from langchain_community.vectorstores import FAISS, AzureSearch
from langchain.docstore.document import Document
import openai
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from langchain_openai import OpenAIEmbeddings
import streamlit as st
from PIL import Image
import math
import subprocess
import re
import base64
import json
from urllib.parse import urljoin
env_path = os.path.join('.env')
load_dotenv(env_path)
azure_api=os.getenv("AZURE_API")
azure_end=os.getenv("AZURE_END")
azure_ai_search=os.getenv("AI_KEY")
azure_ai_search_end= os.getenv("AZURE_AI_SEARCH_END")
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MAX_MIN_RETRIES = 10

def encode(image_path):
  """
   it takes the image and return image in url form instead of uploading to cloud
  """
  with open(image_path,'rb') as image_file:
          return base64.b64encode(image_file.read()).decode("utf-8")
MAX_MIN_RETRIES = 10
def get_gpt_image_description(image_url,model):
  """
    it get the image description from images
  """
  prompt="Describe the fashion item in the image. Specify the type, gender of clothing, color and other details that you see."
  #handling failures with retires
  for i in range(MAX_MIN_RETRIES):
     try:

          response = client.chat.completions.create(
          model = model,
          messages=[{"role": "user",
                      "content":[{"type":"text","text":prompt},
                      {"type":"image_url","image_url":{"url":image_url}}
                            ]
                      }],
          max_tokens=300,
          )
          return response.choices[0].message.content

     except Exception as e:
         print(f"Error: {e}")
         print(f"Retrying GPT Call {i}")
         time.sleep(21)
def get_image_description(image_path,model="gpt-4o"):
  """
  Open AI model takes the image and model and get the description for the image
  """
  base64_image = encode(image_path)
  return get_gpt_image_description(image_url=f"data:image/jpeg:base64,{base64_image}",model=model)
nrows=3000
styles3000=pd.read_csv("/home/abdo/Downloads/LLMs_apps_gihub/visual_recommender_system/fashion_product_small/styles{nrows}_with_decription.csv".format(nrows=nrows))
intent_system_msg = """
               Your task is to rephrase the user query to reflect explicitly what the user is looking for.
               If the user is looking for some clothing that matches another item, and provided the other item features, then you can use your knowledge to expand the list of features that can match whay the user looking for.
               For example, if user query ""I want a polo shirt goes with black shoes"",
               you can rephrase it as
               ""I'm looking for a polo shirt that matches any black shoes. I would like the shirt to be casual and comfortable, suitable  for everyday wear. The color should be a solid color like black, blue or gray.""
               INPUT:
               JSON object:
                  {{
                   'user_query':'the user query'
                  }}
               Example input:
                   {{
                     'user_query':'I want a polo shirt that goes with black shoes'

                   }}
               OUTPUT:
               JSON object:
                  {{results: 'the rephrased user query'
                  }}
               Example output:

                   {{
                     'results':'I'm looking for a polo shirt that matches any black shoes. I would like the shirt to be casual and comfortable, suitable  for everyday wear. The color should be a solid color like black, blue or gray.'
                   }}


             """
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
def create_vector_database(styles,embeddings_model):
  """
  it takes the styles and inserts them into vector database after embedding them.
  """
  styles['description'] =  styles['description'].fillna('')
  docs = []
  for i,row in styles.iterrows():
       docs.append(Document(page_content=row['description'],metadata=dict(index=row['id'])))
  vec_db = FAISS.from_documents(docs,embeddings_model)
  return vec_db


def get_rephrased_query(user_query,system_msg,model="gpt-4o"):
    """
    It takes the user query rephrased to outline the features of the product.
    """
    intent_prompt="""{{
                       'user_query':{user_query}
                   }}"""
    intent_prompt = intent_prompt.format(user_query=user_query)
    response = get_gpt_llm_response(intent_prompt,intent_system_msg,model=model)
    results=parse_llm_response(response)
    return results



def get_vector_recommendations(user_intent,vec_db,k=5):
        #reshape the prompt of user query
        relevant_items_with_scores = vec_db.similarity_search_with_score(user_intent,k=k)
        recomended_items = []
        for item,score in relevant_items_with_scores:
            recomended_items.append({'id':item.metadata['index'],'justification':f"Match score:{score}"})
        return recomended_items
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
def _to_records(df, max_items=300):  # keep context small
    recs = df.to_dict(orient="records")
    return recs[:max_items]

def get_recommendations(user_query, top_n, system_msg, stock_items_json, model="gpt-4o"):
    system_msg = system_msg.format(n=top_n)
    payload = {
        "user_query": str(user_query),
        "stock_items":stock_items_json ,  # tune this
        "n": int(top_n)
    }
    prompt = json.dumps(payload, ensure_ascii=False)
    response = get_gpt_llm_response(prompt, system_msg, model=model)
    results = parse_llm_response(response)
    return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]

def get_random_images(styles,path,size):
    img_ls=[]
    random_images = styles['id'][styles['gender']=="Men"].sample(n=size)
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
                       print(f"Error when displaying the image {result['id']}:{e}")
                       continue
def rephrase_query(user_query,image_path,model="gpt-4o"):
    """
     It takes the user query and image path 
    """
    image_desc = get_image_description(image_path,model=model)
    final_user_query = f"{user_query}\n image:{image_desc}"
    return final_user_query

       


def initialize_session_state():
    """تهيئة جميع متغيرات الحالة المطلوبة"""
    if 'image_likes' not in st.session_state:
        st.session_state.image_likes = {}
    if "image_paths" not in st.session_state:
        st.session_state.image_paths = []   
    if "show_products" not in st.session_state:
        st.session_state.show_products = False
    if "products_initialized" not in st.session_state:
        st.session_state.products_initialized = False
def create_image_grid_with_likes(image_paths_list, cols=4):
    """
    دالة واحدة بسيطة لعرض الصور مع نظام لايكات يحتفظ بحالته
    """
    # تجهيز البيانات لكل صورة إذا لم تكن موجودة
    for img_path in image_paths_list:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        if img_id not in st.session_state.image_likes:
            st.session_state.image_likes[img_id] = {
                "path": img_path,
                "count": 0,
                "liked": False
            }
    
    # عرض الصور مع أزرار اللايك
    rows = math.ceil(len(image_paths_list) / cols)
    
    for row in range(rows):
        columns = st.columns(cols)
        for col in range(cols):
            idx = row * cols + col
            if idx >= len(image_paths_list):
                continue
            
            img_path = image_paths_list[idx]
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            
            with columns[col]:
                # عرض الصورة
                try:
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)
                except Exception:
                    with open(img_path, "rb") as fh:
                        st.image(fh.read(), use_container_width=True)
                
                # الحصول على بيانات اللايك الحالية
                like_data = st.session_state.image_likes[img_id]
                is_liked = like_data["liked"]
                count = like_data["count"]
                
                # عرض العداد
                st.caption(f"❤️ {count} likes")
                
                # زر اللايك
                if is_liked:
                    button_label = "❤️ Liked"
                    button_help = "Click to unlike"
                else:
                    button_label = "🤍 Like"
                    button_help = "Click to like"
                
                if st.button(
                    button_label,
                    key=f"btn_{img_id}_{idx}",
                    help=button_help,
                    use_container_width=True
                ):
                    # تبديل حالة اللايك
                    if is_liked:
                        st.session_state.image_likes[img_id]["liked"] = False
                        st.session_state.image_likes[img_id]["count"] = max(0, count - 1)
                    else:
                        st.session_state.image_likes[img_id]["liked"] = True
                        st.session_state.image_likes[img_id]["count"] = count + 1
                    st.rerun()




def main():
    # Use the tunnel URL if available
    initialize_session_state() 

    st.title("Generative image and text-based query VS AI Recommender 🤖💡")
    # Add a subtitle or description
    st.subheader("Fashion Products Search by Prompt")
    # Here you will face the problem that we didn't the intention of the user
    # we will use another LaLM to rephrase the prompt of users.
    # Add a longer description with markdown
    st.markdown("""
    This app helps you search for fashion product suitable for your current clothes:
    * Write your product description.
    * Upload your Image(Optional).
    * Get top-rank products Are highly related to your product.
    """)
    
    top_n = 20
    nrows = 3000
    stock_items=pd.read_csv("/home/abdo/Downloads/LLMs_apps_gihub/visual_recommender_system/fashion_product_small/styles{nrows}_with_decription.csv".format(nrows=nrows))
    image_path ="/home/abdo/Downloads/LLMs_apps_gihub/visual_recommender_system/fashion_product_small/images/"
      # أزرار التحكم
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Get products"):
            st.session_state.show_products = True
            if not st.session_state.image_paths:
                st.session_state.image_paths = get_random_images(stock_items, image_path, 8)
    
    with col2:
        if st.button("Refresh products"):
            st.session_state.image_paths = get_random_images(stock_items, image_path, 8)
    
    with col3:
        if st.button("Hide products"):
            st.session_state.show_products = False
    
    # عرض الصور مع اللايكات
    if st.session_state.show_products and st.session_state.image_paths:
        create_image_grid_with_likes(st.session_state.image_paths, cols=4)
        
        # عرض ملخص اللايكات
        liked_items_paths = [image_path+"/"+str(k)+".jpg" for k, v in st.session_state.image_likes.items() if v.get("liked", False)]
        if liked_items_paths:
            #st.info(f"You liked {len(liked_items)} items: {', '.join(liked_items[:5][])}{'...' if len(liked_items) > 5 else ''}")
            columns = st.columns(4)
            for idx,item in enumerate(liked_items_paths):  
                with columns[idx%4]:
                   st.write("Image you liked:")
                   st.image(item)
                   st.write(item.split('/')[-1].split('.')[0])
   
    product_description=st.text_area("Please, write the text here")
    search=st.button("search")
    if search:
            # access the ids of image to get the description or use LLMs to generate description
            rephrased_query=rephrase_query(product_description,liked_items_paths[0],model="gpt-4o") 
            user_intent=get_rephrased_query(rephrased_query,intent_system_msg,model="gpt-4o")
            st.write("User intent:",user_intent)
            #intialize vector databases
            vec_db=create_vector_database(stock_items,OpenAIEmbeddings())
            #Get the embeddings based on the similarity between the embeddings of user query and embeddings of desc
            k = 100
            shortlisted_stock=get_vector_recommendations(user_intent,vec_db,k)
            # filterout the only the important information
            shortlisted_stock_json =json.dumps([{'id':item['id'],'description':list(stock_items[stock_items['id']==item['id']]['description']),'score':item['justification']} for item in shortlisted_stock])
            top_n = 5
            results=get_recommendations(user_intent,top_n,system_msg,shortlisted_stock_json,model="gpt-4o")
            visualize_recommendations(results,image_path,stock_items[['id','description','productDisplayName']],n_cols=3)
if __name__ == "__main__":
    main()