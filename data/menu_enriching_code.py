#Imports
import json
import os
import sys
one_level_up = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))
sys.path.append(one_level_up)
from components.llm_call.gpt4_llm import call_llm_gpt4

#Function to get the name of products with their ID for the menus
def get_products_dict(json_data):
        products = {}
        for category, items in json_data.items():
            for item_id, item_info in items.items():
                 if isinstance(item_info, list):
                    name, price, details = item_info
                    products[item_id] = name
        return products

#Function for enriching the data with keywords that better describe the products (Helps with keyword search)
def generate_list_keywords(text):

    prompt = f"""
    Your task is to generate a list of 5 keywords using the info for the following product, the keywords must be related to the product to improve the search engine of the restaurant.

    This are some examples of how to give your output, always follow the format and logic of response:

    Example 1:
    
    Info = Product name: Pepsi  Category: Drinks

    Output: soda, drink, cola, coke, pepsi


    Example 2:

    Info = Product name: Cheeseburger , Category: Burgers

    Output: cheese, food, hamburguer, whopper, No allergens


    Info = {text}

    Output:
    """

    response = call_llm_gpt4(prompt)
    #replace the word "output" with a ""
    response = response.replace("output:", "")
    response = response.replace("Output:", "")
    return response



# Function to add "keywords" field to each entry
def add_keywords(data):
    product_dict = get_products_dict(data)
    for category, item_value in data.items():
        for entry_key, item_info in item_value.items():
            try:
                if isinstance(item_info, dict):
                    name = item_info.get("name", "")
                    print(name)
                    price = item_info.get("price", "")
                    nutritional_info = item_info.get("contents", {})

                    for element in nutritional_info:
                        try:
                            element[0] = product_dict[element[0]]
                        except:
                            pass
                    item_info["contents"] = nutritional_info
                    try:
                        item_info["keywords"] = generate_list_keywords(f"Product name: {name} Content_details: {str(nutritional_info)} Category: {category}")
                        print(item_info["keywords"])
                    except:
                        pass   
                else:
                    name, price, details = item_info
                    print(name)
                    nutritional_info = details.get("nutritionalInfo", {})
                    try:
                        item_info.append({"keywords" : generate_list_keywords(f"Product name: {name}  Category: {category}")})
                        print(item_info)
                    except:
                        pass
            except Exception as e:
                print(e)
                pass
                
    return data
# Main code
if __name__ == "__main__":
    # Read the original JSON data from KFC.json
    with open("KFC", "r") as json_file:
        original_data = json.load(json_file)

    # Add "keywords" field
    enriched_data = add_keywords(original_data)

    # Export the enriched data to a new JSON file (KFC_enriched.json)
    with open("KFC_enriched.json", "w") as json_file:
        json.dump(enriched_data, json_file, indent=4)

    print("Enriched data has been saved to KFC_enriched.json")
