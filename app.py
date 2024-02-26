import os
import json
import sys
import pandas as pd
import openai
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from langchain.schema.output_parser import StrOutputParser
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from pydantic import BaseModel, Field
import pytesseract # For image processing
# import pyodbc  # For SQL connections
# import docx  # For Word document processing
import PyPDF2 # For PDF processing
from typing import List, Type
from typing import Optional
from typing import Dict, Any
import csv
import re
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Set the OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a model
model = ChatOpenAI(temperature=0)

app = FastAPI()
useFASTAPI = True

########################################################################################################
### Static Classes (Not used currently) ###
########################################################################################################

# Classes used for Excel/CSV files
class InternsData(BaseModel):
    """Information about an intern's interview status."""
    name: str = Field(description="intern's name")
    status: str = Field(description="indicates whether the intern passed or not")    

class InternsList(BaseModel):
    """Information to extract."""
    people: List[InternsData] = Field(description="List of interns who passed the interview")

# Classes used for Image files
class InvoiceData(BaseModel):
    """Information about invoice for employees."""
    name: str = Field(description="employee's name")
    date: str = Field(description="indicates when the employee incurredt the expense")
    amount: str = Field(description="amount of the expense")

class InvoiceList(BaseModel):
    """Information to extract."""
    employee: List[InvoiceData] = Field(description="List of employess who incurred expenses with the details")

# Class used for compliance files
class MaxWeeklyHours(BaseModel):
    """Information to extract."""
    hours: str = Field(description="maximum weekly hours limit")

########################################################################################################
### Dynamic Class ###
########################################################################################################

# Class used for generic extraction
class DynamicData(BaseModel):
    """Information to pull from the prompt and set it in each attrib. Depending on the prompt, the number of attribs can change."""
    attrib1: str = Field(description="attribute 1")
    attrib2: str = Field(description="attribute 2")
    attrib3: str = Field(description="attribute 3")
    attrib4: str = Field(description="attribute 4")
    attrib5: str = Field(description="attribute 5")

class DynamicClass(BaseModel):
    """Information to extract."""
    attributes: List[DynamicData] = Field(description="List of attributes")

########################################################################################################
### Attributes(Not used currently) ###
########################################################################################################

def extractEntities(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[ {"role": "system", "content": f"List the entities mentioned in the following prompt:\n{text}"}],        
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    entities = [entity.strip() for entity in response.choices[0].message.content.split(",")]
    cleaned_list = [string.strip("- \n") for string in entities[0].splitlines()]    
    return cleaned_list

# If we want to get the entities from the ask, we can use the above function. However, dynamic class cannot be created based on the entities
#entities = extractEntities(ask)
#print(entities)

########################################################################################################
### Extraction model (Not used currently) ###
########################################################################################################

# Use predefined functions for static classes
# functions = [
#     convert_pydantic_to_openai_function(InternsList),
#     convert_pydantic_to_openai_function(InvoiceList),
#     convert_pydantic_to_openai_function(MaxWeeklyHours)
# ]    

########################################################################################################
### Postprocessing based on the returned data for static functions (Not used currently) ###
########################################################################################################

# Function that creates interns CSV file
def create_interns_csv(data, output_file):
    # Get the dynamic field name (e.g., "interns", "people")
    field_name = next(iter(data.keys()))

    # Extract the list of interns using the dynamic field name
    interns = data[field_name]

    # Filter interns based on the status field
    passed_interns = [item["name"] for item in interns if item["status"] in ["Yes", "Passed", "Accepted", "Pass"]]

    # Write the names to a new CSV file
    with open(output_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows([[name] for name in passed_interns])  # CSV rows

def create_invoice_csv(data, output_file):
    # Get the dynamic field name (e.g., "interns", "people")
    field_name = next(iter(data.keys()))

    # Extract the list of employees using the dynamic field name
    employees = data[field_name]

    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ['name', 'date', 'amount']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()  # Write the header row

        for employee in employees:
            writer.writerow(employee)

def ProcessPredefinedOutput(output):
    os.makedirs("output", exist_ok=True)
    if isinstance(output, dict) and 'people' in output:
        print(output)
        create_interns_csv(output, "output\\accepted_interns.csv")
    if isinstance(output, dict) and 'employee' in output:
        print(output)
        create_invoice_csv(output, "output\\invoice.csv")
    if isinstance(output, dict) and 'hours' in output:
        print("Extracted maximum weekly hours limit:", output)
    elif isinstance(output, str):
        print("Received text output:", output)
    else:
        print("Unhandled output:", output)        

########################################################################################################
### Postprocessing based on the returned data for dynamic function when useFASTAPI is not enabled ###
########################################################################################################

def create_dynamic_csv(data, output_file):
    field_name = next(iter(data.keys()))

    # Extract the list of attributes using the dynamic field name
    attributes = data[field_name]

    with open(output_file, mode='w', newline='') as csv_file:
        fieldnames = ['attrib1', 'attrib2', 'attrib3', 'attrib4', 'attrib5']
        writer = csv.writer(csv_file)

        for attrib in attributes:
            # Create a row with the values of the attributes, in the order of fieldnames
            row = [attrib.get(field) for field in fieldnames if attrib.get(field) not in [None, '']]
            
            writer.writerow(row)

def ProcessDynamicOutput(output):
    print("Output:", output)    
    if config_output_file_name and len(config_output_file_name) > 0:        
        print("Creating: ", config_output_file_name[0])
        os.makedirs("output", exist_ok=True)
        file_to_create = "output\\" + config_output_file_name[0]
        create_dynamic_csv(output, file_to_create)
    else:
        print("No output file specified in the JSON")        

########################################################################################################
### Main processing function using OpenAI operations using Langchain ###
########################################################################################################

def ProcessPrompt(ask, files):
    # Read the contents of the files
    data_list = []

    for file in files:
        extension = file.split(".")[-1]
        if extension in ["csv", "xlsx"]:
            # Structured Excel or CSV data
            if extension == "xlsx":
                data = pd.read_excel(file)
            else:
                data = pd.read_csv(file)
            data_list.append(data.to_string())
        elif extension in ["jpg", "jpeg", "png"]:
            # Unstructured data (images)
            try:
                text = pytesseract.image_to_string(file)
                data_list.append(text)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        elif extension == "pdf":
            # Unstructured data (PDF)
            try:
                with open(file, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text = "\n".join(page.extract_text() for page in pdf_reader.pages)
                data_list.append(text)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        else:
            print(f"Unsupported file type: {file}")
            if not useFASTAPI:
                sys.exit(1)
            else:
                return {"error": f"Unsupported file type: {file}"}

    combined_data = "\n".join(data_list)
    #print(combined_data)

    # Use generic function for dynamic class
    functions = [
        convert_pydantic_to_openai_function(DynamicClass)
    ]    

    # Create the extraction model
    extraction_model = model.bind(functions=functions)   

    # Create the prompt from ChatPromptTemplate and the prompt string from the json above
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"{ask}"),
        ("human", "{input}")
    ])

    # Create the extraction chain
    extraction_chain = prompt | extraction_model | JsonOutputFunctionsParser()

    # Now we are ready to process the data and get the output
    output = extraction_chain.invoke({"input": combined_data})

    if not useFASTAPI:
        ProcessDynamicOutput(output)
    else:
        return output

########################################################################################################
### Config Invocation when useFASTAPI is not enabled ###
########################################################################################################

if not useFASTAPI:
    with open("config.json", "r") as f:
        config = json.load(f)

    # Extract the ask and file fields from the JSON
    config_ask = config["ask"]
    config_files = [os.path.join("files", file) for file in config["file"]]
    config_output_file_name = config.get("output_file")

    ProcessPrompt(config_ask, config_files)

########################################################################################################
### Routes Invocation ###
########################################################################################################

if useFASTAPI:
    @app.post("/process")
    async def process_data(prompt: str = Body(...), files: List[UploadFile] = File(...)):
        file_names = []

        # Save uploaded files locally
        for file in files:
            file_name = file.filename
            with open(file_name, "wb") as f:
                f.write(await file.read())
            file_names.append(file_name)

        print("Input:", prompt, file_names)

        # results = ProcessPrompt(prompt, file_names)
        # Combining the files does not give the correct results at times, so process each file separately and then combine the results
        results = []
        for local_file in file_names:
            result = ProcessPrompt(prompt, [local_file])
            results.append(result)
        
        print("Output:", results)

        # Delete uploaded files after processing
        for file_name in file_names:
            try:
                os.remove(file_name)
            except Exception as e:
                print(f"Error deleting {file_name}: {e}")

        return {"results": results}

    @app.get("/hello")
    async def hello_world():
        return {"message": "Hello from Extraction FAST API!"}

    origins = ["*"]  # Replace with specific origins if necessary

    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"]
    )

    if __name__ == "__main__":
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

