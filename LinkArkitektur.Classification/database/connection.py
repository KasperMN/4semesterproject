import requests
import pandas as pd
import json

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)

url = "https://link.speckle.dk/api/streams/grB5WJmsE/objects"
JSONContent = requests.get(url, verify=False).json()
content = json.dumps(JSONContent, indent = 4, sort_keys=True)
df = pd.read_json(content)

index = {'_id': []}

values = {'Assembly_Code': [],
          'Area': [],
          'Structural': [],
          'Volume': [],
          'Base Constraint': [],
          }

for i, _ in enumerate(df['resources']):
    wall_id = (df['resources'][i]['_id'])  # Specific wall id's
    parameters = (df['resources'][i]['properties']['parameters'])  # Regular parameters for the wall
    type_parameters = (df['resources'][i]['properties']['typeParameters'])  # Parameters connected with the type of wall

    ''' Collecting values to dictionaries {}'''
    index['_id'].append(wall_id)  # Wall Identification Number
    values['Assembly_Code'].append(type_parameters['Assembly Code'])  # The target label
    values['Area'].append(parameters['Area'])  # Area in Cubic Meters
    values['Structural'].append(parameters['Structural'])  # Boolean value for structural use
    values['Volume'].append(parameters['Volume'])  # The volume of the Wall
    values['Base Constraint'].append(parameters['Base Constraint'])

somethinjesperneeds = pd.DataFrame(data=values, index=index['_id'])  # Values as columns, index as indexes

print(somethinjesperneeds)
