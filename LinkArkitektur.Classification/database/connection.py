import requests
import pandas as pd
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)

#response = requests.get("https://link.speckle.dk/api/streams/grB5WJmsE/objects", verify=False)
#df = pd.read_json("https://link.speckle.dk/api/streams/grB5WJmsE/objects", verify=False)

import json

url = "https://link.speckle.dk/api/streams/grB5WJmsE/objects"
JSONContent = requests.get(url, verify=False).json()
content = json.dumps(JSONContent, indent = 4, sort_keys=True)
df = pd.read_json(content)

values = {"Wall_ID": [], "Assembly_Code": [],
          "Area": [],
          "Structural": [],
          "Volume": [],
          "Base Constraint": [],
          }

for i, _ in enumerate(df['resources']):
    wall_id = (df['resources'][i]['_id'])
    values['Wall_ID'].append(wall_id)
    parameters = (df['resources'][i]['properties']['parameters'])
    values['Area'].append(parameters['Area'])
    values['Structural'].append(parameters['Structural'])
    values['Volume'].append(parameters['Volume'])
    values['Base Constraint'].append(parameters['Base Constraint'])
    type_parameters = (df['resources'][i]['properties']['typeParameters'])
    values['Assembly_Code'].append(type_parameters['Assembly Code'])


somethinjesperneeds = pd.DataFrame(values)

print(somethinjesperneeds)
