import requests
import pandas as pd
import json

pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 20)


class ApiConnection:
    def __init__(self):
        self.url = "https://link.speckle.dk/api/streams/grB5WJmsE/objects"
        self.JSONContent = requests.get(self.url, verify=False).json()
        self.content = json.dumps(self.JSONContent, indent=4, sort_keys=True)
        self.df = pd.read_json(self.content)
        self.index = {'_id': []}
        self.values = {'Assembly_Code': [], 'Area': [], 'Structural': [], 'Volume': [], 'Base_Constraint': [], 'Project_id': []}

    def collect_link_data(self):
        for i, _ in enumerate(self.df['resources']):
            wall_id = (self.df['resources'][i]['_id'])  # Specific wall id's
            parameters = (self.df['resources'][i]['properties']['parameters'])  # Regular parameters for the wall
            type_parameters = (self.df['resources'][i]['properties']['typeParameters'])  # Parameters connected with the type of wall

            ''' Collecting values to dictionaries {}'''
            self.index['_id'].append(wall_id)  # Wall Identification Number
            self.values['Assembly_Code'].append(type_parameters['Assembly Code'])  # The target label
            self.values['Area'].append(parameters['Area'])  # Area in Cubic Meters
            self.values['Structural'].append(parameters['Structural'])  # Boolean value for structural use
            self.values['Volume'].append(parameters['Volume'])  # The volume of the Wall
            self.values['Base_Constraint'].append(parameters['Base Constraint'])
            self.values['Project_id'].append(1) # SKEJBY hospital project

        return pd.DataFrame(data=self.values, index=self.index['_id'])  # Values as columns, index as indexes