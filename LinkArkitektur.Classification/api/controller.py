from flask import Flask
from flask_restful import Resource, Api, reqparse
from src import main

app = Flask(__name__)
api = Api(app)


class Users(Resource):
    @app.route('/api/users')
    def get():
        data = main.Application()
        data = data.get_data()
        return {'data': data.to_dict()}, 200  # return data and 200 OK

    @app.route('/api/users')
    def post():
        parser = reqparse.RequestParser()  # initialize
        parser.add_argument('api_url', required=True)  # add args
        args = parser.parse_args()  # parse arguments to dictionary
        return args['api_url', 'access_token']


api.add_resource(Users)  # '/users' is our entry point for Users
app.run()  # Initialize the api server
