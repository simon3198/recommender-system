import json

import pandas as pd
from flask import Flask, jsonify, make_response, request, send_file

app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False

# category = None

@app.route('/api/files/<filename>', methods=['GET'])
def get_file_by_name(filename):
    file_path = f'full_matrix/{filename}_views_fullmatrix.csv'  # Adjust the path to your file directory
    # /Users/simon3198/Desktop/졸작/flask-server/full_matrix/게임_views_fullmatrix.csv

    data = pd.read_csv(file_path,index_col=0)
    columns = list(data.columns)
    
    result = json.dumps(columns, ensure_ascii=False)
    res = make_response(result)
    
    return res

@app.route('/api/columns/<filename>', methods=['GET'])
def get_column_name(filename):
    
    column_name = request.args.get('column')
    # print(column_name)
    result=[]
    for method in ['views','likes','comments']:
        file_path = f'full_matrix/{filename}_{method}_fullmatrix.csv'  # Adjust the path to your file directory
        # /Users/simon3198/Desktop/졸작/flask-server/full_matrix/게임_views_fullmatrix.csv
        data = pd.read_csv(file_path,index_col=0)
        
        data = data.sort_values(by=column_name,ascending=False).head(40)
        
        chan_dict = dict(data[column_name])
        id_dict = dict(data['channedid'])

        result.append([chan_dict,id_dict])
    
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')
        
    
    data = pd.read_csv(f'full_matrix/{filename}_views_fullmatrix.csv',index_col=0)
    chanid = data['channedid']
    
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    data = data.T
    data=(data-data.min())/(data.max()-data.min())
    data = data.T
    data['channedid']=chanid
    
    data = data.sort_values(by=column_name,ascending=False).head(40)
    
    chan_dict = dict(data[column_name])
    id_dict = dict(data['channedid'])

    result.append([chan_dict,id_dict])
    
    result = json.dumps(result, ensure_ascii=False)
    res = make_response(result)
    
    return res

if __name__ == '__main__':
    app.run(debug=True)
