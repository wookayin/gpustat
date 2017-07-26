from flask import Flask
import gpustat
import json
import time
import pprint
import sys


app = Flask(__name__)


@app.route('/')
def get_status():
    data = gpustat.print_gpustat(**{'show_cmd': False, 'gpuname_width': 16, 'show_user': False, 'show_pid': False, 'json': True, 'no_color': False, 'ret_json': True})
    data['query_time']= time.time()
    
    return('{0}'.format(json.dumps(data,indent=4)))

if __name__ == "__main__":
    port = 5000
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
  
    app.run(host='0.0.0.0',port=port)
