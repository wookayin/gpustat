from flask import Flask
import gpustat

app = Flask(__name__)


@app.route('/')
def get_status():
    data = gpustat.print_gpustat(**{'show_cmd': True, 'gpuname_width': 16, 'show_user': True, 'show_pid': True, 'json': True, 'no_color': False, 'ret_json': True} )    
    return('{0}'.format(data))

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5000)
