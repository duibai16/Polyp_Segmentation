import os
import sys
import tempfile

def get_resource_path(relative_path):
    
    try:
      
        base_path = sys._MEIPASS
    except AttributeError:
       
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
