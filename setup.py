from setuptools import find_packages,setup
from typing import List




def get_requirements(file_path:str)-> List[str]:
    "retrurn requirments"
    
    req=[]
    HYPEN_E_DOT = '-e .'
    with open(file_path) as file_obj:
        req = file_obj.readlines()
        
        req = [r.replace("\n","")for r in req]
        
        if HYPEN_E_DOT in req:
            req.remove(HYPEN_E_DOT)
        
    return req


setup(
    name = "ML Project",
    version='0.0.1',
    author = 'prajwal ganvir',
    author_email='prajwalganvir001@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)