#######################

This script demonstrate how deployed ML model for LIVECell instance segmentation works as backend service

Before using, install requirements by running ``pip install -r requirements.txt``


#######################

Backend server starts with 

``py manage.py migrate``

``py manage.py runserver``

After that, feel free to pass to script any image from dataset https://sartorius-research.github.io/LIVECell/ as below:

``py .\main.py C:\Users\MEDIA\Desktop\LiveCELL\A172_Phase_C7_1_00d00h00m_2.tif
``