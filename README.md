# Welcome to Investigation App!
This application provides users with the ability to create accounts, log in, explore a 2D map, and place markers on the map, each with accompanying images and comments, which are first scanned for NSFW and deleted if NSFW score if high. It also includes social features such as following or unfollowing friends and sending direct messages to other users. 

Advanced feature:
\
Data from map can be downloaded. This downloaded file can be fed to anonymiser to then give to Visualisers that can explain the data visually. The new anonymised file can be shared with others without risking personal data loss. Thus giving control over privacy to each user.

Attached video demonstrates functioning app.

The Translate and NSFW APIs are functional and require the user's credentials for operation. To configure these, update the API keys for all apis(opencage, sightengine, etc.) and credentials within the `routes.py` file located in the `main` folder of the application directory and the translate.py file. 
\


Once the updates are complete, open a new terminal in Visual Studio Code or navigate to the project directory using the command prompt to proceed

1. Create and run virtual environtment

 python -m venv venv

 venv\Scripts\activate


2. Install requirements file

 pip install -r requirements.txt



3. Run app

 flask run