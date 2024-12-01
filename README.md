# Welcome to Investigation App!
Application Overview:

This application enables users to create accounts, log in, navigate a 2D map interface, and add markers with associated images and comments. Submitted content is automatically scanned for NSFW material, and any flagged content with a high NSFW score is removed. Social features include the ability to follow or unfollow friends and exchange direct messages with other users.

Advanced Features:

Users can download map data, which can then be processed through an anonymizer to remove personally identifiable information. The anonymized dataset can be visualized using external visualization tools, enabling users to share insights without compromising privacy. This ensures each user retains control over their personal information while facilitating safe data sharing.

Application Demonstration:

The attached video provides a demonstration of the application in action.

API Integration:

The application supports functional APIs for translation and NSFW detection, which require user credentials for operation. To configure these, ensure the API keys for services such as OpenCage and SightEngine are updated in the routes.py file within the main application directory, as well as in the translate.py file.
\


Once the updates are complete, open a new terminal in Visual Studio Code or navigate to the project directory using the command prompt to proceed

1. Create and run virtual environtment

 python -m venv venv

 venv\Scripts\activate


2. Install requirements file

 pip install -r requirements.txt



3. Run app

 flask run