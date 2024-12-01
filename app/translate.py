import requests
from flask_babel import _
from langdetect import detect, LangDetectException

# Replace with your actual API key
GOOGLE_API_KEY = 'add_key_here'

def translate(text):
    try:
        # Detect the language of the input text
        source_language = detect(text)
    except LangDetectException:
        return _('Error: Unable to detect the source language.')

    # Define the destination language (always English in this case)
    dest_language = 'en'

    # Make the request to the Google Translate API
    try:
        r = requests.post(
            'https://translation.googleapis.com/language/translate/v2',
            params={
                'key': GOOGLE_API_KEY
            },
            json={
                'q': text,
                'source': source_language,
                'target': dest_language,
                'format': 'text'
            }
        )
        r.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        return _(f'Error: the translation service failed. Details: {e}')

    response = r.json()

    # Print the raw response for debugging
    print("Translation API response:", response)

    if 'data' in response and 'translations' in response['data'] and len(response['data']['translations']) > 0:
        return response['data']['translations'][0]['translatedText']
    else:
        return _('Error: the translation service returned an unexpected response.')



# FOllowing, is the right way to add key to environment variables


# import os
# import requests
# from flask_babel import _
# from langdetect import detect, LangDetectException

# GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

# def translate(text):
#     if not GOOGLE_API_KEY:
#         return _('Error: Google API key not configured.')

#     try:
#         source_language = detect(text)
#     except LangDetectException:
#         return _('Error: Unable to detect the source language.')

#     dest_language = 'en'

#     try:
#         r = requests.post(
#             'https://translation.googleapis.com/language/translate/v2',
#             params={'key': GOOGLE_API_KEY},
#             json={
#                 'q': text,
#                 'source': source_language,
#                 'target': dest_language,
#                 'format': 'text'
#             }
#         )
#         r.raise_for_status()
#         response = r.json()

#         if 'data' in response and 'translations' in response['data'] and len(response['data']['translations']) > 0:
#             return response['data']['translations'][0]['translatedText']
#         else:
#             return _('Error: The translation service returned an unexpected response.')

#     except requests.exceptions.HTTPError as e:
#         if e.response.status_code == 400:
#             return _('Error: Invalid request. Please check your input text.')
#         elif e.response.status_code == 401:
#             return _('Error: Invalid API key.')
#         elif e.response.status_code == 429:
#             return _('Error: Translation quota exceeded.')
#         else:
#             return _(f'Error: The translation service failed. Details: {e}')
#     except requests.exceptions.RequestException as e:
#         return _(f'Error: The translation service failed. Details: {e}')
