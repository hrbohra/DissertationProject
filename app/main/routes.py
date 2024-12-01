import json
from datetime import datetime, timezone
from flask import render_template, flash, redirect, url_for, jsonify, request, g, current_app, send_from_directory
from flask_login import current_user, login_required
from flask_babel import _, get_locale
import sqlalchemy as sa
from langdetect import detect, LangDetectException
from app import db
from app.main.forms import EditProfileForm, EmptyForm, PostForm, SearchForm, MessageForm
from app.models import User, Post, Message, Notification
from app.translate import translate
from app.main import bp
import os
from time import time
from flask import render_template, request, send_file, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import pandas as pd
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
import folium
from folium.plugins import HeatMap
from app.main import bp
from jinja2 import Template
import json
import os
import requests
from flask import render_template, flash, redirect, url_for, jsonify, request, g, current_app, send_from_directory
from flask_login import current_user, login_required
from werkzeug.utils import secure_filename
from time import time
from app.main import bp

# Initialize NLP models
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True, framework="pt")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Helper function to define UPLOAD_FOLDER within the application context
def get_upload_folder():
    return os.path.join(current_app.root_path, 'uploads')

# # NLP page for file uploads and emotion processing
# @bp.route('/nlp', methods=['GET', 'POST'])
# def nlp_page():
#     upload_folder = get_upload_folder()
#     os.makedirs(upload_folder, exist_ok=True)
    
#     if request.method == 'POST':
#         action = request.form.get('action')
#         if action == 'nlp':
#             # Handle file upload
#             uploaded_files = request.files.getlist('files[]')
#             file_paths = []
#             for file in uploaded_files:
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(upload_folder, filename)
#                 file.save(filepath)
#                 file_paths.append(filepath)
#             # Process the files
#             processed_file = process_files(file_paths, upload_folder)
#             return send_file(processed_file, as_attachment=True)
#     return render_template('nlp.html')

# # Function to process uploaded files, remove PII, and perform emotion analysis
# def process_files(file_paths, upload_folder):
#     all_data = []
#     for path in file_paths:
#         df = pd.read_csv(path)
#         processed_rows = []
#         for _, row in df.iterrows():
#             text = f"{row['title']} {row['description']} {row['emoji']}"
#             # Remove PII
#             analysis_results = analyzer.analyze(text=text, language='en')
#             anonymized_text = anonymizer.anonymize(text=text, analyzer_results=analysis_results).text
#             # Emotion Detection
#             emotions = emotion_classifier(anonymized_text)[0]
#             # Calculate positive and negative scores
#             positive_score = sum([score['score'] for score in emotions if score['label'] in ['joy', 'love', 'surprise']])
#             negative_score = sum([score['score'] for score in emotions if score['label'] in ['anger', 'sadness', 'fear', 'disgust']])
#             # Append processed data
#             processed_rows.append({
#                 'longitude': row['longitude'],
#                 'latitude': row['latitude'],
#                 'title': row['title'],
#                 'description': row['description'],
#                 'emoji': row['emoji'],
#                 'positive_score': positive_score,
#                 'negative_score': negative_score
#             })
#         all_data.extend(processed_rows)
#     # Save to new CSV
#     processed_df = pd.DataFrame(all_data)
#     processed_file_path = os.path.join(upload_folder, 'processed_data.csv')
#     processed_df.to_csv(processed_file_path, index=False)
#     return processed_file_path


import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from transformers import pipeline
import os

# Initialize NLP models and Presidio engines
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base', return_all_scores=True, framework="pt")
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Reverse Geocoding Function using OpenCage API
def get_location_name(lat, lng):
    try:
        # OpenCage Data API 
        api_key = 'add_key_here'
        url = f"https://api.opencagedata.com/geocode/v1/json?q={lat}+{lng}&key={api_key}"
        response = requests.get(url)
        data = response.json()
        if data and len(data['results']) > 0:
            return data['results'][0]['formatted']  # Return the best-matched formatted address
        else:
            return f"Unknown ({lat}, {lng})"
    except Exception as e:
        print(f"Error in reverse geocoding: {e}")
        return f"Unknown ({lat}, {lng})"

def process_files(file_paths, upload_folder):
    all_data = []
    for path in file_paths:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip().str.lower()  # Normalize column names
        
        required_columns = {'title', 'description', 'emoji', 'latitude', 'longitude', 'timestamp'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")
        
        processed_rows = []
        for _, row in df.iterrows():
            text = f"{row.get('title', '')} {row.get('description', '')} {row.get('emoji', '')}"
            # Remove PII and filter bad words
            anonymized_text = anonymize_and_filter_bad_words(text)

            # Emotion Detection
            emotions = emotion_classifier(anonymized_text)[0]
            # Calculate positive and negative scores
            positive_score = sum([score['score'] for score in emotions if score['label'] in ['joy', 'love', 'surprise']])
            negative_score = sum([score['score'] for score in emotions if score['label'] in ['anger', 'sadness', 'fear', 'disgust']])
            # Process timestamps
            timestamp = row.get('timestamp', '')
            try:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%fZ")  # If ISO format
            except ValueError:
                # If a different format is expected, adjust the parsing
                timestamp = None
            
            timestamp = pd.to_datetime(row['timestamp'], errors='coerce')  # Handles any incorrect formats

            # Calculate emotion intensity
            emotion_intensity = positive_score + negative_score
            emotion_score = positive_score - negative_score  # New emotion score for time series


            # Get location names from coordinates
            location1 = get_location_name(row['latitude'], row['longitude'])  # Reverse geocode location 1
            location2 = get_location_name(row['latitude'], row['longitude'] + 0.1)  # Reverse geocode location 2

            # Append processed data for all visualizations
            processed_rows.append({
                'longitude': row['longitude'],
                'latitude': row['latitude'],
                'title': row['title'],
                'description': row['description'],
                'emoji': row['emoji'],
                'positive_score': positive_score,
                'emotion_score': emotion_score,  # New Emotion Score for Time Series 
                'negative_score': negative_score,
                'emotion_intensity': emotion_intensity,  # For Bubble Chart, Heat Map
                'location1': location1,  # For Network Graph
                'location2': location2,  # For Network Graph
                'anonymized_description': anonymized_text,  # For Word Cloud
                'timestamp': timestamp
                })
        
        all_data.extend(processed_rows)

    # Save to new CSV after processing
    processed_df = pd.DataFrame(all_data)
    processed_file_path = os.path.join(upload_folder, 'processed_data_enhanced.csv')
    processed_df.to_csv(processed_file_path, index=False)
    return processed_file_path

# Removed profanity check and switched to simple word list filtering
bad_words = ["badword1", "badword2", "badword3"]  # Can update with easily available real list during use. For now, I aim to avoid using real bad words to prevent reader distress of any kind.
# Function to check if text contains bad words (profanity)
def contains_bad_words(text):
    words = text.lower().split()
    return any(word in bad_words for word in words)

# Function to anonymize text using Presidio and filter bad words
def anonymize_and_filter_bad_words(text):
    # Anonymize PII using Presidio
    analyzer_results = analyzer.analyze(text=text, language='en')
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=analyzer_results).text
    
    # Filter bad words using the word list
    if contains_bad_words(anonymized_text):
        return '[REDACTED]'
    return anonymized_text

# Function to check if text contains NSFW content based on emotions
def is_nsfw_text(text):
    result = emotion_classifier(text)[0]
    nsfw_labels = ['murder', 'revenge', 'gun', 'knife', 'steal']  # Using aggressive emotions as a proxy for NSFW content
    for emotion in result:
        if emotion['label'] in nsfw_labels and emotion['score'] > 0.5:
            return True
    return False

# NLP page for file uploads and emotion processing + CSV conversion
@bp.route('/nlp', methods=['GET', 'POST'])
def nlp_page():
    upload_folder = get_upload_folder()
    os.makedirs(upload_folder, exist_ok=True)
    converted_files = []

    if request.method == 'POST':
        action = request.form.get('action')

        # Handle NLP file upload and processing
        if action == 'nlp':
            uploaded_files = request.files.getlist('files[]')
            file_paths = []
            for file in uploaded_files:
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_folder, filename)
                file.save(filepath)
                file_paths.append(filepath)
            try:
                processed_file = process_files(file_paths, upload_folder)
                return send_file(processed_file, as_attachment=True)
            except ValueError as e:
                flash(str(e), 'danger')
                return redirect(url_for('main.nlp_page'))

        # Handle CSV conversion for visualization tools
        if action == 'convert':
            conversion_file = request.files['conversion_file']
            if conversion_file and allowed_file(conversion_file.filename):
                filename = secure_filename(conversion_file.filename)
                filepath = os.path.join(upload_folder, filename)
                conversion_file.save(filepath)

                # Process the file and generate different CSV formats
                converted_files = convert_csv_for_visualizations(filepath, upload_folder)

    return render_template('nlp.html', converted_files=converted_files)

# Function to convert the "All Entries" CSV into various formats for visualizations with anonymization, NSFW filtering, and bad words filtering
def convert_csv_for_visualizations(filepath, upload_folder):
    df = pd.read_csv(filepath)

    # Ensure proper column names
    df.columns = df.columns.str.strip().str.lower()

    converted_files = []

    # 1. Choropleth Map Format
    choropleth_df = df[['latitude', 'longitude', 'title']].copy()
    choropleth_file = os.path.join(upload_folder, 'choropleth_data.csv')
    choropleth_df.to_csv(choropleth_file, index=False)
    converted_files.append('choropleth_data.csv')

    # 2. Bubble Chart Format (with NSFW and bad words filtering)
    bubble_df = df[['longitude', 'latitude', 'positive_score', 'negative_score', 'description']].copy()

    # Use NSFW detection and bad words filtering on the description text
    bubble_df['is_nsfw'] = bubble_df['description'].apply(is_nsfw_text)
    bubble_df['contains_bad_words'] = bubble_df['description'].apply(contains_bad_words)
    bubble_df_filtered = bubble_df[(bubble_df['is_nsfw'] == False) & (bubble_df['contains_bad_words'] == False)].copy()

    bubble_df_filtered['emotion_intensity'] = bubble_df_filtered['positive_score'] + bubble_df_filtered['negative_score']
    bubble_file = os.path.join(upload_folder, 'bubble_chart_data.csv')
    bubble_df_filtered[['longitude', 'latitude', 'emotion_intensity']].to_csv(bubble_file, index=False)
    converted_files.append('bubble_chart_data.csv')

    # 3. Word Cloud Format (with PII anonymization and bad words filtering)
    word_cloud_df = df[['description']].copy()

    # Use Presidio to anonymize PII and filter bad words
    word_cloud_df['anonymized_description'] = word_cloud_df['description'].apply(lambda x: anonymize_and_filter_bad_words(x))
    word_cloud_file = os.path.join(upload_folder, 'word_cloud_data.csv')
    word_cloud_df[['anonymized_description']].to_csv(word_cloud_file, index=False)
    converted_files.append('word_cloud_data.csv')

    # 4. Network Graph Format (Dummy data for location1 and location2)
    network_graph_df = df[['latitude', 'longitude']].copy()
    network_graph_df['location1'] = network_graph_df['latitude']
    network_graph_df['location2'] = network_graph_df['longitude'] + 0.1  # Dummy data for the second location
    network_graph_file = os.path.join(upload_folder, 'network_graph_data.csv')
    network_graph_df[['location1', 'location2']].to_csv(network_graph_file, index=False)
    converted_files.append('network_graph_data.csv')

    # 5. Heat Map Format
    heat_map_df = df[['latitude', 'longitude', 'positive_score', 'negative_score']].copy()
    heat_map_df['emotion_intensity'] = heat_map_df['positive_score'] - heat_map_df['negative_score']
    heat_map_file = os.path.join(upload_folder, 'heat_map_data.csv')
    heat_map_df[['latitude', 'longitude', 'emotion_intensity']].to_csv(heat_map_file, index=False)
    converted_files.append('heat_map_data.csv')

    return converted_files

# @bp.route('/heat_map', methods=['POST'])
# def heat_map():
#     file = request.files['processed_file']
#     df = pd.read_csv(file)

#     # Prepare data for heat map
#     heat_data = [[row['latitude'], row['longitude'], row['positive_score'] - row['negative_score']] for index, row in df.iterrows()]

#     # Create map
#     m = folium.Map(location=[0, 0], zoom_start=2)
#     HeatMap(heat_data).add_to(m)

#     # Add custom legend with a color scale
#     legend_html = '''
#     {% raw %}
#     <div style="position: fixed; 
#     bottom: 50px; left: 50px; width: 200px; height: 150px; 
#     background-color: white; z-index:9999; font-size:14px;
#     border:2px solid grey; padding: 10px;">
#         <strong>Emotion Intensity</strong><br>
#         <i style="background: linear-gradient(to right, blue, green, red); width: 100%; height: 20px; display: inline-block;"></i><br>
#         <div style="display: flex; justify-content: space-between; text-align: center; width: 100%;">
#             <div style="width: 33%;"><span>High<br>Negative</span></div>
#             <div style="width: 33%;"><span>Neutral</span></div>
#             <div style="width: 33%;"><span>High<br>Positive</span></div>
#         </div>
#     </div>
#     {% endraw %}
#     '''
    
#     # Add legend to map
#     m.get_root().html.add_child(folium.Element(Template(legend_html).render()))

#     # Save map to HTML in memory
#     map_html = m._repr_html_()
#     return render_template('heat_map.html', map_html=map_html)



# # Path to uploads folder (update this according to your project structure)
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

@bp.before_app_request
def before_request():
    if current_user.is_authenticated:
        current_user.last_seen = datetime.now(timezone.utc)
        db.session.commit()
        g.search_form = SearchForm()
    g.locale = str(get_locale())


@bp.route('/', methods=['GET', 'POST'])
@bp.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    form = PostForm()
    if form.validate_on_submit():
        try:
            language = detect(form.post.data)
        except LangDetectException:
            language = ''
        post = Post(body=form.post.data, author=current_user, language=language)
        db.session.add(post)
        db.session.commit()
        flash(_('Your post is now live!'))
        return redirect(url_for('main.index'))
    page = request.args.get('page', 1, type=int)
    posts = db.paginate(current_user.following_posts(), page=page,
                        per_page=current_app.config['POSTS_PER_PAGE'],
                        error_out=False)
    next_url = url_for('main.index', page=posts.next_num) if posts.has_next else None
    prev_url = url_for('main.index', page=posts.prev_num) if posts.has_prev else None
    return render_template('index.html', title=_('Home'), form=form,
                           posts=posts.items, next_url=next_url,
                           prev_url=prev_url)

# Sightengine API details (replace with your credentials)
SIGHTENGINE_API_USER = 'User_Number'
SIGHTENGINE_API_SECRET = 'Secret'

# Function to check if an image is NSFW using Sightengine API
def is_image_nsfw(image_path):
    print(f"Checking if image is NSFW: {image_path}")  # Console logging

    api_url = 'https://api.sightengine.com/1.0/check.json'
    with open(image_path, 'rb') as image_file:
        response = requests.post(api_url, data={
            'models': 'nudity-2.1',
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }, files={'media': image_file})

    result = response.json()
    print(f"NSFW API response: {result}")  # Console logging

    # Check different nudity-related scores
    nudity_raw = result.get('nudity', {}).get('raw', 0)
    sexual_activity = result.get('nudity', {}).get('sexual_activity', 0)
    sexual_display = result.get('nudity', {}).get('sexual_display', 0)
    erotica = result.get('nudity', {}).get('erotica', 0)
    very_suggestive = result.get('nudity', {}).get('very_suggestive', 0)

    print(f"Nudity raw score: {nudity_raw}")
    print(f"Sexual activity score: {sexual_activity}")
    print(f"Sexual display score: {sexual_display}")
    print(f"Erotica score: {erotica}")
    print(f"Very suggestive score: {very_suggestive}")

    # Consider NSFW if any score is higher than a threshold (e.g., 0.5)
    if (nudity_raw >= 0.5 or
        sexual_activity >= 0.5 or
        sexual_display >= 0.5 or
        erotica >= 0.5 or
        very_suggestive >= 0.5):
        return True

    return False

# Function to check face attributes using Sightengine API
def check_face_attributes(image_path):
    print(f"Checking face attributes: {image_path}")  # Console logging

    api_url = 'https://api.sightengine.com/1.0/check.json'
    with open(image_path, 'rb') as image_file:
        response = requests.post(api_url, data={
            'models': 'face-attributes',
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }, files={'media': image_file})

    result = response.json()
    print(f"Face attributes API response: {result}")  # Console logging

    # Iterate over faces detected in the image
    for face in result.get('faces', []):
        minor_score = face.get('attributes', {}).get('minor', 0)
        if minor_score > 0.5:  # Threshold for detecting minors
            print(f"Detected a minor with a score of {minor_score}.")
            return {'minor_detected': True, 'minor_score': minor_score}
    
    return {'minor_detected': False, 'faces': result.get('faces', [])}  # Return the face attribute results

# Route to handle image uploads with NSFW detection and face attribute check
@bp.route('/upload_marker_image', methods=['POST'])
def upload_marker_image():
    UPLOADS_FOLDER = os.path.join(current_app.root_path, 'uploads')

    # Ensure uploads folder exists
    if not os.path.exists(UPLOADS_FOLDER):
        os.makedirs(UPLOADS_FOLDER)

    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400  # Return JSON response in case of error
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400  # Return JSON response in case of error

    if file:
        filename = f'marker-image-{int(time())}.{file.filename.rsplit(".", 1)[1].lower()}'
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        file.save(file_path)

        # Log image upload
        print(f"Image uploaded: {file_path}")

        # Check if the image is NSFW
        if is_image_nsfw(file_path):
            os.remove(file_path)  # Delete the image if flagged as NSFW
            print(f"NSFW content detected, file removed: {file_path}")  # Console logging
            return jsonify({'error': 'NSFW content detected and removed.'}), 400

        # Check face attributes in the image for minors
        face_attributes = check_face_attributes(file_path)

        if face_attributes.get('minor_detected'):
            os.remove(file_path)  # Delete the image if a minor is detected
            print(f"Minor detected with score {face_attributes['minor_score']}, file removed: {file_path}")
            return jsonify({'error': 'Minor detected, file removed.'}), 400

        # Generate URL for the uploaded image
        image_url = url_for('main.uploaded_file', filename=filename, _external=True)
        
        # Return JSON response with the image URL and face attributes
        return jsonify({'imageUrl': image_url, 'faceAttributes': face_attributes['faces']}), 200
    
    return jsonify({'error': 'Invalid file type'}), 400  # Return JSON response in case of error

# Route to serve uploaded images
@bp.route('/uploads/<filename>')
def uploaded_file(filename):
    UPLOADS_FOLDER = os.path.join(current_app.root_path, 'uploads')
    return send_from_directory(UPLOADS_FOLDER, filename)

# Helper function to validate file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




@bp.route('/map')
@login_required
def map():
    user_map_data = current_user.map_data if current_user.map_data else '{}'
    return render_template('map.html', map_data=user_map_data)



@bp.route('/save_map', methods=['POST'])
@login_required
def save_map():
    data = request.get_json()
    if 'map_data' in data:
        # Store map data for the current user
        current_user.map_data = data['map_data']
        db.session.commit()
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'No map data provided'}), 400



@bp.route('/explore')
@login_required
def explore():
    page = request.args.get('page', 1, type=int)
    query = sa.select(Post).order_by(Post.timestamp.desc())
    posts = db.paginate(query, page=page,
                        per_page=current_app.config['POSTS_PER_PAGE'],
                        error_out=False)
    next_url = url_for('main.explore', page=posts.next_num) if posts.has_next else None
    prev_url = url_for('main.explore', page=posts.prev_num) if posts.has_prev else None
    return render_template('index.html', title=_('Explore'),
                           posts=posts.items, next_url=next_url,
                           prev_url=prev_url)


@bp.route('/user/<username>')
@login_required
def user(username):
    user = db.first_or_404(sa.select(User).where(User.username == sa.bindparam('username')).params(username=username))
    page = request.args.get('page', 1, type=int)
    query = user.posts.select().order_by(Post.timestamp.desc())
    posts = db.paginate(query, page=page,
                        per_page=current_app.config['POSTS_PER_PAGE'],
                        error_out=False)
    next_url = url_for('main.user', username=user.username, page=posts.next_num) if posts.has_next else None
    prev_url = url_for('main.user', username=user.username, page=posts.prev_num) if posts.has_prev else None
    form = EmptyForm()
    return render_template('user.html', user=user, posts=posts.items,
                           next_url=next_url, prev_url=prev_url, form=form)


@bp.route('/user/<username>/popup')
@login_required
def user_popup(username):
    user = db.first_or_404(sa.select(User).where(User.username == sa.bindparam('username')).params(username=username))
    form = EmptyForm()
    return render_template('user_popup.html', user=user, form=form)


@bp.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    form = EditProfileForm(current_user.username)
    if form.validate_on_submit():
        current_user.username = form.username.data
        current_user.about_me = form.about_me.data
        db.session.commit()
        flash(_('Your changes have been saved.'))
        return redirect(url_for('main.edit_profile'))
    elif request.method == 'GET':
        form.username.data = current_user.username
        form.about_me.data = current_user.about_me
    return render_template('edit_profile.html', title=_('Edit Profile'), form=form)


@bp.route('/follow/<username>', methods=['POST'])
@login_required
def follow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = db.session.scalar(sa.select(User).where(User.username == sa.bindparam('username')).params(username=username))
        if user is None:
            flash(_('User %(username)s not found.', username=username))
            return redirect(url_for('main.index'))
        if user == current_user:
            flash(_('You cannot follow yourself!'))
            return redirect(url_for('main.user', username=username))
        current_user.follow(user)
        db.session.commit()
        flash(_('You are following %(username)s!', username=username))
        return redirect(url_for('main.user', username=username))
    else:
        return redirect(url_for('main.index'))


@bp.route('/unfollow/<username>', methods=['POST'])
@login_required
def unfollow(username):
    form = EmptyForm()
    if form.validate_on_submit():
        user = db.session.scalar(sa.select(User).where(User.username == sa.bindparam('username')).params(username=username))
        if user is None:
            flash(_('User %(username)s not found.', username=username))
            return redirect(url_for('main.index'))
        if user == current_user:
            flash(_('You cannot unfollow yourself!'))
            return redirect(url_for('main.user', username=username))
        current_user.unfollow(user)
        db.session.commit()
        flash(_('You are not following %(username)s.', username=username))
        return redirect(url_for('main.user', username=username))
    else:
        return redirect(url_for('main.index'))


@bp.route('/translate', methods=['POST'])
@login_required
def translate_text():
    data = request.get_json()
    if 'text' in data:
        translated_text = translate(data['text'])
        return jsonify({'text': translated_text})
    else:
        return jsonify({'error': 'No text provided for translation'}), 400

@bp.route('/api/submit', methods=['POST'])
def track_submit():
    print(f"Submit API called with data: {request.json}")
    return "This is a test route", 404


@bp.route('/search')
@login_required
def search():
    if not g.search_form.validate():
        return redirect(url_for('main.explore'))
    
    page = request.args.get('page', 1, type=int)
    search_term = f"%{g.search_form.q.data}%"
    
    # Safely search for posts using parameterized query
    query = sa.select(Post).where(Post.body.ilike(sa.bindparam('search_term'))).order_by(Post.timestamp.desc())
    
    posts = db.paginate(query.params(search_term=search_term), page=page,
                        per_page=current_app.config['POSTS_PER_PAGE'], error_out=False)
    
    next_url = url_for('main.search', q=g.search_form.q.data, page=page + 1) if posts.has_next else None
    prev_url = url_for('main.search', q=g.search_form.q.data, page=page - 1) if page > 1 else None
    
    return render_template('search.html', title=_('Search'), posts=posts.items, next_url=next_url, prev_url=prev_url)


@bp.route('/send_message/<recipient>', methods=['GET', 'POST'])
@login_required
def send_message(recipient):
    user = db.first_or_404(sa.select(User).where(User.username == recipient))
    form = MessageForm()
    if form.validate_on_submit():
        msg = Message(author=current_user, recipient=user, body=form.message.data)
        db.session.add(msg)
        user.add_notification('unread_message_count', user.unread_message_count())
        db.session.commit()
        flash(_('Your message has been sent.'))
        return redirect(url_for('main.user', username=recipient))
    return render_template('send_message.html', title=_('Send Message'), form=form, recipient=recipient)


@bp.route('/messages')
@login_required
def messages():
    current_user.last_message_read_time = datetime.now(timezone.utc)
    current_user.add_notification('unread_message_count', 0)
    db.session.commit()
    page = request.args.get('page', 1, type=int)
    query = current_user.messages_received.select().order_by(Message.timestamp.desc())
    messages = db.paginate(query, page=page, per_page=current_app.config['POSTS_PER_PAGE'], error_out=False)
    next_url = url_for('main.messages', page=messages.next_num) if messages.has_next else None
    prev_url = url_for('main.messages', page=messages.prev_num) if messages.has_prev else None
    return render_template('messages.html', messages=messages.items, next_url=next_url, prev_url=prev_url)


@bp.route('/export_posts')
@login_required
def export_posts():
    if current_user.get_task_in_progress('export_posts'):
        flash(_('An export task is currently in progress'))
    else:
        current_user.launch_task('export_posts', _('Exporting posts...'))
        db.session.commit()
    return redirect(url_for('main.user', username=current_user.username))


@bp.route('/notifications')
@login_required
def notifications():
    since = request.args.get('since', 0.0, type=float)
    query = current_user.notifications.select().where(Notification.timestamp > since).order_by(Notification.timestamp.asc())
    notifications = db.session.scalars(query)
    return [{'name': n.name, 'data': n.get_data(), 'timestamp': n.timestamp} for n in notifications]




from app.models import Entry  # Assuming Entry model holds geolocation, text, and emojis

from flask import Response
import csv
import json

@bp.route('/download_entries')
@login_required
def download_entries():
    # Check if the current user has map data stored
    if not current_user.map_data:
        return jsonify({'status': 'error', 'message': 'No map data found'}), 404

    # Parse the JSON map data
    try:
        map_data = json.loads(current_user.map_data)
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid map data format'}), 400

    # Prepare the CSV content with title, description, emoji, and timestamp fields
    def generate():
        # CSV header with timestamp
        data = ['Latitude,Longitude,Title,Description,Emoji,Timestamp\n']
        
        for marker in map_data.get('markers', []):
            lat = str(marker.get('lat', '')).strip()
            lng = str(marker.get('lng', '')).strip()
            title = str(marker.get('title', '')).strip()  # Add title from marker
            description = str(marker.get('description', '')).strip()  # Updated to 'description'
            emoji = str(marker.get('emoji', '')).strip()
            timestamp = str(marker.get('timestamp', '')).strip()  # Add timestamp

            # Construct the CSV row and append it to the data list
            data.append(f'{lat},{lng},{title},{description},{emoji},{timestamp}\n')
        
        # Yield each row from the data list
        for row in data:
            yield row

    # Send the CSV as a downloadable response
    return Response(generate(), mimetype='text/csv', headers={'Content-Disposition': 'attachment;filename=entries.csv'})



import os
from werkzeug.utils import secure_filename
import pandas as pd

# Utility functions for processing CSV files for each visualization
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['csv']

def process_choropleth_data(filepath):
    data = pd.read_csv(filepath)
    geo_data = data[['location', 'latitude', 'longitude']]
    emotion_data = data[['emotion_score']]
    return geo_data, emotion_data

def process_time_series_data(filepath):
    data = pd.read_csv(filepath)
    time_series_data = data[['timestamp', 'emotion_score']]
    return time_series_data

def process_network_graph_data(filepath):
    data = pd.read_csv(filepath)
    network_data = data[['location1', 'location2', 'emotion_score']]
    return network_data

def process_bubble_chart_data(filepath):
    data = pd.read_csv(filepath)
    bubble_data = data[['emotion_intensity', 'emotion_frequency']]
    return bubble_data

def process_word_cloud_data(filepath):
    data = pd.read_csv(filepath)
    words = data['text_description']
    return words



import os
from flask import render_template, request, current_app
from flask_login import login_required
from werkzeug.utils import secure_filename
import pandas as pd
from app.main import bp

# Ensure the allowed file types are CSV
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

# Helper function to create the upload folder if it doesn't exist
def ensure_upload_folder_exists():
    upload_folder = os.path.join(current_app.root_path, 'uploads')
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return upload_folder

# Function to process CSV data for each visualization
def process_csv_data(filepath):
    data = pd.read_csv(filepath)
    return data

@bp.route('/heat_map', methods=['POST'])
def heat_map():
    file = request.files['processed_file']
    df = pd.read_csv(file)

    # Prepare data for heat map
    heat_data = [[row['latitude'], row['longitude'], row['positive_score'] - row['negative_score']] for index, row in df.iterrows()]

    # Create map
    m = folium.Map(location=[0, 0], zoom_start=2)
    HeatMap(heat_data).add_to(m)

    # Add custom legend with a color scale
    legend_html = '''
    {% raw %}
    <div style="position: fixed; 
    bottom: 50px; left: 50px; width: 200px; height: 150px; 
    background-color: white; z-index:9999; font-size:14px;
    border:2px solid grey; padding: 10px;">
        <strong>Emotion Intensity</strong><br>
        <i style="background: linear-gradient(to right, blue, green, red); width: 100%; height: 20px; display: inline-block;"></i><br>
        <div style="display: flex; justify-content: space-between; text-align: center; width: 100%;">
            <div style="width: 33%;"><span>High<br>Negative</span></div>
            <div style="width: 33%;"><span>Neutral</span></div>
            <div style="width: 33%;"><span>High<br>Positive</span></div>
        </div>
    </div>
    {% endraw %}
    '''
    
    # Add legend to map
    m.get_root().html.add_child(folium.Element(Template(legend_html).render()))

    # Save map to HTML in memory
    map_html = m._repr_html_()
    return render_template('heat_map.html', map_html=map_html)

# New routes for additional visualizations

@bp.route('/choropleth', methods=['GET', 'POST'])
@login_required
def choropleth_map():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            upload_folder = ensure_upload_folder_exists()
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Process CSV and convert DataFrame to a list of dictionaries
            data = pd.read_csv(filepath)
            data_dict = data.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
            
            # Pass the data_dict to the template instead of the DataFrame
            return render_template('choropleth_map.html', data=data_dict)
    return render_template('choropleth_map.html')


@bp.route('/time_series', methods=['GET', 'POST'])
@login_required
def time_series():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            upload_folder = ensure_upload_folder_exists()
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Process CSV and convert DataFrame to list of dictionaries
            data = pd.read_csv(filepath)
            data_dict = data.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
            
            return render_template('time_series.html', data=data_dict)
    return render_template('time_series.html')

@bp.route('/network_graph', methods=['GET', 'POST'])
@login_required
def network_graph():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            upload_folder = ensure_upload_folder_exists()
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Process CSV and convert DataFrame to list of dictionaries
            data = pd.read_csv(filepath)
            data_dict = data.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
            
            return render_template('network_graph.html', data=data_dict)
    return render_template('network_graph.html')

@bp.route('/bubble_chart', methods=['GET', 'POST'])
@login_required
def bubble_chart():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            upload_folder = ensure_upload_folder_exists()
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Process CSV and ensure the relevant fields are present
            data = pd.read_csv(filepath)
            
            # Ensure the necessary fields exist and are valid
            if 'emotion_intensity' not in data.columns or 'longitude' not in data.columns or 'latitude' not in data.columns:
                flash("CSV file is missing necessary fields for bubble chart", 'danger')
                return render_template('bubble_chart.html')

            # Filter out rows with missing or invalid values
            data = data.dropna(subset=['longitude', 'latitude', 'emotion_intensity'])

            # Convert DataFrame to list of dictionaries for D3.js
            data_dict = data[['longitude', 'latitude', 'emotion_intensity']].to_dict(orient='records')
            
            # Pass the data to the template for rendering
            return render_template('bubble_chart.html', data=data_dict)
    return render_template('bubble_chart.html')

@bp.route('/word_cloud', methods=['GET', 'POST'])
@login_required
def word_cloud():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            upload_folder = ensure_upload_folder_exists()
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_folder, filename)
            file.save(filepath)
            
            # Process CSV and convert DataFrame to list of dictionaries
            data = pd.read_csv(filepath)
            data_dict = data.to_dict(orient='records')  # Convert DataFrame to list of dictionaries
            
            return render_template('word_cloud.html', data=data_dict)
    return render_template('word_cloud.html')
