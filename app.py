from flask import Flask, request, render_template, Response
from flask_mail import Mail, Message
from wordcloud import WordCloud
from langdetect import detect
import matplotlib.pyplot as plt
import base64
import io
import requests, time, os, json, re
import pandas as pd
from gtts import gTTS
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import warnings
warnings.filterwarnings('ignore')
from youtube_transcript_api import YouTubeTranscriptApi
from deep_translator import GoogleTranslator
import openai

openai.api_key = os.getenv("API_KEY")


# Configuring Flask
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = os.getenv("SENDER_MAIL")
app.config['MAIL_PASSWORD'] = os.getenv("SENDER_PASSWORD")
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def trans(to_translate, source='auto', target='en'):
    #to_translate = 'Ich möchte diesen Text übersetzen'
    translated = GoogleTranslator(source='auto', target='en').translate(to_translate)
    return translated

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]

def annotator(text):
    messages=[{"role": "system","content":f"You are given a text, understand it and annotate the important parts. Text: {text}"}]
    return get_completion_from_messages(messages)

def youtube_trancription(youtube_video):
    video_id = youtube_video.split("=")[1]
    YouTubeTranscriptApi.get_transcript(video_id)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    df = pd.DataFrame(transcript)
    result = ""
    for i in transcript:
        result += ' ' + i['text']
    return df, result

def query(prompt):
    HuggingFaceAPI = os.getenv("HUGGING_FACE_API")
    API_URL = "https://api-inference.huggingface.co/models/Adrenex/fastgen"
    headers = {"Authorization": f"Bearer {HuggingFaceAPI}"}
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content

def transcribe(audio_file, language, audio_type):
    
    host = "https://api.voicegain.ai/v1"
    JWT = os.getenv("JWT")
    audio_type = f"audio/{audio_type}"
    data_url = "{}/data/file".format(host)
    headers = {"Authorization": JWT}

    print(language, flush=True)
    asr_body = {
     "sessions": [{
      "asyncMode": "OFF-LINE",
      "audioChannelSelector": "two-channel",
      "poll": {
       "persist": 120000
      },
      "content": {
       "incremental": ["progress"],
       "full" : ["words","transcript"]
      }
     }],
     "audio":{
      "source": {
       "dataStore": {
        "uuid": "<data-object-UUID>"
       }
      }
     },
     "settings": {
      "asr": {
        "acousticModelNonRealTime" : "whisper",
        "languages" : [language]
      }
     }
    }
    
    def helper(asr_body,headers,fname):
        asr_response = requests.post("{}/asr/transcribe/async".format(host), json=asr_body, headers=headers).json()
        print(asr_response, flush=True)
        session_id = asr_response["sessions"][0]["sessionId"]
        polling_url = asr_response["sessions"][0]["poll"]["url"]

        index = 0
        c=0
        while True:
            if (index < 5):
                time.sleep(0.3)
            else:
                time.sleep(4)
            poll_response = requests.get(polling_url + "?full=false", headers=headers).json()
            phase = poll_response["progress"]["phase"]
            is_final = poll_response["result"]["final"]
            print("Phase: {} Final: {}".format(phase, is_final), flush=True)

            if phase == "QUEUED":
                c+=1

            if c>5:
                return " "

            index += 1
            if is_final:
                break

        txt_url = "{}/asr/transcribe/{}/transcript?format=json-mc".format(host, session_id)
        txt_response = requests.get(txt_url, headers=headers).json()
        return txt_response
    
    def process_one_file(audio_file):

        #Base filename
        filename = audio_file.filename

        print("Processing {}".format(filename), flush=True)

        audio_data = audio_file.read()
        with open('output.wav', 'wb') as f:
            f.write(audio_data)

        #uploading file
        data_body = {
            "name": re.sub("[^A-Za-z0-9]+", "-", filename),
            "description": 'output.wav',
            "contentType": audio_type,
            "tags": ["test"]
        }

        multipart_form_data = {
            'file': ('output.wav', open('output.wav', 'rb'), audio_type),
            'objectdata': (None, json.dumps(data_body), "application/json")
        }

        data_response = None
        data_response_raw = None
        try:
            data_response_raw = requests.post(data_url, files=multipart_form_data, headers=headers)
            data_response = data_response_raw.json()
        except Exception as e:
            print(str(data_response_raw))
            exit()
        if data_response.get("status") is not None and data_response.get("status") == "BAD_REQUEST":
            exit()
        object_id = data_response["objectId"]
        asr_body["audio"]["source"]["dataStore"]["uuid"] = object_id

        #Change the language as per the aws dataset
        txt_response = helper(asr_body,headers,filename)

        #Save into dataset
        dataset = pd.DataFrame(columns=['file', 'utterance', 'confidence', 'start', 'duration', 'spk', 'language'])
        for idx, item in enumerate(txt_response):
            utterances = [word['utterance'] for word in item['words']]
            dataset.loc[idx] = [filename, ' '.join(utterances), item['words'][0]['confidence'], item['start'], item['duration'], item['spk'], language]

        sorted_df = dataset.sort_values(by=['file', 'start'])
        sorted_df.reset_index(drop=True, inplace=True)  # Reset the index to have unique values

        return sorted_df
    
    df = pd.DataFrame(columns=['file', 'utterance', 'confidence', 'start', 'duration', 'spk', 'language'])
    df = process_one_file(audio_file)
    df['endTime'] = df['start'] + df['duration']
    df.rename(columns={'start': 'startTime', 'spk': 'speakerId', 'utterance': 'Dialogue'}, inplace=True)
    df['startTime'] = df['startTime'] / 1000
    df['endTime'] = df['endTime'] / 1000
    grouped_df = df.groupby('file')['Dialogue'].agg(' '.join).reset_index()
    columns_to_remove = ['confidence', 'language', 'duration','file']
    df = df.drop(columns=columns_to_remove)
    return grouped_df['Dialogue'][0], df

########################### Routing Functions ########################################

@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/browse')
def browse():
    return render_template('browse.html')

@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/annotate')
def annotate():
    return render_template('annotate.html')

@app.route('/translate')
def translate():
    return render_template('translate.html')

@app.route('/yttranscribe')
def yttranscribe():
    return render_template('yttranscribe.html')

@app.route('/speech2text')
def speech2text():
    return render_template('speech2text.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')

@app.route('/summarize')
def summarize():
    return render_template('summarize.html')

@app.route('/extract')
def extract():
    return render_template('extract.html')

@app.route('/textbot')
def textbot():

    global bot_messages
    global history
    bot_messages = []
    history = []

    return render_template('textbot.html')

@app.route('/text2image')
def text2image():
    return render_template('text2image.html')

@app.route('/text2speech')
def text2speech():
    return render_template('text2speech.html')

@app.route('/wordcloud')
def wordcloud():
    return render_template('wordcloud.html')


@app.route('/sendfeedback', methods=['POST'])
def sendfeedback():
    if request.method == 'POST':

        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        subject = 'Feedback from Website'
        body = f"Name: {name}\nEmail: {email}\nMessage: {message}"
        msg = Message(subject, sender ='anonymousadrenex@gmail.com', recipients=['sherushernisheru@gmail.com'])  # Receiver email address
        msg.body = body
        mail.send(msg)

        subject = 'TextMasters'
        body = f"Thank you for your Feedback.\nIf you like the project, please follow me on Github: https://github.com/adrenex and connect with me on Linkedin: https://www.linkedin.com/in/nileshpopli.\n\nNilesh Popli"
        msg = Message(subject, sender ='anonymousadrenex@gmail.com', recipients=[email])  # Receiver email address
        msg.body = body
        mail.send(msg)
        
        return render_template('feedback.html', ft="Feedback Sent, Thank you for your time :)")

@app.route('/result_annotate', methods=['POST'])
def result_annotate():
    if request.method == 'POST':
        original_text = request.form['originalText']
        annotated_text = annotator(original_text)
        return render_template('result_annotate.html', ot=original_text, at=annotated_text)


@app.route('/result_translate', methods=['POST'])
def result_translate():
    if request.method == 'POST':
        original_text = request.form['originalText']
        source_language = request.form['sourceLanguage']
        destination_language = request.form['destinationLanguage']
        translated_text = trans(original_text)
        return render_template('result_translate.html', ot=original_text, sl=source_language, dl=destination_language, tt=translated_text)

@app.route('/result_yttranscribe', methods=['POST'])
def result_yttranscribe():
    if request.method == 'POST':
        video_link = request.form['videolink']
        df, combined =youtube_trancription(video_link)
        df_html = df.to_html(classes='table table-bordered')
        return render_template('result_yttranscribe.html', vl=video_link, cb=combined, df=df_html)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    
    # Get the DataFrame from the POST request
    html_table = request.form['data']
    df = pd.read_html(html_table)[0]
    # Generate the CSV data
    csv_data = df.to_csv(index=False)

    # Create a Response object with the CSV data
    response = Response(
        csv_data,
        content_type='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=data.csv'
        }
    )
    return response


@app.route('/result_speech2text', methods=['POST'])
def result_speech2text():
    if request.method == 'POST':
        audio_file = request.files['audio_file']
        language = request.form['language']
        audio_type = request.form['type']
        combined, df = transcribe(audio_file, language, audio_type)
        df_html = df.to_html(classes='table table-bordered')
        return render_template('result_speech2text.html', cb=combined, df=df_html)
    

@app.route('/result_detection', methods=['POST'])
def result_detection():
    if request.method == 'POST':
        original_text = request.form['originalText']
        detected_language = detect(original_text)
        return render_template('result_detection.html', ot=original_text, dl=detected_language)
    

@app.route('/result_summarize', methods=['POST'])
def result_summarize():
    if request.method == 'POST':
        original_text = request.form['originalText']
        messages=[
            {"role": "system","content":" Summarize the following text. Reduce the content by 75%."},
            {"role": "user", "content": original_text}
        ]
        summarized_text =  get_completion_from_messages(messages)
        return render_template('result_summarize.html', ot=original_text, st=summarized_text)
    

@app.route('/result_extract', methods=['POST'])
def result_extract():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        reader = PdfReader(pdf_file)
        num_pages = len(reader.pages)
        result = ""
        for i in range(num_pages):
            result += (f"Page: {i+1}\n\n")
            page = reader.pages[i]
            text = page.extract_text() 
            result += (text+"\n")
        return render_template('result_extract.html', np=num_pages, et=result)
    
bot_messages = []
history = []
    
@app.route('/result1_textbot', methods=['POST'])
def result1_textbot():
    if request.method == 'POST':

        global bot_messages
        global history
        bot_messages = []
        history = []

        original_text = request.form['originalText']
        base={"role": "system","content":f"""You are given a text. Understand the text thoroughly and deeply, and then answer the following questions, considering only the provided text.
             Please find the text enclosed within triple back ticks. ```{original_text}```"""}
        bot_messages.append(base)

        history.append(f"Text: {original_text}")

        return render_template('result_textbot.html', input_history=history)
    

    
@app.route('/result2_textbot', methods=['POST'])
def result2_textbot():
    if request.method == 'POST':

        original_text = request.form['originalText']

        bot_messages.append({'role':'user', 'content':f"{original_text}"})
        history.append(f"User: {original_text}")

        response = get_completion_from_messages(bot_messages)
        print("Assistant: "+ response)

        bot_messages.append({'role':'assistant', 'content':f"{response}"})
        history.append(f"Assitant: {response}")

        return render_template('result_textbot.html', input_history=history)
    

@app.route('/result_text2image', methods=['POST'])
def result_text2image():
    if request.method == 'POST':
        prompt = request.form['prompt']
        image_bytes = query(prompt)
        print(image_bytes, flush=True)
        image = Image.open(io.BytesIO(image_bytes))
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='JPEG')
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return render_template('result_text2image.html', pmt=prompt, ib=img_data)
    

@app.route('/result_text2speech', methods=['POST'])
def result_text2speech():
    if request.method == 'POST':
        original_text = request.form['originalText']
        language = request.form['language']

        # Generate the speech and save it to an in-memory file
        speech = gTTS(text=original_text, lang=language, slow=False, tld="com.au")
        audio_file = io.BytesIO()
        speech.write_to_fp(audio_file)
        audio_file.seek(0)

        # Convert the BytesIO object to a base64-encoded string
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        return render_template('result_text2speech.html', txt=original_text, l=language, audio_base64=audio_base64)

    

@app.route('/result_wordcloud', methods=['POST'])
def result_wordcloud():
    if request.method == 'POST':
        original_text = request.form['originaltext']
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(original_text)
        img_buffer = BytesIO()
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(img_buffer, format='png')
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return render_template('result_wordcloud.html', ot=original_text, wc=img_data)
    

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
