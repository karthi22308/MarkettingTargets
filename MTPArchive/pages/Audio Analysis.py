import streamlit as st
from openai import AzureOpenAI


def transcribe_audio(audio_path):
    client = AzureOpenAI( api_version='2024-06-01',azure_endpoint='https://hexavarsity-secureapi.azurewebsites.net/api/azureai',api_key='4ceeaa9071277c5b')
    audio_path = open(audio_path, "rb")

    transcript = client.audio.transcriptions.create(
        model="whisper-1",  # Allowed values for ApiUser: whisper-1
        file=audio_path
    )
    return transcript

def analyze_text(transcribed_text):
    prompt = f"""
    You are an AI assisting bank employees. Based on the following customer conversation:
    {transcribed_text}
    Suggest how the employee can increase his/her efficiency in converting the potential customer to by the products or increase his skills in communication .
    """
    client = AzureOpenAI( api_version='2024-06-01',azure_endpoint='https://hexavarsity-secureapi.azurewebsites.net/api/azureai',api_key='4ceeaa9071277c5b')
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
        max_tokens=256,
        top_p=0.6,
        frequency_penalty=0.7)

    return res.choices[0].message.content
# Streamlit app
st.title("Marketing Targets - Audio Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a call recording (audio file)", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    if st.button('Upload'):
        # Save uploaded file locally
        audio_path = f"temp_{uploaded_file.name}"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Transcription
        st.write("Transcribing audio...")
        transcribed_text = transcribe_audio(audio_path)
        st.write("Transcription Complete:")
        st.text(transcribed_text)

        # Analysis and Suggestions
        st.write("Analyzing transcription for improvements")
        suggestions = analyze_text(transcribed_text)
        st.write("Suggested Improvements")
        st.text(suggestions)