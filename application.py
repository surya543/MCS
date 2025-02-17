from flask import Flask, request, render_template, send_file
import torch
from diffusers import MusicLDMPipeline
import scipy.io.wavfile
import io

application = Flask(__name__, static_folder='static', template_folder='templates')

# Load the pipeline
path = "model"
pipe = MusicLDMPipeline.from_pretrained(path, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@application.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        negative_prompt = request.form['negative_prompt']
        num_steps = int(request.form['num_steps'])
        audio_length = float(request.form['audio_length'])

        # Ensure only one value per parameter is passed
        audio = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            audio_length_in_s=audio_length,
            num_waveforms_per_prompt=5  
        ).audios[0]

        audio_bytes = io.BytesIO()
        scipy.io.wavfile.write(audio_bytes, rate=16000, data=audio)
        audio_bytes.seek(0)
        return send_file(audio_bytes, mimetype='audio/wav')

    return render_template('index.html')

if __name__ == '__main__':
    application.run(host='0.0.0.0')
