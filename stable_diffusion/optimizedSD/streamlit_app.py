#!/usr/bin/env python

####################
# Required Modules #
####################

import argparse
import asyncio
import base64
import json
import os
import re
import time
from contextlib import contextmanager, nullcontext
from itertools import islice
from pathlib import Path
from random import randint

import numpy as np
import pyaudio
#Generic Built in
import streamlit as st
import torch
import websockets
from einops import rearrange
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from optimizedSD.optimUtils import logger, split_weighted_subprompts
from PIL import Image,ImageDraw
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from stqdm import stqdm
from transformers import logging
from keybert import KeyBERT
from summarizer import Summarizer
import textwrap


##################
# Configurations #
##################

# from samplers import CompVisDenoiser
logging.set_verbosity_error()
# Audio parameters 
st.sidebar.header('Audio Parameters')

FRAMES_PER_BUFFER = int(st.sidebar.text_input('Frames per buffer', 3200))
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = int(st.sidebar.text_input('Rate', 16000))
p = pyaudio.PyAudio()
TRANSCRIPTION_PATH = "./transcription.txt"
TRANSCRIPTION_OUTPUT_PATH = "./transcription_output.txt"
SUMMARISED_TEXT = "./summarised.txt"
STYLES=",stylised,concept,infographic,professional"

#############
# Functions #
#############
# Start/stop audio transmission
def start_listening():
	st.session_state['run'] = True

def download_transcription():
	read_txt = open(TRANSCRIPTION_PATH, 'r')
	st.download_button(
		label="Download transcription",
		data=read_txt,
		# file_name=TRANSCRIPTION_OUTPUT_PATH,
		file_name='transcription_output.txt',
		mime='text/plain')

def stop_listening():
	st.session_state['run'] = False



async def terminate_session(socket):
	try:
		print("Terminating session...")
		termination = await socket.send(json.dumps({
					"terminate_session":True
				}))
		print(f"Termination...:{termination}")
		termination_status = await socket.recv()
		print(f"Termination status:{termination_status}")
	except Exception as e:
		print("Error occured while terminating sessions...")
		print(e)
	return True

# Send audio (Input) / Receive transcription (Output)
async def send_receive():
	if st.session_state['run']:
		URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate={RATE}"
		# RECONNECT_URL = f"wss://api.assemblyai.com/v2/realtime/ws/{st.session_state['session_id']}"
		# print("Checking if session already exists...")
		
		print(f'Connecting websocket to url ${URL}')
		CONNECTIONS = []
		async with websockets.connect(
			URL,
			extra_headers=(("Authorization", st.secrets['api_key']),),
			ping_interval=5,
			ping_timeout=20
		) as _ws:
			
			# CONNECTIONS.append(_ws)
			# print(f"Connections:{CONNECTIONS}")
			# st.session_state["socket"] = _ws
			r = await asyncio.sleep(0.1)
			print("Receiving messages ...")

			session_begins = await _ws.recv()
			session_begins_json = json.loads(session_begins)
			print(f"PRINTING SESSION STATE:{st.session_state}")
			if "session_id" in session_begins_json.keys():
				st.session_state["session_id"] = session_begins_json["session_id"]
				print(f"Session id is {st.session_state['session_id']}")
			elif "error" in session_begins_json.keys():
				error_msg = session_begins_json["error"]
				if "exceeded the number of allowed streams" in error_msg:
					print(error_msg)
					st.markdown(f'{error_msg}.\n Please try again in a few minutes when the stream expires and is reset.')

			print("Sending messages ...")
			async def send():
				while st.session_state['run']:
					try:
						print("Trying to read the stream")
						data = stream.read(FRAMES_PER_BUFFER)
						data = base64.b64encode(data).decode("utf-8")
						json_data = json.dumps({"audio_data":str(data)})
						print("Sending the stream")
						r = await _ws.send(json_data)

					except websockets.exceptions.ConnectionClosedError as e:
						print(e)
						print("ConnectionClosedError while sending in a connected session.")
						# await terminate_session(_ws)
						assert e.code == 4008
						break

					except Exception as e:
						print(e)
						assert False, "Not a websocket 4008 error"

					r = await asyncio.sleep(0.01)


			async def receive():
				while st.session_state['run']:
					try:
						result_str = await _ws.recv()
						result = json.loads(result_str)['text']

						if json.loads(result_str)['message_type']=='FinalTranscript':
							print(result)
							st.session_state['text'] = result
							st.write(st.session_state['text'])

							transcription_txt = open('transcription.txt', 'a')
							transcription_txt.write(st.session_state['text'])
							transcription_txt.write(' ')
							transcription_txt.close()
				

					except websockets.exceptions.ConnectionClosedError as e:
						print(e)
						print("ConnectionClosedError while receiving in a connected session")
						# await terminate_session(_ws)
						assert e.code == 4008
						break

					except Exception as e:
						print("Other exception occured in a connected session...")
						print(e)	
						assert False, "Not a websocket 4008 error"
			print("\n Awaiting results...")
			send_result, receive_result = await asyncio.gather(send(), receive())
			print("\n Results awaited!")

			# await terminate_session(_ws)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

def generate_img(prompt:str):
	opt = OPT(prompt=prompt)
	tic = time.time()
	os.makedirs(opt.outdir, exist_ok=True)
	outpath = opt.outdir
	grid_count = len(os.listdir(outpath)) - 1

	if opt.seed == None:
		opt.seed = randint(0, 1000000)
	seed_everything(opt.seed)

	# Logging
	logger(vars(opt), log_csv = "logs/txt2img_logs.csv")

	sd = load_model_from_config(f"{opt.ckpt}")
	li, lo = [], []
	for key, value in sd.items():
		sp = key.split(".")
		if (sp[0]) == "model":
			if "input_blocks" in sp:
				li.append(key)
			elif "middle_block" in sp:
				li.append(key)
			elif "time_embed" in sp:
				li.append(key)
			else:
				lo.append(key)
	for key in li:
		sd["model1." + key[6:]] = sd.pop(key)
	for key in lo:
		sd["model2." + key[6:]] = sd.pop(key)

	config = OmegaConf.load(f"{opt.config}")

	model = instantiate_from_config(config.modelUNet)
	_, _ = model.load_state_dict(sd, strict=False)
	model.eval()
	model.unet_bs = opt.unet_bs
	model.cdevice = opt.device
	model.turbo = opt.turbo

	modelCS = instantiate_from_config(config.modelCondStage)
	_, _ = modelCS.load_state_dict(sd, strict=False)
	modelCS.eval()
	modelCS.cond_stage_model.device = opt.device

	modelFS = instantiate_from_config(config.modelFirstStage)
	_, _ = modelFS.load_state_dict(sd, strict=False)
	modelFS.eval()
	del sd

	if opt.device != "cpu" and opt.precision == "autocast":
		model.half()
		modelCS.half()

	start_code = None
	if opt.fixed_code:
		start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device)


	batch_size = opt.n_samples
	n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
	if not opt.from_file:
		assert opt.prompt is not None
		prompt = opt.prompt
		print(f"Using prompt: {prompt}")
		data = [batch_size * [prompt]]

	else:
		print(f"reading prompts from {opt.from_file}")
		with open(opt.from_file, "r") as f:
			text = f.read()
			print(f"Using prompt: {text.strip()}")
			data = text.splitlines()
			data = batch_size * list(data)
			data = list(chunk(sorted(data), batch_size))


	if opt.precision == "autocast" and opt.device != "cpu":
		precision_scope = autocast
	else:
		precision_scope = nullcontext

	seeds = ""
	with torch.no_grad():

		all_samples = list()
		for n in trange(opt.n_iter, desc="Sampling"):
			for prompts in tqdm(data, desc="data"):

				sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompts[0])))[:150]
				os.makedirs(sample_path, exist_ok=True)
				base_count = len(os.listdir(sample_path))

				with precision_scope("cuda"):
					modelCS.to(opt.device)
					uc = None
					if opt.scale != 1.0:
						uc = modelCS.get_learned_conditioning(batch_size * [""])
					if isinstance(prompts, tuple):
						prompts = list(prompts)

					subprompts, weights = split_weighted_subprompts(prompts[0])
					if len(subprompts) > 1:
						c = torch.zeros_like(uc)
						totalWeight = sum(weights)
						# normalize each "sub prompt" and add it
						for i in range(len(subprompts)):
							weight = weights[i]
							# if not skip_normalize:
							weight = weight / totalWeight
							c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
					else:
						c = modelCS.get_learned_conditioning(prompts)

					shape = [opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f]

					if opt.device != "cpu":
						mem = torch.cuda.memory_allocated() / 1e6
						modelCS.to("cpu")
						while torch.cuda.memory_allocated() / 1e6 >= mem:
							time.sleep(1)

					samples_ddim = model.sample(
						S=opt.ddim_steps,
						conditioning=c,
						seed=opt.seed,
						shape=shape,
						verbose=False,
						unconditional_guidance_scale=opt.scale,
						unconditional_conditioning=uc,
						eta=opt.ddim_eta,
						x_T=start_code,
						sampler = opt.sampler,
					)

					modelFS.to(opt.device)

					print(samples_ddim.shape)
					print("saving images")
					for i in range(batch_size):

						x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
						x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
						x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
						Image.fromarray(x_sample.astype(np.uint8)).save(
							os.path.join(sample_path, "seed_" + str(opt.seed) + "_" + f"{base_count:05}.{opt.format}")
						)
						seeds += str(opt.seed) + ","
						opt.seed += 1
						base_count += 1

					if opt.device != "cpu":
						mem = torch.cuda.memory_allocated() / 1e6
						modelFS.to("cpu")
						while torch.cuda.memory_allocated() / 1e6 >= mem:
							time.sleep(1)
					del samples_ddim
					print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

	toc = time.time()

	time_taken = (toc - tic) / 60.0

	print(
		(
			"Samples finished in {0:.2f} minutes and exported to "
			+ sample_path
			+ "\n Seeds used = "
			+ seeds[:-1]
		).format(time_taken)
	)
	
def prep_generation():
	print("Checking for transcription_output.txt...")	
	if Path(TRANSCRIPTION_OUTPUT_PATH).is_file():
		print("transcription output found")

		# read the transcipted prompt
		with open(TRANSCRIPTION_OUTPUT_PATH,"r") as f:
			doc = f.readlines()
		print(f"DOC:{doc}")
		print(f"DOC shape:{np.shape(doc)}")
		st.text("Extracting keywords....")
		kw_extractor = KeyBERT()
		st.text("Keywords extracted!")
		# keywords are in the format [[("keyword",prob),("keyword",prob),("keyword",prob)]]
		extracted = kw_extractor.extract_keywords(doc)
		print(f"EXTRACTED shape:{len(np.shape(extracted))}")
		keywords_list = []
		print(f"EXTRACTED :\n{extracted}")
		for keyword in extracted:
			print(f"KEYWORD:{keyword}, LENGTH:{len(keyword)}")
			if len(keyword)> 0:
				keywords_list.append(keyword[0])
		keywords = " ".join(keywords_list)	
		st.text(f"keywords are : {keywords}")
		keywords+= STYLES
		print("KEYWORDS ARE....")
		print(keywords)
		# Summariser the transcribed text
		text = " ".join(doc)
		summariser = Summarizer()
		
		print(f"TRANSCRIBED TEXT IS :{text}")
		print("\n")

		# edit ratio to change how much the text is summarised
		summarised_doc = summariser(text,ratio=0.3)
		with open(SUMMARISED_TEXT,"w") as file:
			file.write(summarised_doc)

		print(f"SUMMARISED TEXT IS :{summarised_doc}")
		st.markdown(f"Summarised text is :\n{summarised_doc}")
		# Use Stable Diffusion to generate keywords
		generate_img(prompt=keywords)
		os.remove(TRANSCRIPTION_OUTPUT_PATH)
		# look for output folder
		# keywords = "transcribe multimodal transcription text speech"
		replaced_keywords = keywords.replace(" ","_")
		# styles = ",chinese_inkbrush,stylised,concept"
		output_folder = replaced_keywords
		# Get all the generated images
		infographics = []
		for root,dirs,files in os.walk(f"./stable_diffusion/outputs/txt2img-samples/{output_folder}"):
			for file in files:
				filename = "./"+root+"/"+file
				infographics.append(filename)
		text_list = textwrap.wrap(summarised_doc,width=40)
		base_img = Image.new('RGB', (1024, 1024))
		d = ImageDraw.Draw(base_img)
		infographic = Image.open(infographics[0])
		text = ""
		for each in text_list:
			text += each+"\n"
		d.multiline_text((20,20),text=text,anchor="la", fill=(255, 0, 0))
		base_img.paste(infographic,(500,20))
		base_img.save("generated_infographic.png")
		st.write(f"Generated infographic saved to base folder!")
	
###########
# Classes #
###########

class OPT():
	def __init__(self,prompt="Samurai fighing dragon in an abandoned battlefield, inkbrush chinese") -> None:
		self.prompt = prompt
		self.outdir = "./stable_diffusion/outputs/txt2img-samples"
		self.skip_grid = False
		self.skip_save = False
		self.ddim_steps = 50
		# self.plms = False
		# self.laion400m = False
		self.fixed_code = False
		self.ddim_eta = 0.0
		self.n_iter = 1
		self.H = 512
		self.W = 512
		self.C = 4
		self.f = 8
		self.n_samples = 1
		self.n_rows = 0
		self.scale = 7.5
		self.device = "cuda"
		self.from_file = None
		self.config = "./stable_diffusion/optimizedSD/v1-inference.yaml"
		self.ckpt = "./stable_diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt"
		self.seed = 42
		self.unet_bs = 1
		self.turbo = False
		self.format  ="png"
		self.sampler = "plms"
		self.precision = "autocast"

##########
# Script #
##########


# Session state
if 'text' not in st.session_state:
	st.session_state['text'] = 'Listening...'
	st.session_state['run'] = False

# Open an audio stream with above parameter settings
stream = p.open(
   format=FORMAT,
   channels=CHANNELS,
   rate=RATE,
   input=True,
   frames_per_buffer=FRAMES_PER_BUFFER
)



# Web user interface
st.title('🎙️ AssemblyAI Hackathon Real-Time Speech to Image Generation App')

with st.expander('About this App'):
	st.markdown('''
	This Streamlit app uses the AssemblyAI API to perform real-time transcription.The transcribed text 
	then has its keywords extracted and an image is then generated using the extracted keywords
	
	Libraries used:
	- `stable-diffusion` - text to image generator
	- `KeyBERT` - keyword extraction
	- `AssemblyAI` - Real time transciption

	- `streamlit` - web framework
	- `pyaudio` - a Python library providing bindings to [PortAudio](http://www.portaudio.com/) (cross-platform audio processing library)
	- `websockets` - allows interaction with the API
	- `asyncio` - allows concurrent input/output processing
	- `base64` - encode/decode audio data
	- `json` - allows reading of AssemblyAI audio output in JSON format
	''')

col1, col2 = st.columns(2)

col1.button('Start', on_click=start_listening)
col2.button('Stop', on_click=stop_listening)

# Run
asyncio.run(send_receive())

# Checks for the presence of the transcription file
print("Checking for transcription.txt...")
if Path(TRANSCRIPTION_PATH).is_file():
	print("transcription found")
	st.markdown(
		'''
		### Download
		## Please do NOT change the default file name!
		'''
		)
	print("running download transcription")
	download_transcription()
	os.remove(TRANSCRIPTION_PATH)
	# st.button("Start Image generation",on_click=prep_generation())

st.button("Click this after you have downloaded your transcription",on_click=prep_generation())


