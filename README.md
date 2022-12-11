# ASSEMBLYAI HACKATHON

## About the project
Checkout generated_infographic.png for an example of the generated infographic.It ain't pretty, but it works :)

## File structure
    ------  streamlit
    |       |---- secrets.toml(update with YOUR assembyAI API KEY)
    |
    |------ stable_diffusion
    |       |
    |       |----optimizedSD(IMPT)
    |       |   |
    |       |   |----streamlit_app.py(THIS IS THE FILE TO RUN)
    |       |   |
    |       |   |---models
    |       |   |   |---ldm
    |       |   |   |   |---stable-diffusion-v1(MOVE THE sd-v1-4.ckpt weights HERE)
    |       |   |
    |       |----environment.yaml(USE THIS TO CREATE YOUR CONDA ENV)
    |
    |-------run_app.py(RUN THIS FILE TO START THE APP)

## How to run the demo
1. Create a conda environment using the environment file in stable_diffusion with the command 

    ```conda env create -f stable_diffusion/environment.yaml```
    If you have issues with creating the conda environment with environment.yaml or are stuck in idependency hell/pip installing forever,
    Please try using the environment_backup.yaml with the command

    ```conda env create -f stable_diffusion/environment_backup.yaml```
2. update the secrets.toml file in the .streamlit folder with your AssemblyAI API key
3. download the stable diffusion weights with ```curl https://f004.backblazeb2.com/file/aai-blog-files/sd-v1-4.ckpt > sd-v1-4.ckpt```
4. move the sd-v1-4.ckpy weights to stable_diffusion/models/ldm/stable-diffusion-v/
4. execute the run_app.py file in the terminal with ```python run_app.py```


## INSTRUCTIONS ON USING THE STREAM LIT APP
1. Press the Start button to start the real time transcription
2. Speak into the mic - Whatever you say will be transcribed and will be displayed on screen with a little delay.
If you get an error, pleaae wait a while before rerunning the app and trying again. You probably have too many streams active.
3. Press the Stop button when you are done.
4. Press the Download button to download your transcibed speech.(DO NOT CHANGE THE DEFAULT FILE NAME of transcription_output.txt)
5. Press the "Click this after you have downloaded your transcription" button
6. Wait for the image to be generated.(Refer to your terminal for progress)
7. The generated images will be in stable_diffusion/outputs/"keywords extracted"

