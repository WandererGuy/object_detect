# Description:
this script extract all frames from a video (keeps in folder ./frame) <br>
Then , segment isolated every object in a frame and keeps them in strutured folder ./video_objects <br>
great for surveillance system

# run 
```
python main_object_detect.py
```

# installation environment 
can be via environment.yml file or like below 
```
conda create -n env python==3.10
conda activate env
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia 
conda install anaconda::requests
conda install conda-forge::fastapi tqdm uvicorn python-multipart pydantic
pip install ultralytics
```

if needed 
```
pip install numpy==1.24.1
```

