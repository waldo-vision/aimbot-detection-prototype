import sys
import torch
import os

assert len(sys.argv) >= 2, "File requires input path"

inFile = sys.argv[1]
assert os.path.isfile(inFile), "not a valid file"

inDir = False

if len(sys.argv) > 2:
    inDir = sys.argv[2]
    assert os.path.isdir(inDir), "not a valid directory"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load("./models/model.pt")

data = torch.load(inFile)

times = []

if inDir:
    times = [ (f.name).replace("_", ":") for f in os.scandir(inDir) if f.is_dir() ]
    assert len(times) == data.shape[0], "number of clips doesnt match feature data"


model.eval()
with torch.no_grad():

    for i in range(len(data)):
        curr_test = data[i].unsqueeze(0)
        output = model(curr_test)
        output = output.item()

        print(f'Clip at {times[i]}: {"Regular Gameplay Detected" if output < 0.5 else "Aimbot Detected"}')
