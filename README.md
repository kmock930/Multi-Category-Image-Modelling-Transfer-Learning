# Computer Vision based Transfer Learning from this project: https://github.com/kmock930/Texture-Image-Comparison.git
## Project Preview
![image](https://github.com/user-attachments/assets/433afb73-b4bf-4a45-9c30-5ee214dab407)
![image](https://github.com/user-attachments/assets/5c80f46c-63d2-4dce-a90b-5599f9930c4c)
## Aims:
- Build 2 neural-network-based classifiers (i.e., ResNet50 and ResNet50-V2) to recognize more categories in the stonefiles dataset: https://web.engr.oregonstate.edu/~tgd/bugid/stonefly9/
- Perform Data Preprocessing.
- Evaluate the model's performance via some standard metrics and predictions. 
-  Perform Transfer Learning to learn a bounding box which accurately circles an object in an image. 
- Perform Multi-Task Regularization and Data Augmentation. 
- Evaluate the Transfer Learning process. 
## Project Prerequisites
- Run the command `pip freeze > requirements.txt` to generate the latest version of `requirements.txt` file which keeps track of all necessary pip installs.
## Project Structure
- `config.yaml` is a configuration file that stores all parameters for modelling, training, experimenting, evaluating and etc. 
- `poc_hydra.py` is a standalone script to test the interaction with hydra in order to access (read/write) parameters in config. 
- Check out my Jupyter Notebook to see my analysis.
