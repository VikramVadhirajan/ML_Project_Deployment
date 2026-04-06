# Azure Custom Vision – Object Detection from Video 🎥🧠

This project demonstrates how to build an **Object Detection model using Azure Custom Vision** by extracting frames from a driving video dataset, training a model using the **Azure Custom Vision Python SDK**, and generating predictions.

The project also reconstructs the processed frames into a **prediction video showing object detection results**.

---

# 📌 Project Overview

The goal of this project is to automate the **end-to-end object detection workflow** using Azure services and Python.

The pipeline includes:

1. Extracting frames from a video
2. Uploading images to Azure Custom Vision
3. Training an object detection model
4. Generating predictions
5. Creating an output video with detected objects

This project demonstrates how **cloud-based computer vision models can be trained and deployed using Python automation**.

---

# 🎥 Dataset Source

The dataset used in this project comes from Kaggle:

Driving Video with Object Tracking Dataset
https://www.kaggle.com/datasets/robikscube/driving-video-with-object-tracking/data

The dataset contains driving videos with objects such as vehicles and road elements which are useful for **training object detection models**.

---

# 🧠 Project Workflow

```id="wf_pipeline"
Driving Video
     │
     ▼
Frame Extraction (OpenCV)
     │
     ▼
Upload Images → Azure Custom Vision
     │
     ▼
Model Training
     │
     ▼
Prediction on Frames
     │
     ▼
Reconstruct Frames → Output Video
```

---

# 📂 Project Files

```
Project/
│
├── experiments.ipynb               # Main notebook for training & experimentation
├── requirements.txt                # Python dependencies
├── taggedvideo.mp4     # Generated video with predictions
│
├── Delete_Resource_Group.ipynb     # Notebook to delete Azure resources
├── azure_cli_commands.md           # Azure CLI reference commands
│
└── README.md
```

---

# ⚙️ Technologies Used

Python
Azure Custom Vision SDK
OpenCV
NumPy
Azure CLI
Jupyter Notebook

---

# 🚀 Steps Performed in the Notebook

## 1️⃣ Azure Authentication

The notebook connects to **Azure Custom Vision** using the training key and endpoint.

Example environment variables:

```
VISION_TRAINING_KEY="your_training_key"
VISION_TRAINING_ENDPOINT="your_training_endpoint"
```

⚠️ These values should **never be committed to GitHub**.

---

# 2️⃣ Create Azure Custom Vision Project

The project is created programmatically using the Azure SDK.

```python
trainer.create_project()
```

---

# 3️⃣ Extract Frames from Video

The driving video is processed using **OpenCV** to extract frames.

```python
cv2.VideoCapture()
```

These frames serve as the training dataset for the model.

---

# 4️⃣ Upload Images to Azure Custom Vision

The extracted frames are uploaded to the Azure Custom Vision project along with labels.

This prepares the dataset for model training.

---

# 5️⃣ Train the Model

The model is trained using Azure Custom Vision:

```python
trainer.train_project()
```

The trained model iteration is then published.

---

# 6️⃣ Generate Predictions

The trained model is used to predict objects on video frames.

Bounding boxes are drawn on frames to highlight detected objects.

---

# 7️⃣ Reconstruct Prediction Video

The processed frames are combined to create the final video output.

Output file:

```
prediction_output_video.mp4
```

The video shows **object detection results frame-by-frame**.

---

# ⚙️ Installation

Install required dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Running the Notebook

Launch Jupyter Notebook:

```
jupyter notebook
```

Open:

```
experiments.ipynb
```

Run the notebook cells step-by-step.

---

# ☁️ Azure CLI Commands

Useful commands to inspect Azure resources.

### Login to Azure

```
az login
```

---

### List Resource Groups

```
az group list
```

---

### List Azure Resources

```
az resource list
```

---

### View Resource Details

```
az resource show --name <resource-name>
```

---

# 🧹 Deleting Azure Resources

To avoid unnecessary Azure costs, delete the resource group after completing experiments.

### Delete Resource Group

```
az group delete --name <resource-group-name>
```

Alternatively, run the included notebook:

```
delete_resource_group.ipynb
```

---

# 📊 Key Learning Outcomes

This project demonstrates:

✔ Using **Azure Custom Vision SDK with Python**
✔ Automating ML pipelines in the cloud
✔ Video preprocessing with OpenCV
✔ Training object detection models
✔ Generating prediction videos
✔ Managing Azure resources using CLI

---

# 👨‍💻 Author

**Vikram Vadhirajan**

Data Analyst | Machine Learning | Python | Azure

GitHub
https://github.com/VikramVadhirajan

---

# ⭐ Support

If you found this project useful, consider giving the repository a ⭐
