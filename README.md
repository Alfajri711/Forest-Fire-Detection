# 🌲🔥 Forest Fire Detection using Deep Learning

[[![Forest Fire](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Wildfire_nasa.jpg/1200px-Wildfire_nasa.jpg)](https://www.bing.com/images/search?view=detailV2&ccid=CIhsQtPc&id=99768806C116B140029D83062B882CC4797331A7&thid=OIP.CIhsQtPc7v1rG9q-igzoUQHaEI&mediaurl=https%3a%2f%2fwww.hutanhujan.org%2fphotos%2farticle%2fwide%2fxl%2fx2%2fimg-5851.jpg&exph=862&expw=1544&q=kebakaran+hutan+di+indonesia&simid=608052685881360106&FORM=IRPRST&ck=21CBE34075A9B3BAB76359E08A4A0F7F&selectedIndex=10&itb=0)](https://github.com/Alfajri711/Forest-Fire-Detection/blob/main/kebakaran%20hutan.jpg)

This project leverages deep learning to build a **forest fire detection system** using a Convolutional Neural Network (CNN). The model is trained on a dataset of fire and non-fire images, enabling it to distinguish between the two with high accuracy. The project is deployed using **Streamlit Cloud**, providing an interactive web-based interface for testing and predictions.

---

## 🚀 Features

- **Interactive Detection**: Upload an image, and the app predicts whether it shows a forest fire or not.
- **Trained Model**: A CNN trained on the Kaggle dataset: [Forest Fire Images](https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images).
- **Web Deployment**: The app is hosted on **Streamlit Cloud** for easy access.

---

## 🛠️ Installation and Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Alfajri711/Forest-Fire-Detection.git
2. Navigate to the project directory:
   ```bash
   cd Forest-Fire-Detection
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
5. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py

---

## 🌐 Live Demo
Try out the deployed application: [Forest Fire Detection App](https://forest-fire-detection-ur4lmx2d4yx7dtl6jiwpnp.streamlit.app/)

---

## 🧠 Model Training
- The model was built using TensorFlow's Keras API.
- Data Augmentation: The training dataset was augmented with techniques like horizontal flips, zooming, and rescaling.
- Architecture:
  - Convolutional layers for feature extraction.
  - Max pooling for dimensionality reduction.
  - Fully connected layers for classification.
- Loss Function: Binary Crossentropy.
- Optimizer: Adam.

---

## 📦 Dependencies
All required libraries are listed in requirements.txt. Main dependencies include:
- streamlit
- tensorflow
- numpy
- Pillow
- matplotlib

---

## 🤝 Acknowledgments
- Dataset: Forest Fire Images by Mohnish Saiprasad.
- Frameworks: TensorFlow and Streamlit.

---

## 📧 Contact
For any inquiries or feedback, please reach out to kyurazzz771@gmail.com

---

## ⭐ Support
If you find this project helpful, consider giving it a star 🌟 on GitHub!
