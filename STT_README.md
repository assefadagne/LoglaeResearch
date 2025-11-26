This is the exact setup you need to get your "Amharic AI Factory" running on the cloud.

### 📄 1. The `requirements.txt`

Save this file on your computer as `requirements.txt`. These specific versions ensure compatibility with the "8-bit" training trick that saves you money.

```text
torch
torchvision
torchaudio
transformers>=4.40.0
datasets>=2.17.0
peft>=0.10.0
accelerate>=0.27.0
bitsandbytes>=0.43.0
librosa
evaluate
jiwer
tensorboard
protobuf
scipy
```

-----

### ☁️ 2. The RunPod Setup Guide (Copy-Paste Friendly)

This guide assumes you are starting from zero. It uses the **Web Interface (JupyterLab)** so you don't need to mess with complex SSH terminals on your local computer yet.

#### **Phase 1: Rent the GPU**

1.  **Create Account:** Go to [RunPod.io](https://www.runpod.io) and sign up. Add $10 to your balance (this will last you \~10-15 hours of training).
2.  **Deploy Pod:**
      * Click **+ Deploy Pod** in the sidebar.
      * **GPU:** Select **Secure Cloud** (better reliability) and choose **RTX 4090** (Best value: \~$0.74/hr).
      * **Template:** This is critical. Click "Select Template" and search for **"RunPod PyTorch 2.1"** (or newer).
          * *Check specific version:* Ensure it has **CUDA 11.8** or **CUDA 12.x**. The default "RunPod PyTorch" image is usually perfect.
      * **Customize Deployment:** Check the box **"Start Jupyter Lab"** (usually on by default).
      * Click **Deploy**.

#### **Phase 2: Connect & Setup System**

Wait \~2 minutes for the "Connect" button to turn blue.

1.  **Enter the Lab:** Click **Connect** -\> **Connect to Jupyter Lab** (Port 8888). This opens a VS Code-like interface in your browser.
2.  **Open Terminal:** In the launcher (blue buttons), click **Terminal**.
3.  **Install System Audio Tools:** (Most templates miss this\!)
    Paste this into the terminal and hit Enter:
    ```bash
    apt-get update && apt-get install -y ffmpeg
    ```
4.  **Install Python Libraries:**
    Drag and drop your `requirements.txt` file from your computer into the left file sidebar of the browser. Then run:
    ```bash
    pip install -r requirements.txt
    ```

#### **Phase 3: Upload & Train**

1.  **Upload Script:** Drag and drop your `finetune_whisper_amharic.py` script into the file sidebar.
2.  **Upload Data (Optional):** If the script fails to download the dataset automatically (Mozilla sometimes changes links), download the Amharic "tar.gz" from Mozilla Common Voice locally, drag it into the browser sidebar, and unzip it in the terminal:
    ```bash
    tar -xvzf your_amharic_data.tar.gz
    ```
3.  **Run It:**
    In the terminal, type:
    ```bash
    python finetune_whisper_amharic.py
    ```

### 💡 Pro Tips for Success

  * **Don't close the tab?** Actually, you can\! RunPod keeps running even if you close your browser. To see the output again, just log back in and re-open Jupyter Lab.
  * **Download your model:** When training finishes, you will see a new folder `whisper-large-v3-amharic-lora`. Right-click it in the sidebar and select **Download** (or zip it first using `zip -r model.zip foldername` in the terminal for faster download).
  * **Kill the Pod:** As soon as you have your files, go back to the RunPod dashboard and click **Stop** and then **Terminate**. If you only "Stop" it, you still pay for storage (\~$0.10/day). Terminate stops the billing completely.

**Ready to deploy?** If you hit any red error text during the `pip install` phase, paste it here and I'll debug it instantly.