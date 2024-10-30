# Arabic Text Summarization
## Text Summarization
Is done by fine-tuning a pretrained version of mT5 which was trained on a large Arabic corpus for the summarization task.  
<a href="https://huggingface.co/malmarjeh/t5-arabic-text-summarization">mT5 Model on hugging-face</a>, you can find the fine-tuned version on hugging-face through <a href="https://huggingface.co/ranwakhaled/fine-tuned-T5-for-Arabic-summarization">this link</a>  
Training was done using the <a href="">AGS Corpus dataset</a> which is the first publicly accessible abstractive summarization dataset for Arabic. It consists of 142,000 pairs of articles and summaries, all written in Modern Standard Arabic (MSA).  
## User interface 
A website that uses HTML and CSS for the front end and Flask for the back end. A block of text can be typed to be summarized or we can upload a PDF in which the characters are extracted using the <a href="https://huggingface.co/MohamedRashad/arabic-small-nougat">Arabic Small Nougat</a> model on hugging face for Arabic OCR. It's a bit slow but the accuracy shows that it far outperforms other ready-made libraries like *PyMuPDF*, *PyPDF2* and others.  
<img src="https://github.com/user-attachments/assets/ade568c0-63e7-4004-a681-24f6dec2f9ba" width=500>
<img src="https://github.com/user-attachments/assets/8ee57ac2-1c8d-4499-9438-6a17768ddcf5" width=500>

## How to run the website on your PC
1. Download all the folders
2. Create a virtual environment and activate using `.\env\Scripts\Activate`
3. Download all the required libraries by running the command `pip install -r requirements.txt`
4. Run the app using `py app.py` or `flask run`
5.   Open the domain link to see the website
