import sys, fitz
from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, VisionEncoderDecoderModel, NougatProcessor
from PIL import Image
import os 


# initialize the app
app = Flask(__name__)

# loading the fine tuned summarization model and the tokenizer 
summarization_model = AutoModelForSeq2SeqLM.from_pretrained("ranwakhaled/fine-tuned-T5-for-Arabic-summarization")
summarization_tokenizer = AutoTokenizer.from_pretrained("ranwakhaled/fine-tuned-T5-for-Arabic-summarization")

# loading the text extraction model for pdf
nougat_processor = NougatProcessor.from_pretrained("MohamedRashad/arabic-small-nougat")
nougat_model = VisionEncoderDecoderModel.from_pretrained("MohamedRashad/arabic-small-nougat")

context_length = 2048


# method to extract arabic text from pdf 
def extract_text_from_pdf(pdf_path):
    # open the uploaded pdf document
    doc = fitz.open(pdf_path)
    extracted_text = ""

    # loop through the pages of the document
    for page_nbr in range(len(doc)):
        # get the page 
        page = doc[page_nbr]
        # render the page to image
        pix = page.get_pixmap()
        # convert pixel map to PIL image
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        pixel_values = nougat_processor(image, return_tensors="pt").pixel_values
        
        outputs = nougat_model.generate(
        pixel_values,
        min_length=1,
        max_new_tokens=context_length,
        bad_words_ids=[[nougat_processor.tokenizer.unk_token_id]],
        )
        page_sequence = nougat_processor.batch_decode(outputs, skip_special_tokens=True)[0]
        page_sequence = nougat_processor.post_process_generation(page_sequence, fix_markdown=False)
        extracted_text += page_sequence
    
    return extracted_text


# method to summarize a body of text using the summarization model
def summarize_text(text):
    inputs = summarization_tokenizer(text, return_tensors="pt", max_length=512, truncation=True).input_ids
    # generate summary 
    summary_ids = summarization_model.generate(inputs, max_length=200, num_beams=5, length_penalty=1.2, early_stopping=True)
    # decode summary
    summary = summarization_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# index route to read and write data
@app.route('/', methods=['GET', 'POST'])
def index():
    # initialize summary as none at first
    summary = None
    processing = False

    if request.method == 'POST':
        # Check if the form has an article/ textarea input
        if 'article' in request.form and request.form['article']:
            # get article from the text area
            article = request.form['article']
            # summarize the content
            summary = summarize_text(article)
        # check if the form has an uploaded file/ pdf input
        elif 'myfile' in request.files:
            # get the uploaded pdf file
            pdf_file = request.files['myfile']
            if pdf_file and pdf_file.filename.endswith('.pdf'):
                processing = True
                
                # save the pdf file to a temporary location
                pdf_path = os.path.join("uploads", pdf_file.filename)
                pdf_file.save(pdf_path)

                # extract text from pdf
                extracted_text = extract_text_from_pdf(pdf_path)
                # summarize the extracted text
                summary = summarize_text(extracted_text)

                # clean up the temp directory
                os.remove(pdf_path)
                processing=False


    return render_template('index.html', summary=summary, processing = processing)

if __name__ == '__main__':
    app.run(debug=True)