


1.	Public Consultation
    1.	21st Feb – Weekly Report on Text Summarisation
    1.	28th Feb – Given a task to report on the most common points made in feedbacks to the Privacy Act Review Report
    1.	3rd Mar – Weekly Report on Commonwealth Parole Office – Scope and Objectives clarification
    1.	7th Mar – Weekly Report on Public Consultation Submission Analysis – Feedbacks extraction begins
    1.	11th Mar – Submit extracted responses in privacy_act_responses.jsonl
    1.	11th Mar – Umar responded with 3 highlights of inconsistencies found in the above deliverable
    1.	12th Mar – Updated jsonl file with PDF attachment links sent to Umar for review
    1.	13th Mar – Comments sought for responses containing both free text and selections
    1.	13th Mar – Free text and selections are combined into a dictionary as advised by Umar
    1.	14th Mar – Consultation_Responses.txt and consultation_responses.jsonl submitted for review
    1.	14th Mar – Weekly Report on Consultation Response Data gathering
    1.	14th Mar – Types of data such as text vs radio_checkboxes vs PDF and their counts, questions and their response types (text vs radio/checkbox) submitted
    1.	17th Mar – Missing question-answer pairs bug fixed and “\r” foreign characters removed
    1.	17th Mar – New task to focus on extracting PDF attachments from responses
    1.	17th Mar – Now, the task uses public data to simulate working with sensitive information, ie adhering to internal data handling constraints in protected environment such as designated AWS account
    1.	21st Mar – Weekly Report on setting up AWS env and testing small LM – Gemma, Mistral and Qwen, finally settling for Qwen 2.5 7B instruction-following model
    1.	21st Mar – New task to use AI to segment and extract PDF text
    1.	21st Mar – Give a heads-up about upcoming mid-cycle PPI
    1.	28th Mar – Umar going on cultural leave
    1.	28th Mar – New task to summarise responses to a question as 1 line in jsonl format, containing question and a summary of its responses.  More importantly, the GenAI model must satisfy the following criteria:
        1.	capable of fitting on a single Nvidia A10G GPU (this corresponds to a g5.xlarge AWS instance);
        1.	freely and openly available on Hugging Face;
        1.	is licensed under a license that is at least as permissive as CC BY 4.0 NC SA (essentially, it must be permissive of non-commercial usage but it can be share alike/copyleft since we will not be finetuning it); and
        1.	is the product of a reputable organisation or academic.
    1.	28th Mar – Submit Create_Response_Centric_UDS.txt containing extracted text/radiobutton/checkbox/PDF from response 838011019 as an example
    1.	28th Mar – Explain where time is taken to process text/selection/PDF per responses and then all responses per question
    1.	28th Mar – Weekly Report on merging text/selection/PDF per response, segment the merged text to match question and tested using response ID=838011019
    1.	4th Apr – Weekly Report on inconsistent behaviour of summarization of combined (form+PDF) answers to a question.  Also experimenting with relative performances of Llama, Gemma, Mistral and Qwen
    1.	9th Apr – Submitted question-centric form-only data summaries with 10 max responses taken into account
    1.	9th Apr – Submitted updated summaries without the above limitations
    1.	10th Apr – Submitted my response to mid-term PPI against objectives set out in the PPI from the beginning.
    1.	11th Apr – Weekly Report on experimentation of PDF data summarization
    1.	24th Apr – Weekly Report on batching PDF-Data summarization and experimentation with TTS strategies
    1.	28th Apr – Submitted an interim report on the status of the project, ie it has successfully summarised 289 out of X total responses.
    1.	28th Apr – Weekly Report on pretty much the same tasks as previous week
    1.	2th May – Weekly Report on 1) research on improving the summarization pipeline 2) tradeoff between the convenience of hosted LLM via API access vs Local Control of small LM
    1.	8th May – Performance Review

