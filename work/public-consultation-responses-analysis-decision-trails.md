


1.	State of the art in text summarization
1.	Public Consultation
    - 21st Feb – Weekly Report on Text Summarisation
    -	28th Feb – Report on the most common points made in feedbacks to the Privacy Act Review Report
    -	3rd Mar – Weekly Report on Commonwealth Parole Office – Scope and Objectives clarification
    -	7th Mar – Weekly Report on Public Consultation Submission Analysis – Feedbacks extraction begins
    -	11th Mar – Submit extracted responses in privacy_act_responses.jsonl
    -	11th Mar – Umar responded with 3 highlights of inconsistencies found in the above deliverable
    -	12th Mar – Updated jsonl file with PDF attachment links sent to Umar for review
    -	13th Mar – Comments sought for responses containing both free text and selections
    -	13th Mar – Free text and selections are combined into a dictionary as advised by Umar
    -	14th Mar – Consultation_Responses.txt and consultation_responses.jsonl submitted for review
    -	14th Mar – Weekly Report on Consultation Response Data gathering
    -	14th Mar – Types of data such as text vs radio_checkboxes vs PDF and their counts, questions and their response types (text vs radio/checkbox) submitted
    -	17th Mar – Missing question-answer pairs bug fixed and “\r” foreign characters removed
    -	17th Mar – New task to focus on extracting PDF attachments from responses
    -	17th Mar – Now, the task uses public data to simulate working with sensitive information, ie adhering to internal data handling constraints in protected environment such as designated AWS account
    -	21st Mar – Weekly Report on setting up AWS env and testing small LM – Gemma, Mistral and Qwen, finally settling for Qwen 2.5 7B instruction-following model
    -	21st Mar – New task to use AI to segment and extract PDF text
    -	21st Mar – Give a heads-up about upcoming mid-cycle PPI
    -	28th Mar – Umar going on cultural leave
    -	28th Mar – New task to summarise responses to a question as 1 line in jsonl format, containing question and a summary of its responses.  More importantly, the GenAI model must satisfy the following criteria:
        -	capable of fitting on a single Nvidia A10G GPU (this corresponds to a g5.xlarge AWS instance);
        -	freely and openly available on Hugging Face;
        -	is licensed under a license that is at least as permissive as CC BY 4.0 NC SA (essentially, it must be permissive of non-commercial usage but it can be share alike/copyleft since we will not be finetuning it); and
        -	is the product of a reputable organisation or academic.
    -	28th Mar – Submit Create_Response_Centric_UDS.txt containing extracted text/radiobutton/checkbox/PDF from response 838011019 as an example
    -	28th Mar – Explain where time is taken to process text/selection/PDF per responses and then all responses per question
    -	28th Mar – Weekly Report on merging text/selection/PDF per response, segment the merged text to match question and tested using response ID=838011019
    -	4th Apr – Weekly Report on
