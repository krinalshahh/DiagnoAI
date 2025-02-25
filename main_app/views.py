from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.http import JsonResponse
from datetime import date

from django.contrib import messages
from django.contrib.auth.models import User , auth
from .models import patient , doctor , diseaseinfo , consultation ,rating_review
from chats.models import Chat,Feedback
# views.py
import os
import re
import tempfile
import traceback
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

# Ensure your GOOGLE_API_KEY is set (replace with your key)
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyDgporlrHPLX7GxNRLJ9VxFA76wU55NFFc"

# Import your model inference functions
from models.AbdominalTraumaDetection import AbdominalTraumaDetection
from models.KidneyDiseasesClassification import KidneyDiseasesClassification
from models.ChestXRay import ChestXRay
from models.BoneFractures import BoneFractures
from models.Knee_Osteoporosis import Knee_Osteoporosis

# Import libraries for report generation
import google.generativeai as geni
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# --- Inference functions (same as your Streamlit functions) ---

def infer_abdominal_trauma(image_path):
    model_path = r"models/model/efficientnet_b0_Abdominal_Trauma_Detection.pth"
    trauma_detector = AbdominalTraumaDetection(model_path=model_path, threshold=0.7)
    result = trauma_detector.infer(image_path)
    description = (
        "Detects and classifies abdominal trauma using an EfficientNet-based model. "
        "The dataset consists of CT scan images providing ground truth labels for injuries. It identifies potential injuries such as Bowel Injury, Extravasation Injury, Kidney Injury (Healthy, Low, High), Liver Injury (Healthy, Low, High), and Spleen Injury (Healthy, Low, High)."
    )
    return result, description

def infer_kidney_diseases(image_path):
    model_path = r"models/model/Kidney_Diseases_Classfication_Model.pth"
    classifier = KidneyDiseasesClassification(model_path=model_path)
    result = classifier.infer(image_path)
    description = (
        "Classifies various kidney diseases based on the provided medical image. "
        "The dataset was collected from hospital-based PACS focusing on kidney-related diagnoses such as tumor, cyst, normal, or stone."
    )
    return result, description

def infer_chest_xray(image_path):
    model_path = r"models/model/chest_xray.pth"
    classifier = ChestXRay(model_path=model_path)
    result = classifier.infer(image_path)
    description = (
        "Analyzes chest X-rays to detect abnormalities such as infections or diseases. "
        "The dataset focuses on pneumonia, an infection inflaming the air sacs in the lungs."
    )
    return result, description

def infer_bone_fractures(image_path):
    model_path = r"models/model/Bone_Fracture_Binary_Classification.pth"
    classifier = BoneFractures(model_path=model_path)
    result = classifier.infer(image_path)
    description = (
        "Identifies bone fractures using binary classification. "
        "The dataset includes fractured and non-fractured X-ray images from all anatomical body regions."
    )
    return result, description

def infer_knee_osteoporosis(image_path):
    model_weights_path = r"models/model/Knee_model_weights.pth"
    classifier = Knee_Osteoporosis()
    result = classifier.inf(model_weights_path, image_path)
    description = (
        "Classifies knee images to identify osteoporosis. "
        "The dataset is categorized into Normal, Osteopenia, and Osteoporosis."
    )
    return result, description

# --- Report Generation Function ---

def generate_radiology_report(classifier_description, classifier_outputs, patient_name, patient_age, patient_gender, date):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp")
    template = """
    You are an expert radiology report generator specializing in X-ray interpretations. 
    Using the input provided below, create a detailed and professional X-ray report that includes:
    1. A description of the classifier results.
    2. Relevant medical recommendations based on the classifier outputs.
    3. If the condition is negative (indicating a disease), include a note about potential complications or outcomes if the disease is not treated.

    Input details:
    - Classifier Function: {classifier_description}
    - Classifier Results: {classifier_outputs}
    - Patient Name: {patient_name}
    - Patient Age: {patient_age}
    - Patient Gender: {patient_gender}
    - Date: {date}

    Generate the report with a clear structure and provide the necessary medical insights.

    Expected Report Structure:
    - **Patient Information**: Include name, age, gender, and date.
    - **Findings**: Summarize the classifier results.
    - **Impression**: Provide a concise interpretation of the findings.
    - **Recommendations**: Suggest only 2 next steps or treatments.
    - **Warnings**: If the condition is negative, highlight what could happen if untreated in short.
    """
    prompt = PromptTemplate(
        input_variables=[
            "classifier_description",
            "classifier_outputs",
            "patient_name",
            "patient_age",
            "patient_gender",
            "date"
        ],
        template=template
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    report = chain.run(
        classifier_description=classifier_description,
        classifier_outputs=classifier_outputs,
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        date=date
    )
    return report

@csrf_exempt  # For demonstration; ensure proper CSRF handling in production!
def generate_report_view(request):
    try:
        if request.method == 'GET':
            # Render the HTML form for the user
            return render(request, 'patient/checkdisease/report_generator.html')
    
        elif request.method == 'POST':
            # Process the POST request
            classifier = request.POST.get("classifier")
            patient_name = request.POST.get("patient_name")
            patient_age = request.POST.get("patient_age")
            patient_gender = request.POST.get("patient_gender")
            date = request.POST.get("date")
            uploaded_file = request.FILES.get("xray_image")

            # Validate input
            if not all([classifier, patient_name, patient_age, patient_gender, date, uploaded_file]):
                return JsonResponse({"error": "Please complete all fields and upload an image."}, status=400)

            # Save the uploaded image temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            for chunk in uploaded_file.chunks():
                temp_file.write(chunk)
            temp_file.close()

            # Map classifier to the appropriate function
            classifier_options = {
                "Abdominal Trauma": infer_abdominal_trauma,
                "Kidney Diseases": infer_kidney_diseases,
                "Chest X-ray": infer_chest_xray,
                "Bone Fractures": infer_bone_fractures,
                "Knee Osteoporosis": infer_knee_osteoporosis
            }
            infer_function = classifier_options.get(classifier)
            if not infer_function:
                return JsonResponse({"error": "Invalid classifier selected."}, status=400)

            # Run inference and generate the report
            result, description = infer_function(temp_file.name)
            report = generate_radiology_report(
                classifier_description=description,
                classifier_outputs=result,
                patient_name=patient_name,
                patient_age=patient_age,
                patient_gender=patient_gender,
                date=date
            )

            # Clean the report text (e.g., remove markdown formatting)
            cleaned_report = re.sub(r'\*+', '', report)
            return JsonResponse({"report": cleaned_report})
    
        else:
            return JsonResponse({"error": "Unsupported request method."}, status=405)
    
    except Exception as e:
        # Log the full traceback to your server logs for debugging
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)
#loading trained_model
import joblib as jb
model = jb.load('trained_model')


def home(request):

  if request.method == 'GET':
        
      if request.user.is_authenticated:
        return render(request,'homepage/index.html')

      else :
        return render(request,'homepage/index.html')

      

def admin_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        auser = request.user
        Feedbackobj = Feedback.objects.all()

        return render(request,'admin/admin_ui/admin_ui.html' , {"auser":auser,"Feedback":Feedbackobj})

      else :
        return redirect('home')
      
    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')


def patient_ui(request):

    if request.method == 'GET':

      if request.user.is_authenticated:

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)

        return render(request,'patient/patient_ui/profile.html' , {"puser":puser})

      else :
        return redirect('home')



    if request.method == 'POST':

       return render(request,'patient/patient_ui/profile.html')

       


def pviewprofile(request, patientusername):

    if request.method == 'GET':

          puser = User.objects.get(username=patientusername)

          return render(request,'patient/view_profile/view_profile.html', {"puser":puser})




def checkdisease(request):

  diseaselist=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes ',
  'Gastroenteritis','Bronchial Asthma','Hypertension ','Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)',
  'Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D',
  'Hepatitis E', 'Alcoholic hepatitis','Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
  'Heart attack', 'Varicose veins','Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
  'Arthritis', '(vertigo) Paroymsal  Positional Vertigo','Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']


  symptomslist=['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
  'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition','spotting_ urination',
  'fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss','restlessness','lethargy',
  'patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes','breathlessness','sweating',
  'dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea','loss_of_appetite','pain_behind_the_eyes',
  'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
  'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
  'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
  'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
  'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
  'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
  'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
  'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
  'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
  'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
  'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
  'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
  'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
  'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
  'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
  'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
  'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
  'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
  'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
  'yellow_crust_ooze']

  alphabaticsymptomslist = sorted(symptomslist)

  


  if request.method == 'GET':
    
     return render(request,'patient/checkdisease/checkdisease.html', {"list2":alphabaticsymptomslist})

  elif request.method == 'POST':
       
      ## access you data by playing around with the request.POST object
      
      inputno = int(request.POST["noofsym"])
      print(inputno)
      if (inputno == 0 ) :
          return JsonResponse({'predicteddisease': "none",'confidencescore': 0 })
  
      else :

        psymptoms = []
        psymptoms = request.POST.getlist("symptoms[]")
       
        print(psymptoms)

      
        """      #main code start from here...
        """
      

      
        testingsymptoms = []
        #append zero in all coloumn fields...
        for x in range(0, len(symptomslist)):
          testingsymptoms.append(0)


        #update 1 where symptoms gets matched...
        for k in range(0, len(symptomslist)):

          for z in psymptoms:
              if (z == symptomslist[k]):
                  testingsymptoms[k] = 1


        inputtest = [testingsymptoms]

        print(inputtest)
      

        predicted = model.predict(inputtest)
        print("predicted disease is : ")
        print(predicted)

        y_pred_2 = model.predict_proba(inputtest)
        confidencescore=y_pred_2.max() * 100
        print(" confidence score of : = {0} ".format(confidencescore))

        confidencescore = format(confidencescore, '.0f')
        predicted_disease = predicted[0]

        

        #consult_doctor codes----------

        #   doctor_specialization = ["Rheumatologist","Cardiologist","ENT specialist","Orthopedist","Neurologist",
        #                             "Allergist/Immunologist","Urologist","Dermatologist","Gastroenterologist"]
        

        Rheumatologist = [  'Osteoarthristis','Arthritis']
       
        Cardiologist = [ 'Heart attack','Bronchial Asthma','Hypertension ']
       
        ENT_specialist = ['(vertigo) Paroymsal  Positional Vertigo','Hypothyroidism' ]

        Orthopedist = []

        Neurologist = ['Varicose veins','Paralysis (brain hemorrhage)','Migraine','Cervical spondylosis']

        Allergist_Immunologist = ['Allergy','Pneumonia',
        'AIDS','Common Cold','Tuberculosis','Malaria','Dengue','Typhoid']

        Urologist = [ 'Urinary tract infection',
         'Dimorphic hemmorhoids(piles)']

        Dermatologist = [  'Acne','Chicken pox','Fungal infection','Psoriasis','Impetigo']

        Gastroenterologist = ['Peptic ulcer diseae', 'GERD','Chronic cholestasis','Drug Reaction','Gastroenteritis','Hepatitis E',
        'Alcoholic hepatitis','Jaundice','hepatitis A',
         'Hepatitis B', 'Hepatitis C', 'Hepatitis D','Diabetes ','Hypoglycemia']
         
        if predicted_disease in Rheumatologist :
           consultdoctor = "Rheumatologist"
           
        if predicted_disease in Cardiologist :
           consultdoctor = "Cardiologist"
           

        elif predicted_disease in ENT_specialist :
           consultdoctor = "ENT specialist"
     
        elif predicted_disease in Orthopedist :
           consultdoctor = "Orthopedist"
     
        elif predicted_disease in Neurologist :
           consultdoctor = "Neurologist"
     
        elif predicted_disease in Allergist_Immunologist :
           consultdoctor = "Allergist/Immunologist"
     
        elif predicted_disease in Urologist :
           consultdoctor = "Urologist"
     
        elif predicted_disease in Dermatologist :
           consultdoctor = "Dermatologist"
     
        elif predicted_disease in Gastroenterologist :
           consultdoctor = "Gastroenterologist"
     
        else :
           consultdoctor = "other"


        request.session['doctortype'] = consultdoctor 

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
     

        #saving to database.....................

        patient = puser.patient
        diseasename = predicted_disease
        no_of_symp = inputno
        symptomsname = psymptoms
        confidence = confidencescore

        diseaseinfo_new = diseaseinfo(patient=patient,diseasename=diseasename,no_of_symp=no_of_symp,symptomsname=symptomsname,confidence=confidence,consultdoctor=consultdoctor)
        diseaseinfo_new.save()
        

        request.session['diseaseinfo_id'] = diseaseinfo_new.id

        print("disease record saved sucessfully.............................")

        return JsonResponse({'predicteddisease': predicted_disease ,'confidencescore':confidencescore , "consultdoctor": consultdoctor})
   

def pconsultation_history(request):

    if request.method == 'GET':

      patientusername = request.session['patientusername']
      puser = User.objects.get(username=patientusername)
      patient_obj = puser.patient
        
      consultationnew = consultation.objects.filter(patient = patient_obj)
      
    
      return render(request,'patient/consultation_history/consultation_history.html',{"consultation":consultationnew})


def dconsultation_history(request):

    if request.method == 'GET':

      doctorusername = request.session['doctorusername']
      duser = User.objects.get(username=doctorusername)
      doctor_obj = duser.doctor
        
      consultationnew = consultation.objects.filter(doctor = doctor_obj)
      
    
      return render(request,'doctor/consultation_history/consultation_history.html',{"consultation":consultationnew})



def doctor_ui(request):

    if request.method == 'GET':

      doctorid = request.session['doctorusername']
      duser = User.objects.get(username=doctorid)

    
      return render(request,'doctor/doctor_ui/profile.html',{"duser":duser})



      


def dviewprofile(request, doctorusername):

    if request.method == 'GET':

         
         duser = User.objects.get(username=doctorusername)
         r = rating_review.objects.filter(doctor=duser.doctor)
       
         return render(request,'doctor/view_profile/view_profile.html', {"duser":duser, "rate":r} )








       
def  consult_a_doctor(request):


    if request.method == 'GET':

        
        doctortype = request.session['doctortype']
        print(doctortype)
        dobj = doctor.objects.all()
        #dobj = doctor.objects.filter(specialization=doctortype)


        return render(request,'patient/consult_a_doctor/consult_a_doctor.html',{"dobj":dobj})

   


def  make_consultation(request, doctorusername):

    if request.method == 'POST':
       

        patientusername = request.session['patientusername']
        puser = User.objects.get(username=patientusername)
        patient_obj = puser.patient
        
        
        #doctorusername = request.session['doctorusername']
        duser = User.objects.get(username=doctorusername)
        doctor_obj = duser.doctor
        request.session['doctorusername'] = doctorusername


        diseaseinfo_id = request.session['diseaseinfo_id']
        diseaseinfo_obj = diseaseinfo.objects.get(id=diseaseinfo_id)

        consultation_date = date.today()
        status = "active"
        
        consultation_new = consultation( patient=patient_obj, doctor=doctor_obj, diseaseinfo=diseaseinfo_obj, consultation_date=consultation_date,status=status)
        consultation_new.save()

        request.session['consultation_id'] = consultation_new.id

        print("consultation record is saved sucessfully.............................")

         
        return redirect('consultationview',consultation_new.id)



def  consultationview(request,consultation_id):
   
    if request.method == 'GET':

   
      request.session['consultation_id'] = consultation_id
      consultation_obj = consultation.objects.get(id=consultation_id)

      return render(request,'consultation/consultation.html', {"consultation":consultation_obj })

   #  if request.method == 'POST':
   #    return render(request,'consultation/consultation.html' )





def rate_review(request,consultation_id):
   if request.method == "POST":
         
         consultation_obj = consultation.objects.get(id=consultation_id)
         patient = consultation_obj.patient
         doctor1 = consultation_obj.doctor
         rating = request.POST.get('rating')
         review = request.POST.get('review')

         rating_obj = rating_review(patient=patient,doctor=doctor1,rating=rating,review=review)
         rating_obj.save()

         rate = int(rating_obj.rating_is)
         doctor.objects.filter(pk=doctor1).update(rating=rate)
         

         return redirect('consultationview',consultation_id)





def close_consultation(request,consultation_id):
   if request.method == "POST":
         
         consultation.objects.filter(pk=consultation_id).update(status="closed")
         
         return redirect('home')






#-----------------------------chatting system ---------------------------------------------------


def post(request):
    if request.method == "POST":
        msg = request.POST.get('msgbox', None)

        consultation_id = request.session['consultation_id'] 
        consultation_obj = consultation.objects.get(id=consultation_id)

        c = Chat(consultation_id=consultation_obj,sender=request.user, message=msg)

        #msg = c.user.username+": "+msg

        if msg != '':            
            c.save()
            print("msg saved"+ msg )
            return JsonResponse({ 'msg': msg })
    else:
        return HttpResponse('Request must be POST.')



def chat_messages(request):
   if request.method == "GET":

         consultation_id = request.session['consultation_id'] 

         c = Chat.objects.filter(consultation_id=consultation_id)
         return render(request, 'consultation/chat_body.html', {'chat': c})


#-----------------------------chatting system ---------------------------------------------------


