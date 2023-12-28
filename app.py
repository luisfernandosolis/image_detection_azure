from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from dotenv import load_dotenv
import cv2
def main():
    
    try:
        load_dotenv()
        prediction_endpoint=os.getenv("PredictionEndpoint")
        prediction_key=os.getenv("PredictionKey")
        project_id=os.getenv("ProjectID")
        model_name=os.getenv("Modelname")

        ## autenticacion del cliente for the training api
        credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        prediction_client = CustomVisionPredictionClient(prediction_endpoint, credentials)


        ##load image and get height, with, and channels

        image_file="orange_95.jpg"
        print("Detecting object in", image_file)


        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    

        h,w,ch =np.array(img).shape

        ## Detect objects in the test image

        with open(image_file,mode="rb") as image_data:
            results=prediction_client.detect_image(project_id,model_name,image_data)
        
        # create figure
        fig=plt.figure(figsize=(8,8))
        plt.axis("off")

        for prediction in results.predictions:
            if(prediction.probability*100) > 50:
                left=prediction.bounding_box.left*w #x0
                top=prediction.bounding_box.top*h  #y0

                width=prediction.bounding_box.width*w  #x1
                height=prediction.bounding_box.height*h #y1

                

                #draw the box

                #points=((left,top), (left+width, top), (left+width, top+height), (left+width, top+height))

                #draw.line(points, fill=color, width=linewith)

                start_point = (int(left), int(top))
                end_point = (int(width), int(height))
                cv2.rectangle(img, start_point, end_point, color=(255,0,0), thickness=5)
                cv2.putText(
                    img,
                    f"{prediction.tag_name} : {round(prediction.probability*100,2)}%",
                    (start_point[0],start_point[1]+20),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale = 0.7,
                    color = (0, 0, 255) ,
                    thickness=2
                )




                ## add the tag name and probability
                print(prediction.tag_name)
                print(round(prediction.probability*100,2))
                #plt.annotate(f"{prediction.tag_name} : {round(prediction.probability*100,2)}")
        
        plt.imshow(img)
        outputfile="output.jpg"
        fig.savefig(outputfile)
        print("result saved in 10", outputfile)

    except Exception as ex:
        print(ex)


if __name__=="__main__":
    main()