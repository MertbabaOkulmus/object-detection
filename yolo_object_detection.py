
#Gerekli eklentileri import edilmesi

import cv2 #OpenCV nesnesi 
import numpy as np#numpy nesnesi
import glob
import random


#Yolo'un yüklenmesi
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

#Özel nesnenin adı
classes = ["gemi"]

# Resimlerin dizini(yolu)
images_path = glob.glob(r"C:\Users\Gernas\Desktop\images\*.jpg")


#net isimli nestwork'ümüzden, net.getLayerNames() komutunu kullanarak 
#layer name'lerimizi alıyoruz.
layer_names = net.getLayerNames()
#Nesne isimleri nöron network tarafından bize verilen çıktı(gemi,araba, vs.)
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
#Herbir nesnenin adını farklı bir renk atamak için kullanılımış.
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Buraya resimlerinizin yolunu ekleniyor
random.shuffle(images_path)
# Tüm görüntüler arasında döngü
for img_path in images_path:
    # Görüntülerin yüklenmesi
    img = cv2.imread(img_path)
    #Çok büyük resim olduğunda ekranda gözükmüyor bu yüzden resize komutunu kullanıyoruz
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape#resmimiz genişliğini ve 

    #Nesnelerin algılanması
    #yolo bu dnn(deep noron network)'i kullanabilmesi için blob denilen
    #özel bir formata dönüştürülmesi gerekiyor. image'tan blob'a dönüştürüyoruz.
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    #resmin ham hali bizim inputumuzdur.
    net.setInput(blob)
    #çıktı olarak aldığımız resmin içindeki nesnenin başlangıç noktasıdır
    outs = net.forward(output_layers)
    
    # Bilgileri ekranda gösterme
    class_ids = []#class_ids için bir array
    confidences = []#confidences(0-1) ne kadar güvenle doğruluk bilgisi için bir array
    boxes = []#resimde gördüğü nesnelerin kutuları için de bir array oluşturuyoruz.
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Nesne algılandığında 
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Dikdörtgenin koordinatları
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                #dikdörtgenimizi,
                boxes.append([x, y, w, h])
                #güven aralığı değerin ve 
                confidences.append(float(confidence))
                #nesnenin adını append ediyoruz.
                class_ids.append(class_id)
    #Aynı şeyin birden fazla seçilmesinin önüne geçiyor.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    
    #yazı fontu tanımlıyoruz
    font = cv2.FONT_HERSHEY_PLAIN
    
    for i in range(len(boxes)):#kaç tane nesne yani box algıladın
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            #dikdörgeni çidiriyoruz 
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            #sınıf ismini yazdırıyoruz
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

    #tüm bu işlemler bittikten sonra bur resmi gösteriyoruz 
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
#kapattığımızda bütün pencerlerin kapatılmasını sağlıyoruz
cv2.destroyAllWindows()
