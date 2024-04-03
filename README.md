# Real-Time-Object-Detection-Service-Based-on-Mask-R-CNN

### This is Will Wang, Welcome to my Github! 👋
💻 Computer Engineering Student at <a href="https://www.engineering.utoronto.ca/">University of Toronto</a> <br>

### A little more about me and the tech stacks...  

```javascript
const Real-Time-Object-Detection = {
  programming language: [Python,
  frameworks & packages: [NumPy, Pandas, Scikit-learn, PyTorch],
  tools: [SQL, Git, Linux/Unix],
  take_away: "Stay hungry"
}
```

### About the research
Real-Time-Object-Detection-Service-Based-on-Mask-R-CNN is a research Will(Liyourong) led at <a href="https://www.engineering.utoronto.ca/">University of Toronto</a>. The research is focused on the deep-learning based Object Detection. His initial thrust was his previous suffering in driving from GAD. Critical lines of research delve into the depth of the existing object detection industry, adding object segregation to build a Deep-Learning model based on the R-CNN method to detect various classes of objects, classify them, and bind each entity with a rectangle boundary. Will(Liyourong) has also been exposed to machine learning earlier in his career, as shown in <a href="https://dl.acm.org/doi/10.1145/3449301.3449321">Judging Pre-owned-bicycle Deals Type Using Machine Learning Methods</a>. 

These research initiatives enabled Will(Liyourong) to be appointed as the research lead of this research team.

For the source code of the project, please refer to the ji_model_23.py in the same directory. 

### Abstract
With the development of technology, object detection, and segmentation have turned from outlines on paper to pragmatic applications like Automated vehicles (AVs) and have recently become a popular subject in both research and industry domains, implying the societal and industrial revolution. Pedestrian safety is one of the prime targets of advanced driver assistance systems. This project delves into the depth of the existing object detection industry, adding object segregation to build a Deep-Learning model based on the R-CNN method to detect various classes of objects, classify them, and bind each entity with a rectangle boundary. The team has built a Mask R-CNN model that has accuracy as high as 99% and can detect pedestrians and localize them using a bounding box. The report below illustrates what the team has done so far, gives practical step-by-step instructions for any colleague to review, qualitative and quantitative results, and possible further developments upon that as well as Ethical Considerations.


### Motivation & Goal
The research is based on my experience: I possessed a strong enthusiasm for driving and held an FIA silver equivalent racing license. Tragically, I had been suffering from Generalized Anxiety Disorder (GAD) that revolves around the fear of unintentionally hitting pedestrians. This fear is particularly exacerbated during nighttime driving or when facing intense sunshine. My research team aims to create a system that differentiates pedestrians from other traffic attendees, thereby mitigating the anxiety caused by GAD and enabling the return to the driver's seat.

### Illustration
In the research, Mask Region-based Convolutional Neural Networks (Mask R-CNN) are used. The Mask R-CNN method extends from the regular CNN method. This method first sees the input picture and finds all possible region proposals where an object might be, then the R-CNN method will compute each proposal’s CNN features and classify features to a known label using the regular CNN method. After identifying the proposals’ segments, the R-CNN will draw the rectangle bounding polygon, as shown below.
<img width="1495" alt="截屏2024-04-02 下午8 33 47" src="https://github.com/Liyourong/Real-Time-Object-Detection-Service-Based-on-Mask-R-CNN/assets/105031962/e68e5629-37ed-481a-b3c7-4adfa7fd11ca">

### Qualitative Results
After the team had the model trained, the team tested the model on the images. To protect the privacy of identifiable humans in the public database, the team selected 3 pictures from the test set as an example shown in the figure. The model could accurately predict the person in the image and draw the bounding box according to it. The team could easily see the correctness of the model by discerning the correct segmentation for the image. 
<img width="1438" alt="截屏2024-04-02 下午8 35 39" src="https://github.com/Liyourong/Real-Time-Object-Detection-Service-Based-on-Mask-R-CNN/assets/105031962/d4493523-f002-4989-b865-56c52e205b25">

### Model Performance & Explanation
Compared to the baseline model, the primary model achieved a 48.58% increase in the accuracy regarding the accuracy of class prediction, and a 33% time increase in the training process. The increase in the accuracy is attributed to the more complex architecture of the Mask R-CNN model with pixel-wise segmentation to better understand the content of the image. Such ability ensures the Mask R-CNN model acquires more information than RestNet-50. Also, due to the involvement in more context and complex pattern recognition of Mask R-CNN, it exhibits better generalization to new, unseen data, especially data that the team collected under different light and time conditions. The increase in the processing time is mainly attributed to the surge in the requirement for computation per input image. The Mask R-CNN model requires significantly more computation due to the more complex architecture.
