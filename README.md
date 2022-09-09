# M2-internship-Automatic-detection-of-biomarkers-of-osteoporosis-using-biplane-radiographs

Relevant files employed to extract features from raw data and train several Machine Learning algorithms and Deep Learning networks. The project was carried out locally on a Windows PC using Jupyter notebook and in a remote Ubuntu server to access GPUs. Hence, the project files are divided in two folders: "Local" and "Server".

## List of files
### Local files:
 - __DRR_Random_Forest.ipynb__: trains a Random Forest model, computes its confusion matrix for the 10 cross validation folds and calculates its performance metrics.
 - __DRR_SVM.ipynb__: trains a SVM model, computes its confusion matrix for the 10 cross validation folds and calculates its performance metrics.
 - __Feature_extraction.ipynb__: converts the dicom images to png, preprocesses the images using histogram equalization and then extracts all the different features, saving them into individual Excel files for each type. It can work both with the raw DRR images or with a csv containing the filenames of the cropped ROI.
 - __DRR_RF_feature_importance.ipynb__: enables the visualization and saving of the feature importance of Random Forest models saved as .sav.

### Server files:
 - __DRR_EfficientNet-B7_CV.py__: trains an EfficientNet-B7 model (optional finetuning of all layers of the last 3 layers included) using transfer learning and cross validation and saves its different performance parameters.
 - __DRR_RF_multiclass.py__: trains a multiclass Random Forest model.
 - __DRR_RF_regressor.py__: trains a Random Forest regressor to predict trabecular aand total Bone Mineral Densitym as well as uniaxial and anterior vertebral strengths.
 - __DRR_ResNet-152_CV.py__: trains a ResNet-152 model (optional finetuning of all layers of the last 3 layers included) using transfer learning and cross validation and saves its different performance parameters.
 - __vertebral_height_angle_computation.py__:  computes vertebral height and plate angles from DRR images and the masks for the corresponding vertebrae.

## Intership summary

Osteoporosis is a bone disease that occurs when bone mineral density and bone mass decrease. This condition can lead to a diminution of bone strength, increasing the risk of low energy fractures, and affecting the quality of life of patients, augmenting their risk of morbidity and mortality. The clinical routine to diagnose this disease is dual-energy X-ray absorptiometry (DXA), an imaging technique that employs two X-ray beams at different energy levels to measure areal Bone Mineral Density (BMD). Despite the advantages of this technique (such as its wide availability, low cost, and low radiation dose), it also has important drawbacks, such as its insufficient sensitivity when predicting osteoporotic fractures (50%). Moreover, since osteoporosis is a disease with silent symptoms, a DXA exam is usually prescribed at the advanced stages of osteoporosis, after an osteoporotic fracture has already occurred. For these reasons, there is a need for an alternative diagnostic method that can be employed on routine exams, reducing costs and radiation dose whilst enabling the early diagnosis of this condition.

New advances in Machine Learning (ML) allow the use of a large number of input variables for model development, which could become essential tools in the field of fracture prediction. Several studies have been published on this field employing images obtained using X-ray-based modalities using Trabecular Bone Score (TBS), textural features and geometric and biomechanical variables in combination with demographic, clinical and lifestyle parameters, for tasks such as vertebral fracture risk prediction and osteoporosis diagnosis, obtaining promising results. Notwithstanding, these methods do not exploit all the information that can be extracted from biomedical image and often rely on features extracted from Finite Element (FE) models whose creation process is long, as they are generated manually.

In this internship, several Machine Learning models (Random Forest and Support Vector Machines or SVM) were trained on features extracted using image processing (Initial Slope of the Variogram which is an alternative to TBS, textural and morphologic parameters) from biplanar Digitally Reconstructed Radiographs (DRR) on in vitro vertebrae to classify them according to their trabecular BMD, a biomarker of vertebral fragility. Moreover, the feature importance attributed by the algorithms what analyzed to identify quantitative osteoporosis image biomarkers. Finally, a series of Deep Learning networks (based on EfficientNet-B7 and ResNet-152) were trained as well on the DRR images themselves to compare the performance between traditional radiomics and deep features.

Keywords: Osteoporosis, Bone Mineral Density, Machine Learning, Digitally Reconstructed Radiographs, biplanar radiographs 
