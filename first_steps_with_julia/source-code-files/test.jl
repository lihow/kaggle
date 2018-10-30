
imageSize = 400 # 20 x 20 pixel

path_proj = "/home/erick/Documents/LancsPC/Kaggle/SpiPeruStats/" *
"FirstStepsWithJulia"
#Set location of data files, folders
path_data = "$(path_proj)/1.Data/1.RowData"
path_data 
#Read information about training data , IDs.
labelsInfoTrain = readtable("$(path_data)/trainLabels.csv")
typeof(labelsInfoTrain)
names(labelsInfoTrain) 
head(labelsInfoTrain)

#Read training matrix
xTrain = read_data("train", labelsInfoTrain, imageSize, path)

#Read information about test data ( IDs ).
labelsInfoTest = readtable("$(path)/sampleSubmission.csv")

#Read test matrix
xTest = read_data("test", labelsInfoTest, imageSize, path)


typeData = "train"
labelsInfo = labelsInfoTrain;

function read_data(typeData, labelsInfo, imageSize, path)
 #Intialize x matrix
 x = zeros(size(labelsInfo, 1), imageSize) # n images by image size
 
 for (index, idImage) in enumerate(labelsInfo[:ID])
   index = 1
   idImage = labelsInfoTrain[1, :ID]
   #Read image file 
   nameFile = "$(path_data)/$(typeData)Resized/$(idImage).Bmp"
   # img = imread(nameFile) # deprecated
   img = load(nameFile)
   temp = convert(Image{Images.Gray}, img) 
# temp = temp0 
   #Convert img to float values 
   # temp = float32sc(img) # deprecated
   # temp = float32(img) # changes image into real values
 
   #Convert color images to gray images
   #by taking the average of the color scales. 
   # if ndims(temp) == 3
    # temp = mean(temp.data, 1)
   # end
     
   #Transform image matrix to a vector and store 
   #it in data matrix 
   x[index, :] = reshape(temp, 1, imageSize)
  end 
 return x
end
