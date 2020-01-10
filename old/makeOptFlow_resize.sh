# Specify the path to the optical flow utility here.
# Also check line 44 and 47 whether the arguments are in the correct order.
flowCommandLine="bash run-deepflow.sh"

if [ -z "$flowCommandLine" ]; then
  echo "Please open makeOptFlow.sh and specify the command line for computing the optical flow."
  exit 1
fi

if [ ! -f ./consistencyChecker/consistencyChecker ]; then
  if [ ! -f ./consistencyChecker/Makefile ]; then
    echo "Consistency checker makefile not found."
    exit 1
  fi
  cd consistencyChecker/
  make
  cd ..
fi

filePattern=$1
folderName=$2
startFrame=${3:-1}
stepSize=${4:-1}
processsize=$5
finalsize=$6

if [ "$#" -le 1 ]; then
   echo "Usage: ./makeOptFlow <filePattern> <outputFolder> [<startNumber> [<stepSize>]] processsize finalsize"
   echo -e "\tfilePattern:\tFilename pattern of the frames of the videos."
   echo -e "\toutputFolder:\tOutput folder."
   echo -e "\tstartNumber:\tThe index of the first frame. Default: 1"
   echo -e "\tstepSize:\tThe step size to create long-term flow. Default: 1"
   echo -e "\tprocesssize:\tx resolution to process flow"
   echo -e "\tfinalsize:\tx final resolution to resize flow"
   exit 1
fi

motionratio=$(echo "scale=25;$finalsize/$processsize" | bc)
echo -e "motionratio = $motionratio"
    
i=$[$startFrame]
j=$[$startFrame + $stepSize]

mkdir -p "${folderName}"
mkdir -p "${folderName}/tmp_$$"

while true; do
  file1=$(printf "$filePattern" "$i")
  file2=$(printf "$filePattern" "$j")
  if [ -a $file2 ]; then
    #resize file1 and file2
    echo -e "\033[1;32m--> resizing $file1 $file2 [x resolution : $processsize]\033[0m"
    gmic -i $file1 -resize2dx $processsize -o ${folderName}/tmp_$$/file1.png
    gmic -i $file2 -resize2dx $processsize -o ${folderName}/tmp_$$/file2.png
    if [ ! -f ${folderName}/forward_${i}_${j}.flo ]; then
      echo -e "\033[1;32m--> process forward flow [x resolution : $processsize]\033[0m"
      eval $flowCommandLine "${folderName}/tmp_$$/file1.png" "${folderName}/tmp_$$/file2.png" "${folderName}/tmp_$$/forward.flo"
      #eval $flowCommandLine "$file1" "$file2" "${folderName}/forward_${i}_${j}.flo"
    fi
    if [ ! -f ${folderName}/backward_${j}_${i}.flo ]; then
      echo -e "\033[1;32m--> process backward flow [x resolution : $processsize]\033[0m"
      eval $flowCommandLine "${folderName}/tmp_$$/file2.png" "${folderName}/tmp_$$/file1.png" "${folderName}/tmp_$$/backward.flo"
      #eval $flowCommandLine "$file2" "$file1" "${folderName}/backward_${j}_${i}.flo"
    fi
    #convert flo to exr
    echo -e "\033[1;32m--> convert flo to exr\033[0m"
    ./flo2exr ${folderName}/tmp_$$/forward.flo ${folderName}/tmp_$$/forward.exr
    ./flo2exr ${folderName}/tmp_$$/backward.flo ${folderName}/tmp_$$/backward.exr
    #resize exr
    echo -e "\033[1;32m--> resize exr [x resolution : $finalsize]\033[0m"
    gmic -i ${folderName}/tmp_$$/forward.exr -resize2dx $finalsize -mul $motionratio -o ${folderName}/tmp_$$/forward.exr
    gmic -i ${folderName}/tmp_$$/backward.exr -resize2dx $finalsize -mul $motionratio -o ${folderName}/tmp_$$/backward.exr
    #convert resized exr to flo
    echo -e "\033[1;32m--> writing final flow file [x resolution : $finalsize]\033[0m"
    ./exr2flo ${folderName}/tmp_$$/forward.exr ${folderName}/forward_${i}_${j}.flo
    ./exr2flo ${folderName}/tmp_$$/backward.exr ${folderName}/backward_${j}_${i}.flo
    #consistency check
    echo -e "\033[1;32m--> process consistency check files\033[0m"
    ./consistencyChecker/consistencyChecker "${folderName}/backward_${j}_${i}.flo" "${folderName}/forward_${i}_${j}.flo" "${folderName}/reliable_${j}_${i}.pgm"
    ./consistencyChecker/consistencyChecker "${folderName}/forward_${i}_${j}.flo" "${folderName}/backward_${j}_${i}.flo" "${folderName}/reliable_${i}_${j}.pgm"
  else
    break
  fi
  i=$[$i +1]
  j=$[$j +1]
done
