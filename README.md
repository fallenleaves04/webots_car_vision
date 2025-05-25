# webots_car_vision
## Funkcjonalność projektu

 W danym projekcie realizuje się system percepcji samochodu autonomicznego, zawierający obsługę czujników ultradzwiękowych oraz układ z 6 kamer widoku 360°. 

 Kamery pozwalają stworzyć widok z lotu ptaka.

 Czujniki ultradzwiękowe pomagają w lokalizacji pojazdu względem najbliższych przeszkód. Również sporządzono wizualizację strefy detekcji.

 Ponadto w planach jest realizacja algorytmu automatycznego parkowania samochodu przy wyznaczonej stronie jezdni.

 Segmentacja wizualna środowiska za pomocą sieci YOLO.

 Optymalizacja obróbki graficznej za pomocą bibliotek z obsługą jąder CUDA na platformie Windows 10 22H2 Home (OpenCV oraz PyTorch) - linki do instalacji bibliotek w źródłach, a wymagania są w głównej gałęzi.

 Repo zawiera pliki kontrolera oraz niezbędne pliki z macierzami przekształceń homograficnzych o rozszerzeniu .npy.
 
## Zasoby programistyczne
https://pytorch.org/get-started/locally/#start-locally

https://github.com/cudawarped/opencv-python-cuda-wheels/releases

https://github.com/360ls/stitcher/blob/master/app/stitcher/core/stitcher.py#L79

## Zasoby dydaktyczne
https://github.com/AtsushiSakai/PythonRobotics.git

https://en.wikipedia.org/wiki/Tesla_Autopilot_hardware

https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

https://research.google/blog/seamless-google-street-view-panoramas/

https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec07_panoramas.pdf

