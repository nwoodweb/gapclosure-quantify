# gapclosure-quantify

Gap closure assays are rudimentary ways of assessing cell migration.
This is accomplished by growing a confluent monolayer of mammalian
cells on a substrate such that the entire surface is covered with cells.
The monolayer is then abraded and changes in the gap size as cells
migrate are then measured.

This respository details two methods of automating gap closure quantification:
preprocessing followed by Otsu's Method, then a neural network based approach. 
The neural network based approach has advantages over Otus's method in several
cases, including the presence of cell debris, poor or uneven illumination,
or when the gap is nearing full closure.

Neural network models can be found on [Hugging Face](https://huggingface.co/nwoodweb/gapclosure-quantify-unet)
