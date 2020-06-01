_median: Noise reduced by alignment and median combination of bursts as follows:

        # First removed blurry photos by eye, resulting in usable photos listed in ListOfPhotos.txt

        # Loop over each burst set (of remaining non-blurry photos)
	for i in 045 046 047 048 086 087 124 125 126 127 236 237 238 239 240; do

            # Crop out the top row of drone watermark
            for f in B${i}*.JPG; do convert $f -crop 4000x2750+0+250 ${f/B/C}; done

            # Align the images
            align_image_stack -c 20 -v --corr=0.5 -m -a A${i} C${i}*.JPG
            
            # Combine images taking the median pixel value
            convert A${i}* -evaluate-sequence median ${i}.jpg

        done

Note: Potentially some systematic shift during labeling, due to training of brain neural net (i.e. discriminating bolt reflections or transmission through glass/acrylic).

Update May 20, 2020:

	1) Nick's reconstruction found mistakes in the labeling:
		https://github.com/nickwp/PhotogrammetryAnalysis/commit/02ce86036703daf40141d5121b3f234f71b2b015
	The text files here have been updated accordingly.

	2) After comparing fitted camera depths to drone sensor, it was found that images 12*.JPG were labeled incorrectly, around light injector B2 instead of B1. These files have been moved to an "Mislabeled" directory.
