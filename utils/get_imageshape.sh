#!/bin/sh

for file in ./*.tif; do
	magick identify -format "%w,%h";
done 
