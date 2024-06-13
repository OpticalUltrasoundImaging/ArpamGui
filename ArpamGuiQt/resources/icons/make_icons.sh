IMAGE=249_PAUS_314.png

# Create a 1000x1000 mask with 150 pixels of rounded corners
MASK=mask.png
magick -size 1000x1000 xc:none -draw "roundrectangle 0,0,1000,1000,500,500" $MASK
IMAGE_ROUNDED=rounded.png
magick $IMAGE -alpha Set $MASK -compose DstIn -composite $IMAGE_ROUNDED 

# Make Windows icon file
DIR=ArpamGui.ico
mkdir $DIR
magick $IMAGE_ROUNDED -resize 16x16 $DIR/icon-16.png
magick $IMAGE_ROUNDED -resize 32x32 $DIR/icon-32.png
magick $IMAGE_ROUNDED -resize 256x256 $DIR/icon-256.png
magick $DIR/icon-16.png $DIR/icon-32.png $DIR/icon-256.png icon.ico

### Make mac icon file
DIR=ArpamGui.iconset
mkdir $DIR
magick $IMAGE_ROUNDED -resize 16x16 $DIR/icon_16x16.png
magick $IMAGE_ROUNDED -resize 32x32 $DIR/icon_16x16@2x.png
magick $IMAGE_ROUNDED -resize 32x32 $DIR/icon_32x32.png
magick $IMAGE_ROUNDED -resize 64x64 $DIR/icon_64x64@2x.png
magick $IMAGE_ROUNDED -resize 128x128 $DIR/icon_128x128.png
magick $IMAGE_ROUNDED -resize 256x256 $DIR/icon_128x128@2x.png
magick $IMAGE_ROUNDED -resize 256x256 $DIR/icon_256x256@.png
magick $IMAGE_ROUNDED -resize 512x512 $DIR/icon_256x256@2x.png
magick $IMAGE_ROUNDED -resize 512x512 $DIR/icon_512x512.png
iconutil -c icns $DIR
