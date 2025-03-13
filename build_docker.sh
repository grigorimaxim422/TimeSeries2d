#!/bin/bash
docker system prune -a --volumes
#!/bin/bash

VERSION=$1  # Pass version as an argument
REPO="grigorimaxim/btcpredict"  # Change this to your repository

if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

# Build, tag, and push the image
docker build -t $REPO:$VERSION .
docker push $REPO:$VERSION


docker save $REPO:$VERSION | gzip > btcpredict.tar.gz
echo "Saved into btcpredict.tar.gz"

# Optionally tag as latest
docker tag $REPO:$VERSION $REPO:latest
docker push $REPO:latest

echo "Image $REPO:$VERSION pushed successfully!"

echo "Prepare uploading big file to hub..."
#huggingface-cli repo create btcpredict

git lfs install
huggingface-cli upload grigorimaxim/merlin_tts ./btcpredict.tar.gz


# git clone https://huggingface.co/grigorimaxim/parler58
# cd parler58
# sudo apt install git-lfs
# git lfs install
# mv ../parler58.tar.gz ./parler58dock.image
# git lfs track "*.image"

# git add .
# git commit -m 'upload large file"
# git push origin main
