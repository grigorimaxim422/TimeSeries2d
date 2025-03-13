#!/bin/bash
docker system prune -a --volumes
#!/bin/bash
rm -rf btcpredict2/
VERSION=$1  # Pass version as an argument
REPO="grigorimaxim/btcpredict"  # Change this to your repository

if [ -z "$VERSION" ]; then
  echo "Usage: $0 <version>"
  exit 1
fi

# Build, tag, and push the image
docker build -t $REPO:$VERSION .
docker push $REPO:$VERSION


docker save $REPO:$VERSION | gzip > btcpredict2.tar.gz
echo "Saved into btcpredict2.tar.gz"

# Optionally tag as latest
docker tag $REPO:$VERSION $REPO:latest
docker push $REPO:latest

echo "Image $REPO:$VERSION pushed successfully!"

echo "Prepare uploading big file to hub..."
# huggingface-cli repo create btcpredict

# git lfs install
huggingface-cli upload grigorimaxim/btcpredict ./btcpredict2.tar.gz

huggingface-cli repo create btcpredict2 #if not created yet

git clone https://huggingface.co/grigorimaxim/btcpredict2
git config user.name grigorimaxim
git config user.email grigorimaxim422@gmail.com
cd btcpredict2
git lfs install
git lfs track "*.bin"  # Track specific file types
git lfs track "*.gz"  # Track specific file types
git add .gitattributes  # Ensure LFS settings are committed
git add btcpredict2.tar.gz
git commit -m "upload bigger"
git push
# git clone https://huggingface.co/grigorimaxim/parler58
# cd parler58
# sudo apt install git-lfs
# git lfs install
# mv ../parler58.tar.gz ./parler58dock.image
# git lfs track "*.image"

# git add .
# git commit -m 'upload large file"
# git push origin main
